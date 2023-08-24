"""Script to finetune CLM models on chat data. The dataset is expected to containg a column "dialog" that contains a list
of turns (similar to Llama 2's expected format. See https://huggingface.co/datasets/BramVanroy/dutch_chat_datasets
for an example.

Note that the default system prompt is in Dutch. You may need to adapt for your use case.
"""
import dataclasses
import json
import logging
import math
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import datasets
import numpy as np
import torch
import transformers
from datasets import Dataset, DatasetDict, load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from peft.utils import TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING
from transformers import (
    CONFIG_MAPPING,
    MODEL_FOR_CAUSAL_LM_MAPPING,
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
    EarlyStoppingCallback,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils.versions import require_version
from trl.trainer.utils import PeftSavingCallback


logger = logging.getLogger(__name__)


MODEL_CONFIG_CLASSES = list(MODEL_FOR_CAUSAL_LM_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    model_name_or_path: str = field(
        metadata={"help": ("The model checkpoint for weights initialization.")},
    )
    model_type: Optional[str] = field(
        default=None,
        metadata={"help": "If training from scratch, pass a model type from the list: " + ", ".join(MODEL_TYPES)},
    )
    config_overrides: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Override some existing default config settings when a model is trained from scratch. Example: "
                "n_embd=10,resid_pdrop=0.2,scale_attn_weights=false,summary_type=cls_index"
            )
        },
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": (
                "Will use the token generated when running `huggingface-cli login` (necessary to use this script "
                "with private models)."
            )
        },
    )
    max_length: Optional[int] = field(
        default=None,
        metadata={"help": ("Max. sequence length. Data will be truncated up to this length.")},
    )
    torch_dtype: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Override the default `torch.dtype` and load the model under this dtype. If `auto` is passed, the "
                "dtype will be automatically derived from the model's weights."
            ),
            "choices": ["auto", "bfloat16", "float16", "float32"],
        },
    )
    low_cpu_mem_usage: bool = field(
        default=False,
        metadata={
            "help": (
                "It is an option to create the model as an empty shell, then only materialize its parameters when the pretrained weights are loaded."
                "set True will benefit LLM loading time and RAM consumption."
            )
        },
    )
    load_in_4bit: bool = field(
        default=True,
        metadata={
            "help": (
                "This flag is used to enable 4-bit quantization by replacing the Linear layers with FP4/NF4 layers from`bitsandbytes`."
            )
        },
    )
    bnb_4bit_compute_dtype: Optional[str] = field(
        default="float16",
        metadata={
            "help": (
                "This sets the computational type which might be different than the input time. For example, inputs might be fp32, but computation can be set to bf16 for speedups."
            ),
        },
    )
    bnb_4bit_quant_type: Optional[str] = field(
        default="nf4",
        metadata={
            "help": (
                "This sets the quantization data type in the bnb.nn.Linear4Bit layers. Options are FP4 and NF4 data types which are specified by `fp4` or `nf4`."
            ),
            "choices": ["fp4", "nf4"],
        },
    )
    trust_remote_code: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to trust the code that may be downloaded alongside some models. This may be necessary to run models like Falcon who are not fully integrated in `transformers` yet."
            )
        },
    )
    lora_alpha: int = field(
        default=16,
        metadata={"help": ("The alpha parameter for LoRA scaling")},
    )
    lora_dropout: float = field(
        default=0.1,
        metadata={"help": ("The dropout probability for LoRA layers")},
    )
    lora_r: int = field(
        default=64,
        metadata={"help": ("LoRA attention dimension")},
    )
    lora_target_linear: bool = field(
        default=False,
        metadata={
            "help": (
                "The LoRA paper mentions that in addition to attention layers, Linear layers should also be targetted"
                " to achieve the best results. This can lead to a big increase in trainable parameters, though, and"
                " most of the time performance is also quite good if we only target attention layers. So enabling this"
                " option may yield better performance but is bound to train (much) slower."
            )
        },
    )
    use_nested_quant: Optional[bool] = field(
        default=True,
        metadata={"help": "Activate nested quantization for 4bit base models"},
    )
    use_peft: bool = field(default=False, metadata={"help": "Wether to use PEFT or not to train adapters"})
    use_flash_attention: bool = field(
        default=True,
        metadata={
            "help": (
                "Whether to use Flash Attention. This currently requires a GPU with compute capability of 8.0 or higher"
                " (3090/A100 and up). The patch currently only works for Llama models. Requires you to install"
                " flash-attention: https://github.com/Dao-AILab/flash-attention"
            )
        },
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    dialog_column_name: Optional[str] = field(default="dialog", metadata={"help": "Column that contains dialogs."})
    train_file: Optional[str] = field(default=None, metadata={"help": "The input training data file (a text file)."})
    validation_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input evaluation data file to evaluate the perplexity on (a text file)."},
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )
    streaming: bool = field(default=False, metadata={"help": "Enable streaming mode"})

    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    validation_split_percentage: Optional[int] = field(
        default=5,
        metadata={
            "help": "The percentage of the train set used as validation set in case there's no validation split"
        },
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    use_presplit_validation: bool = field(
        default=True,
        metadata={
            "help": "Whether to look for and use a 'validation' split in the given HF dataset. If"
            " disabled, will use 'validation_split_percentage' to turn a portion of"
            " the training set into a validation set"
        },
    )
    early_stopping_patience: Optional[int] = field(
        default=None,
        metadata={
            "help": "Stop training when the evaluation metric worsens (instead of improves) for"
            " early_stopping_patience evaluation calls."
        },
    )
    early_stopping_threshold: Optional[float] = field(
        default=None,
        metadata={"help": "Denote how much the evaluation metric must improve to satisfy early stopping conditions."},
    )
    dataset_batch_size: int = field(
        default=1000,
        metadata={
            "help": "Number of examples per batch provided to function if batched=True. If batch_size <= 0 or "
            "batch_size == None, provide the full dataset as a single batch to function."
        },
    )
    system_prompt: str = field(
        default="Je bent een behulpzame, respectvolle en eerlijke assistent. Antwoord altijd zo behulpzaam mogelijk. Je antwoorden mogen geen schadelijke, onethische, racistische, seksistische, gevaarlijke of illegale inhoud bevatten. Zorg ervoor dat je antwoorden sociaal onbevooroordeeld en positief van aard zijn.\n\nAls een vraag nergens op slaat of feitelijk niet coherent is, leg dan uit waarom in plaats van iets niet correct te antwoorden. Als je het antwoord op een vraag niet weet, deel dan geen onjuiste informatie.",
        metadata={"help": "System prompt to use for training"},
    )

    def __post_init__(self):
        if self.streaming:
            require_version("datasets>=2.0.0", "The streaming feature requires `datasets>=2.0.0`")


class DataCollatorForLlamaChatCompletion(DataCollatorForLanguageModeling):
    def __init__(
        self,
        *args,
        mlm: bool = False,
        ignore_index: int = -100,
        skip_mismatched_sequences: bool = False,
        **kwargs,
    ):
        super().__init__(*args, mlm=mlm, **kwargs)
        self.ignore_index = ignore_index
        self.skip_mismatched_sequences = skip_mismatched_sequences
        self.start_idxs = self.tokenizer("[/INST]", add_special_tokens=False, return_tensors="pt")[
            "input_ids"
        ].squeeze()

    def torch_call(self, examples: List[Union[List[int], Any, Dict[str, Any]]]) -> Dict[str, Any]:
        batch = super().torch_call(examples)

        idxs_to_keep = torch.ones(batch["labels"].size(0), dtype=torch.bool)
        for idx in range(len(examples)):
            sequence = batch["labels"][idx]

            # The super() collator has turned all `pad_token_id` to -100 but if the model did not have a pad_token
            # we probably set pad_token==eos_token, so now eos_tokens are also incorrectly ignored
            if self.tokenizer.pad_token_id == self.tokenizer.eos_token_id:
                last_non_eos = np.where(sequence != -100)[0][-1]
                eos_idxs = np.where(sequence == -100)[0]
                # Reset the -100's that the default collator has given, back to
                # the EOS ID so that the model learns to predict it correctly
                non_pad_eos = eos_idxs[eos_idxs <= last_non_eos + 1]
                sequence[non_pad_eos] = self.tokenizer.eos_token_id
                # Make sure the attention is set correctly for non-padding EOS
                if "attention_mask" in batch:
                    batch["attention_mask"][idx][non_pad_eos] = 1

            start_idxs = []
            for start_idx in np.where(sequence == self.start_idxs[0])[0]:
                if torch.equal(self.start_idxs, sequence[start_idx : start_idx + self.start_idxs.size(0)]):
                    start_idxs.append(start_idx + self.start_idxs.size(0))

            # +1 so we also include the EOS token for the model to predict
            end_idxs = (np.where(sequence == self.tokenizer.eos_token_id)[0] + 1).tolist()

            if len(start_idxs) != len(end_idxs):
                logger.warning(
                    f"Malformed input. Expected that the EOS token ({len(end_idxs)}) occurred the same number of times"
                    f" as `[/INST]` ({len(start_idxs)}). This error may occur when your regular text has tokens that"
                    f" look like the special EOS token {self.tokenizer.eos_token}, or when parts of the original data"
                    f" were truncated due to the given max_length."
                )
                if self.skip_mismatched_sequences:
                    idxs_to_keep[idx] = False
                    continue

            # Mask everything that is NOT between [/INST] and EOS tokens
            mask = torch.ones_like(sequence, dtype=torch.bool)
            for start, end in zip(start_idxs, end_idxs):
                mask[start:end] = False
            batch["labels"][idx][mask] = self.ignore_index

        # Remove items with invalid prompts (cf. warning above)
        batch = {k: v[idxs_to_keep] for k, v in batch.items()}
        num_fails = (~idxs_to_keep).to(torch.int32).sum().item()
        if num_fails:
            logger.warning(f"Failed {num_fails} times. Current batch size: {len(batch['labels'])}. See above.")

        return batch


def main():
    # See https://gist.github.com/BramVanroy/f78530673b1437ed0d6be7c61cdbdd7c
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))

    try:
        # Assumes that the first .json file is the config file (if any)
        config_file = next(iter(arg for arg in sys.argv if arg.endswith(".json")))
    except StopIteration:
        config_file = None

    run_name_specified = False
    if config_file:
        config_args = parser.parse_json_file(json_file=os.path.abspath(config_file))
        raw_config_json = json.loads(Path(config_file).read_text(encoding="utf-8"))

        config_arg_idx = sys.argv.index(config_file)
        other_args = sys.argv[config_arg_idx + 1 :]
        arg_names = {arg[2:] for arg in other_args if arg.startswith("--")}

        if "run_name" in arg_names or "run_name" in raw_config_json:
            run_name_specified = True

        required_args = [
            (act.option_strings[0], "dummy")
            for act in parser._actions
            if act.required and not any(act_s[2:] in arg_names for act_s in act.option_strings)
        ]
        required_args = [arg for req_dummy_args in required_args for arg in req_dummy_args]  # Flatten

        cli_args = other_args + required_args
        cli_args = parser.parse_args_into_dataclasses(args=cli_args, look_for_args_file=False)

        all_args = []

        for cfg_dc, cli_dc in zip(config_args, cli_args):
            # Have to check explicitly for no_ for the automatically added negated boolean arguments
            # E.g. find_unused... vs no_find_unused...
            cli_d = {k: v for k, v in dataclasses.asdict(cli_dc).items() if k in arg_names or f"no_{k}" in arg_names}
            all_args.append(dataclasses.replace(cfg_dc, **cli_d))
        model_args, data_args, training_args = all_args
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Normally, post_init of training_args sets run_name to output_dir (defaults to "results/" in our config file)
    # But if we overwrite output_dir with a CLI option, then we do not correctly update
    # run_name to the same value. Which in turn will lead to wandb to use the original "results/" as a run name
    # see: https://github.com/huggingface/transformers/blob/fe861e578f50dc9c06de33cd361d2f625017e624/src/transformers/integrations.py#L741-L742
    # Instead we explicitly have to set run_name to the output_dir again -- but of course only if the user
    # did not specifically specify run_name in the config or in the CLI
    if not run_name_specified:
        training_args.run_name = training_args.output_dir

    if training_args.do_eval and data_args.streaming and not data_args.use_presplit_validation:
        raise ValueError(
            "When using 'streaming=True' it is not possible to automatically generate a split from the"
            " training set. This is not supported by 'datasets'. Specify a validation set, disable"
            " streaming, or enable 'use_presplit_validation'"
        )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if training_args.should_log:
        # The default of training_args.log_level is passive, so we set log level at info here to have that default.
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    if model_args.use_flash_attention:
        if (gpu_capability := torch.cuda.get_device_capability()[0]) >= 8:
            from patch_llama_flash_attn import replace_attn_with_flash_attn

            replace_attn_with_flash_attn()
        else:
            logger.warning(
                "Your GPU does not support Flash Attention. Requires at least capability 8.0. You have"
                f" {gpu_capability}."
            )
            model_args.use_flash_attention = False

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Get the datasets: you can either provide your own CSV/JSON/TXT training and evaluation files (see below)
    # or just provide the name of one of the public datasets available on the hub at https://huggingface.co/datasets/
    # (the dataset will be downloaded automatically from the datasets Hub).
    #
    # For CSV/JSON files, this script will use the column called 'text' or the first column if no column called
    # 'text' is found. You can easily tweak this behavior (see below).
    #
    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    if data_args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        raw_datasets = load_dataset(
            data_args.dataset_name,
            data_args.dataset_config_name,
            cache_dir=model_args.cache_dir,
            use_auth_token=True if model_args.use_auth_token else None,
            streaming=data_args.streaming,
            num_proc=data_args.preprocessing_num_workers,
        )
        if "validation" not in raw_datasets.keys() or not data_args.use_presplit_validation:
            if data_args.streaming:
                raise ValueError(
                    "When using 'streaming=True' it is not possible to automatically generate a split from the"
                    " training set. This is not supported by 'datasets'. Specify a validation set, disable"
                    " streaming, or enable 'use_presplit_validation'"
                )
            raw_datasets["validation"] = load_dataset(
                data_args.dataset_name,
                data_args.dataset_config_name,
                split=f"train[:{data_args.validation_split_percentage}%]",
                cache_dir=model_args.cache_dir,
                use_auth_token=True if model_args.use_auth_token else None,
                streaming=data_args.streaming,
                num_proc=data_args.preprocessing_num_workers,
            )
            raw_datasets["train"] = load_dataset(
                data_args.dataset_name,
                data_args.dataset_config_name,
                split=f"train[{data_args.validation_split_percentage}%:]",
                cache_dir=model_args.cache_dir,
                use_auth_token=True if model_args.use_auth_token else None,
                streaming=data_args.streaming,
                num_proc=data_args.preprocessing_num_workers,
            )
    else:
        data_files = {}
        dataset_args = {}
        if data_args.train_file is not None:
            data_files["train"] = data_args.train_file
        if data_args.validation_file is not None:
            data_files["validation"] = data_args.validation_file
        extension = (
            data_args.train_file.split(".")[-1]
            if data_args.train_file is not None
            else data_args.validation_file.split(".")[-1]
        )
        if extension == "txt":
            extension = "text"
            dataset_args["keep_linebreaks"] = data_args.keep_linebreaks
        raw_datasets = load_dataset(
            extension,
            data_files=data_files,
            cache_dir=model_args.cache_dir,
            use_auth_token=True if model_args.use_auth_token else None,
            num_proc=data_args.preprocessing_num_workers,
            **dataset_args,
        )
        # If no validation data is there, validation_split_percentage will be used to divide the dataset.
        if "validation" not in raw_datasets.keys() or not data_args.use_presplit_validation:
            raw_datasets["validation"] = load_dataset(
                extension,
                data_files=data_files,
                split=f"train[:{data_args.validation_split_percentage}%]",
                cache_dir=model_args.cache_dir,
                use_auth_token=True if model_args.use_auth_token else None,
                num_proc=data_args.preprocessing_num_workers,
                **dataset_args,
            )
            raw_datasets["train"] = load_dataset(
                extension,
                data_files=data_files,
                split=f"train[{data_args.validation_split_percentage}%:]",
                cache_dir=model_args.cache_dir,
                use_auth_token=True if model_args.use_auth_token else None,
                num_proc=data_args.preprocessing_num_workers,
                **dataset_args,
            )

    logger.info("Loaded dataset!")
    logger.info(str(raw_datasets))
    config_kwargs = {
        "cache_dir": model_args.cache_dir,
        "revision": model_args.model_revision,
        "use_auth_token": True if model_args.use_auth_token else None,
        "trust_remote_code": model_args.trust_remote_code,
    }
    if model_args.config_name:
        config = AutoConfig.from_pretrained(model_args.config_name, **config_kwargs)
    elif model_args.model_name_or_path:
        config = AutoConfig.from_pretrained(model_args.model_name_or_path, **config_kwargs)
    else:
        config = CONFIG_MAPPING[model_args.model_type]()
        logger.warning("You are instantiating a new config instance from scratch.")
        if model_args.config_overrides is not None:
            logger.info(f"Overriding config: {model_args.config_overrides}")
            config.update_from_string(model_args.config_overrides)
            logger.info(f"New config: {config}")

    tokenizer_kwargs = {
        "cache_dir": model_args.cache_dir,
        "use_fast": model_args.use_fast_tokenizer,
        "revision": model_args.model_revision,
        "use_auth_token": True if model_args.use_auth_token else None,
        "trust_remote_code": model_args.trust_remote_code,
    }
    if model_args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_name, **tokenizer_kwargs)
    elif model_args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, **tokenizer_kwargs)
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script."
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )

    if getattr(tokenizer, "pad_token", None) is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    compute_dtype = getattr(torch, model_args.bnb_4bit_compute_dtype)

    bnb_config = None
    if model_args.load_in_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=model_args.load_in_4bit,
            bnb_4bit_quant_type=model_args.bnb_4bit_quant_type,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=model_args.use_nested_quant,
        )

    torch_dtype = (
        model_args.torch_dtype if model_args.torch_dtype in ["auto", None] else getattr(torch, model_args.torch_dtype)
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        quantization_config=bnb_config,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=model_args.low_cpu_mem_usage,
        trust_remote_code=model_args.trust_remote_code,
    )
    model.config.use_cache = False
    model.config.pretraining_tp = 1

    if model_args.use_flash_attention:
        from patch_llama_flash_attn import forward

        if model.model.layers[0].self_attn.forward.__doc__ != forward.__doc__:
            logger.error(
                f"Model is not using flash attention. Flash Attention currently only supported for LlamaModel."
                f" You are using {type(model)}."
            )
            model_args.use_flash_attention = False
        else:
            logger.info("Successfully enabled Flash Attention!")

    callbacks = []
    if model_args.use_peft:
        try:
            target_modules = TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING[model.config.model_type]
            if model_args.lora_target_linear:
                if model.config.model_type in ["RefinedWebModel", "RefinedWeb", "falcon"]:
                    target_modules += ["dense", "dense_h_to_4h", "dense_4h_to_h"]
                elif model.config.model_type == "llama":
                    target_modules += ["gate_proj", "up_proj", "down_proj"]
        except (KeyError, AttributeError):
            if model_args.model_type is not None:
                target_modules = TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING[model_args.model_type]
            else:
                raise KeyError(
                    "Cannot automatically derive model type. Specify '--model_type' explicitly."
                    " See https://github.com/huggingface/peft/blob/e06d94ddeb6c70913593740618df76908b918d66/src/peft/utils/other.py#L262"
                )

        peft_config = LoraConfig(
            lora_alpha=model_args.lora_alpha,
            lora_dropout=model_args.lora_dropout,
            r=model_args.lora_r,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=target_modules,
        )
        logger.info(f"Targetting {target_modules} with LoRA.")

        if getattr(model, "is_loaded_in_8bit", False) or getattr(model, "is_loaded_in_4bit", False):
            model = prepare_model_for_kbit_training(model)

        model = get_peft_model(model, peft_config)
        callbacks.append(PeftSavingCallback)

    # We resize the embeddings only when necessary to avoid index errors. If you are creating a model from scratch
    # on a small vocab and want a smaller embedding size, remove this test.
    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        model.resize_token_embeddings(len(tokenizer))

    max_length = model_args.max_length if model_args.max_length else tokenizer.model_max_length

    def promptify_sample(
        tokenizer, dialog: list[dict[str, str]], system_prompt: str, column_name: str = "text"
    ) -> Dict[str, str]:
        text = f"{tokenizer.bos_token}[INST] "
        if system_prompt:
            text += f"<<SYS>>\n{system_prompt}\n<</SYS>>\n\n"

        for turn_idx, turn in enumerate(dialog):
            if turn["role"] == "user":
                if turn_idx > 0:
                    text += f"{tokenizer.bos_token}[INST] "
                text += f'{turn["content"]} [/INST]'
            else:
                text += f' {turn["content"]} {tokenizer.eos_token}'

        return {column_name: text}

    def tokenize(examples):
        outputs = tokenizer(
            examples["text"],
            truncation=True,
            max_length=max_length,
            add_special_tokens=False,
        )
        return {"input_ids": outputs["input_ids"], "attention_mask": outputs["attention_mask"]}

    # Merge samples BUT make sure that conversations are never split so that the model better learns document-level
    # conversational features for consistency. So we basically try to merge the largest to the smallest samples
    # and if they're smaller than max_length, we try to add more, and so on.
    non_sys_prompt_ds = raw_datasets.map(
        lambda s: promptify_sample(tokenizer, s["dialog"], system_prompt=None, column_name="no_sys_text"),
        num_proc=data_args.preprocessing_num_workers,
        desc="Running prompter for building efficient batches",
    )
    non_sys_prompt_ds = non_sys_prompt_ds.map(
        lambda batch: {
            "input_lengths": [
                len(inputs)
                for inputs in tokenizer(
                    batch["no_sys_text"],
                    truncation=True,
                    max_length=max_length,
                    add_special_tokens=False,
                )["input_ids"]
            ]
        },
        batched=True,
        num_proc=data_args.preprocessing_num_workers,
        batch_size=data_args.dataset_batch_size,
        desc="Running tokenizer for building efficient batches",
    )

    sys_prompt_size = len(tokenizer(f"<<SYS>>\n{data_args.system_prompt}\n<</SYS>>\n\n")["input_ids"])
    logging.info(f"System prompt length (after tokenization): {sys_prompt_size}")

    non_sys_prompt_ds = non_sys_prompt_ds.sort("input_lengths", reverse=True)

    split_batch_idxs = {}
    # For each split, collect sub-batches that can be merged so that their combined length is still
    # smaller than max_length. We do this by sorting the dataset by tokenized length (above)
    # and then trying to combine the largest and smallest items iteratively
    for splitname, splitds in non_sys_prompt_ds.items():
        lengths = splitds["input_lengths"]
        last_idx = len(splitds["input_lengths"]) - 1
        split_batch_idxs[splitname] = []
        for first_idx in range(len(lengths)):
            large_length = lengths[first_idx]
            batch = [(first_idx, large_length)]
            total_length = lengths[last_idx] + large_length + sys_prompt_size
            while total_length <= max_length:
                batch.append((last_idx, lengths[last_idx]))
                last_idx -= 1
                total_length += lengths[last_idx]

                if first_idx >= last_idx:
                    break

            split_batch_idxs[splitname].append(batch)
            if first_idx >= last_idx:
                break

    all_ids = []
    raw_datasets = {}
    for splitname, batches in split_batch_idxs.items():
        logging.info(f"Dataset {splitname} size (no. orig. samples)", len(non_sys_prompt_ds[splitname]))
        logging.info(f"Dataset {splitname} size after merging up to max_length", len(batches))
        merged_dataset = []
        for batch in batches:
            total_batch_length = sum([x[1] for x in batch])
            assert total_batch_length <= max_length
            batch_ids = [x[0] for x in batch]
            # Merge and flatten batch
            batch_data = {
                "dialog": [d for dialog in non_sys_prompt_ds[splitname].select(batch_ids)["dialog"] for d in dialog]
            }
            merged_dataset.append(batch_data)
            all_ids.extend(batch_ids)

        raw_datasets[splitname] = Dataset.from_list(merged_dataset)

    assert not set(range(len(non_sys_prompt_ds))).difference(all_ids)

    raw_datasets = DatasetDict(raw_datasets)

    # Now that that's done, we can start processing the merged datasets (prompting+tokenizing)
    raw_datasets = raw_datasets.map(
        lambda s: promptify_sample(tokenizer, s[data_args.dialog_column_name], system_prompt=data_args.system_prompt),
        batched=False,
        remove_columns=raw_datasets["train"].column_names,
        num_proc=data_args.preprocessing_num_workers,
        desc="Converting dataset into prompts",
    )

    raw_datasets = raw_datasets.map(
        tokenize,
        batched=True,
        remove_columns=raw_datasets["train"].column_names,
        num_proc=data_args.preprocessing_num_workers,
        batch_size=data_args.dataset_batch_size,
        desc="Running tokenizer on datasets",
    )

    if training_args.do_train:
        if "train" not in raw_datasets:
            raise ValueError("--do_train requires a train dataset")

        train_dataset = raw_datasets["train"]
        if data_args.max_train_samples is not None:
            max_train_samples = min(len(train_dataset), data_args.max_train_samples)
            train_dataset = train_dataset.select(range(max_train_samples))

    if training_args.do_eval:
        if "validation" not in raw_datasets:
            raise ValueError("--do_eval requires a validation dataset")

        eval_dataset = raw_datasets["validation"]
        if data_args.max_eval_samples is not None:
            max_eval_samples = min(len(eval_dataset), data_args.max_eval_samples)
            eval_dataset = eval_dataset.select(range(max_eval_samples))

    if data_args.early_stopping_patience is not None and data_args.early_stopping_threshold is not None:
        callbacks.append(
            EarlyStoppingCallback(
                early_stopping_patience=data_args.early_stopping_patience,
                early_stopping_threshold=data_args.early_stopping_threshold,
            )
        )
        logger.info(
            f"Early stopping enabled (patience: {data_args.early_stopping_patience};"
            f" threshold: {data_args.early_stopping_threshold})!"
        )
    elif (data_args.early_stopping_patience is None or data_args.early_stopping_threshold is None) and not (
        data_args.early_stopping_patience is None and data_args.early_stopping_threshold is None
    ):
        raise ValueError(
            "Both 'early_stopping_patience' and 'early_stopping_threshold' must be given, or none of them."
            " If none are given, early stopping will not be used."
        )

    collator = DataCollatorForLlamaChatCompletion(tokenizer)
    # Initialize our Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=collator,
        callbacks=callbacks,
    )

    if model_args.load_in_4bit:
        for name, module in trainer.model.named_modules():
            if "norm" in name:
                module = module.to(torch.float32)

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()  # Saves the tokenizer too for easy upload

        metrics = train_result.metrics

        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        metrics = trainer.evaluate()

        max_eval_samples = data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
        metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))
        try:
            perplexity = math.exp(metrics["eval_loss"])
        except OverflowError:
            perplexity = float("inf")
        metrics["perplexity"] = perplexity

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    kwargs = {"finetuned_from": model_args.model_name_or_path, "tasks": "text-generation"}
    if data_args.dataset_name is not None:
        kwargs["dataset_tags"] = data_args.dataset_name
        if data_args.dataset_config_name is not None:
            kwargs["dataset_args"] = data_args.dataset_config_name
            kwargs["dataset"] = f"{data_args.dataset_name} {data_args.dataset_config_name}"
        else:
            kwargs["dataset"] = data_args.dataset_name

    if training_args.push_to_hub:
        trainer.push_to_hub(**kwargs)
    else:
        trainer.create_model_card(**kwargs)


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()

import logging
import math
import os
import sys
from dataclasses import dataclass, field
from itertools import chain
from pathlib import Path
from typing import Optional

import psutil
from datasets import load_dataset
from transformers import AutoTokenizer, HfArgumentParser, set_seed


logger = logging.getLogger(__name__)


@dataclass
class TokenizerArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
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
    trust_remote_code: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to trust the code that may be downloaded alongside some models. This may be necessary to run"
                " models like Falcon who are not fully integrated in `transformers` yet."
            )
        },
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    output_dir: str = field(
        metadata={
            "help": "The dataset will be saved under output directory so that it can be"
            " loaded directly from disk without relying on cache."
        }
    )
    dataset_name: str = field(metadata={"help": "The name of the dataset to use (via the datasets library)."})
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    text_column_name: Optional[str] = field(default="text", metadata={"help": "Text column to tokenize."})
    block_size: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "Optional input sequence length after tokenization. "
                "The training dataset will be truncated in block of this size for training. "
                "Default to the model max input length for single sentence inputs (take into account special tokens)."
            )
        },
    )
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
    batch_size: int = field(
        default=1000,
        metadata={
            "help": "Number of examples per batch provided to function if batched=True. If batch_size <= 0 or "
            "batch_size == None, provide the full dataset as a single batch to function."
        },
    )
    use_presplit_validation: bool = field(
        default=True,
        metadata={
            "help": "Whether to look for and use a 'validation' split in the given HF dataset. If"
            " disabled, will use 'validation_split_percentage' to turn a portion of"
            " the training set into a validation set"
        },
    )
    seed: int = field(default=42, metadata={"help": "Random seed."})


def convert_size(size_bytes):
    if size_bytes == 0:
        return "0B"
    size_name = ("B", "KB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB")
    i = int(math.floor(math.log(size_bytes, 1024)))
    p = math.pow(1024, i)
    s = round(size_bytes / p, 2)
    return f"{s} {size_name[i]}"


def main():
    # See https://gist.github.com/BramVanroy/f78530673b1437ed0d6be7c61cdbdd7c
    parser = HfArgumentParser((TokenizerArguments, DataTrainingArguments))
    tok_args, data_args = parser.parse_args_into_dataclasses()

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
        level=logging.INFO,
    )

    mem = convert_size(psutil.virtual_memory().total)
    cpu_cores = os.cpu_count()

    logger.info(f"Running on {cpu_cores:,} CPU cores and {mem} memory")
    logger.info(
        f"Running with batch size {data_args.batch_size:,}" f" and {data_args.preprocessing_num_workers:,} workers"
    )

    # Set seed before initializing model.
    set_seed(data_args.seed)

    # Downloading and loading a dataset from the hub.
    proc_datasets = load_dataset(
        data_args.dataset_name,
        data_args.dataset_config_name,
        cache_dir=tok_args.cache_dir,
        use_auth_token=True if tok_args.use_auth_token else None,
        num_proc=data_args.preprocessing_num_workers,
    )
    if "validation" not in proc_datasets.keys() or not data_args.use_presplit_validation:
        proc_datasets["validation"] = load_dataset(
            data_args.dataset_name,
            data_args.dataset_config_name,
            split=f"train[:{data_args.validation_split_percentage}%]",
            cache_dir=tok_args.cache_dir,
            use_auth_token=True if tok_args.use_auth_token else None,
            num_proc=data_args.preprocessing_num_workers,
        )
        proc_datasets["train"] = load_dataset(
            data_args.dataset_name,
            data_args.dataset_config_name,
            split=f"train[{data_args.validation_split_percentage}%:]",
            cache_dir=tok_args.cache_dir,
            use_auth_token=True if tok_args.use_auth_token else None,
            num_proc=data_args.preprocessing_num_workers,
        )

    tokenizer_kwargs = {
        "cache_dir": tok_args.cache_dir,
        "use_fast": tok_args.use_fast_tokenizer,
        "revision": tok_args.model_revision,
        "use_auth_token": True if tok_args.use_auth_token else None,
        "trust_remote_code": tok_args.trust_remote_code,
    }
    if tok_args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(tok_args.tokenizer_name, **tokenizer_kwargs)
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script."
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )

    if getattr(tokenizer, "pad_token", None) is None:
        tokenizer.pad_token = tokenizer.eos_token

    def tokenize(examples):
        # Might throw warnings that thetext is too long
        # but that is okay as we will chunk into smaller pieces later on
        outputs = tokenizer(examples[data_args.text_column_name])

        return {"input_ids": outputs["input_ids"], "attention_mask": outputs["attention_mask"]}

    # Process datasets so that they are cached and we can use them later on in the training scripts
    proc_datasets = proc_datasets.map(
        tokenize,
        batched=True,
        remove_columns=proc_datasets["train"].column_names,
        num_proc=data_args.preprocessing_num_workers,
        batch_size=data_args.batch_size,
        desc="Running tokenizer on datasets",
        keep_in_memory=True,
    )

    # Taken from
    # https://github.com/huggingface/transformers/blob/e75cb0cb3c5fef887abea6f099252e59a659af9d/examples/pytorch/language-modeling/run_clm.py#L490
    def group_texts(examples):
        dkeys = list(examples.keys())
        # Concatenate all texts.
        examples = {k: list(chain(*examples[k])) for k in dkeys}
        # We drop the small remainder, and if the total_length < block_size  we exclude this batch and return an empty dict.
        total_length = (len(examples[dkeys[0]]) // data_args.block_size) * data_args.block_size
        # Split by chunks of max_len, so return value will be a multidimensional tensor for input_ids and attention mask
        # We do not add labels here but ise DataCollatorForLanguageModeling during training which automatically adds them
        return {
            k: [t[i : i + data_args.block_size] for i in range(0, total_length, data_args.block_size)]
            for k, t in examples.items()
        }

    logger.info(
        "You can ignore the 'length is longer than' errors because we will chunk the texts into"
        " 'block_size' sized blocks later"
    )
    proc_datasets = proc_datasets.map(
        group_texts,
        batched=True,
        num_proc=data_args.preprocessing_num_workers,
        batch_size=data_args.batch_size,
        desc=f"Grouping texts in chunks of {data_args.block_size}",
        keep_in_memory=True,
    )

    dataset_name_cfg = f"{data_args.dataset_name.split('/')[-1]}--{data_args.dataset_config_name}"
    output_dir = (
        Path(data_args.output_dir)
        / f"{dataset_name_cfg}-{tok_args.tokenizer_name.split('/')[-1]}-{data_args.block_size}"
    )
    output_dir.mkdir(exist_ok=True, parents=True)
    proc_datasets.save_to_disk(output_dir)

    logger.info(f"Dataset saved to {str(output_dir)}")
    logger.info(str(proc_datasets))


if __name__ == "__main__":
    main()

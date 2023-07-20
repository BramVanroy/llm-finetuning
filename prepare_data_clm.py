import logging
import sys
from dataclasses import dataclass, field
from typing import Optional

from datasets import load_dataset

from transformers import (
    AutoTokenizer,
    HfArgumentParser,
    set_seed
)

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
                "Whether to trust the code that may be downloaded alongside some models. This may be necessary to run models like Falcon who are not fully integrated in `transformers` yet."
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
    train_file: Optional[str] = field(default=None, metadata={"help": "The input training data file (a text file)."})
    validation_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input evaluation data file to evaluate the perplexity on (a text file)."},
    )
    streaming: bool = field(default=False, metadata={"help": "Enable streaming mode"})
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
        metadata={"help": "Number of examples per batch provided to function if batched=True. If batch_size <= 0 or "
                          "batch_size == None, provide the full dataset as a single batch to function."},
    )
    use_presplit_validation: bool = field(
        default=True,
        metadata={"help": "Whether to look for and use a 'validation' split in the given HF dataset. If"
                          " disabled, will use 'validation_split_percentage' to turn a portion of"
                          " the training set into a validation set"}
    )
    seed: int = field(default=42, metadata={"help": "Random seed."})


def main():
    # See https://gist.github.com/BramVanroy/f78530673b1437ed0d6be7c61cdbdd7c
    parser = HfArgumentParser((TokenizerArguments, DataTrainingArguments))
    tok_args, data_args = parser.parse_args_into_dataclasses()

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    # Set seed before initializing model.
    set_seed(data_args.seed)

    if data_args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        raw_datasets = load_dataset(
            data_args.dataset_name,
            data_args.dataset_config_name,
            cache_dir=tok_args.cache_dir,
            use_auth_token=True if tok_args.use_auth_token else None,
            streaming=data_args.streaming,
            num_proc=data_args.preprocessing_num_workers,
        )
        if "validation" not in raw_datasets.keys() or not data_args.use_presplit_validation:
            if data_args.streaming:
                raise ValueError(
                    "When using 'streaming=True' it is not possible to automatically generate a split from the"
                    " training set. This is not supported by 'datasets'. Specify a validation set, disable"
                    " streaming, or enable 'use_presplit_validation'")
            raw_datasets["validation"] = load_dataset(
                data_args.dataset_name,
                data_args.dataset_config_name,
                split=f"train[:{data_args.validation_split_percentage}%]",
                cache_dir=tok_args.cache_dir,
                use_auth_token=True if tok_args.use_auth_token else None,
                streaming=data_args.streaming,
                num_proc=data_args.preprocessing_num_workers,
            )
            raw_datasets["train"] = load_dataset(
                data_args.dataset_name,
                data_args.dataset_config_name,
                split=f"train[{data_args.validation_split_percentage}%:]",
                cache_dir=tok_args.cache_dir,
                use_auth_token=True if tok_args.use_auth_token else None,
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
            cache_dir=tok_args.cache_dir,
            use_auth_token=True if tok_args.use_auth_token else None,
            num_proc=data_args.preprocessing_num_workers,
            **dataset_args,
        )
        # If no validation data is there, validation_split_percentage will be used to divide the dataset.
        if "validation" not in raw_datasets.keys() or not data_args.use_presplit_validation:
            raw_datasets["validation"] = load_dataset(
                extension,
                data_files=data_files,
                split=f"train[:{data_args.validation_split_percentage}%]",
                cache_dir=tok_args.cache_dir,
                use_auth_token=True if tok_args.use_auth_token else None,
                num_proc=data_args.preprocessing_num_workers,
                **dataset_args,
            )
            raw_datasets["train"] = load_dataset(
                extension,
                data_files=data_files,
                split=f"train[{data_args.validation_split_percentage}%:]",
                cache_dir=tok_args.cache_dir,
                use_auth_token=True if tok_args.use_auth_token else None,
                num_proc=data_args.preprocessing_num_workers,
                **dataset_args,
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

    column_names = list(raw_datasets["train"].features)
    text_column_name = "text" if "text" in column_names else column_names[0]

    def tokenize(element):
        outputs = tokenizer(
            element[text_column_name],
            truncation=True,
            padding=False,
            max_length=data_args.block_size,
            return_overflowing_tokens=False,
            return_length=False,
        )

        return {"input_ids": outputs["input_ids"], "attention_mask": outputs["attention_mask"]}

    # Process datasets so that they are cached and we can use them later on in the training scripts
    _ = raw_datasets["train"].map(
        tokenize,
        batched=True,
        remove_columns=raw_datasets["train"].column_names,
        num_proc=data_args.preprocessing_num_workers,
        batch_size=data_args.batch_size,
    )
    if "validation" in raw_datasets:
        _ = raw_datasets["validation"].map(
            tokenize,
            batched=True,
            remove_columns=raw_datasets["train"].column_names,
            num_proc=data_args.preprocessing_num_workers,
            batch_size=data_args.batch_size,
        )


if __name__ == "__main__":
    main()

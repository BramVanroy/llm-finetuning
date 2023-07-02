import logging
from argparse import Namespace
from typing import Dict, Union, Literal

import numpy as np
from datasets import DatasetDict, Dataset
import sys
import os

sys.path.append(os.getcwd())  # noqa

from prompt_templates import format_sample, get_lm_prefix

from tqdm import tqdm

logger = logging.getLogger(__name__)


def formatting_prompts_func(examples, template_name: Literal["alpaca"] = "alpaca"):
    output_text = []
    cols = list(examples.keys())
    for idx in range(len(examples[cols[0]])):
        args = {col: examples[col][idx] for col in cols}
        formatted = format_sample(**args, template_name=template_name)
        output_text.append(formatted)

    return output_text


def _check_if_response_in_prompt_ids(input_idxs: np.ndarray, response_idxs: np.ndarray):
    """Check whether the response prefix (its tokenized indices) are completely part of the input indices AND
    that there is content after the prompt as well.

    This is useful if we use a completion-only collator, which will error if a sample does not have the response prefix.
    This can happen due to truncation in the tokenization process"""
    for idx in np.where(np.atleast_1d(input_idxs == response_idxs[0]))[0]:
        last_span_idx = idx + len(response_idxs)
        if np.array_equiv(response_idxs, input_idxs[idx:last_span_idx]):
            if last_span_idx < len(input_idxs):
                return True

    return False


def _find_suitable_samples(dataset: Dataset, tokenizer, data_args: Namespace, dataset_split: str = None):
    """Find samples where, even after tokenization and truncation to the max seq length,
    the response prefix is still fully present."""
    valid_idxs = set()

    response_token_ids = tokenizer(get_lm_prefix(data_args.template_name), add_special_tokens=False, return_tensors="np").input_ids[0]
    for idx, sample in tqdm(enumerate(dataset),
                            desc=f"Processing {dataset_split}" if dataset_split else "Processing",
                            total=len(dataset),
                            leave=False):
        prompt = format_sample(**sample)
        input_ids = tokenizer(prompt, truncation=True, max_length=data_args.max_seq_length, return_tensors="np").input_ids[0]

        if _check_if_response_in_prompt_ids(input_ids, response_token_ids):
            valid_idxs.add(idx)
    return valid_idxs


def filter_on_prefix_present(datasets: Union[Dict, DatasetDict], tokenizer, data_args: Namespace):
    # Extract samples where the Response prefix is visible, even after truncation
    preprocessed_dataset = {}
    for split, ds in datasets.items():
        idxs = _find_suitable_samples(ds, tokenizer, data_args, split)
        preprocessed_dataset[split] = datasets[split].select(idxs)

    datasets = DatasetDict(preprocessed_dataset)

    return datasets


def maybe_undersample_datasets(datasets: Union[Dict, DatasetDict], data_args: Namespace):
    train_dataset = None
    if "train" in datasets:
        train_dataset = datasets["train"]
        if data_args.max_train_samples is not None:
            max_train_samples = min(len(train_dataset), data_args.max_train_samples)
            train_dataset = train_dataset.select(range(max_train_samples))

        logger.info(f"Final TRAIN size: {len(train_dataset):,}")

    eval_dataset = None
    if "validation" in datasets:
        eval_dataset = datasets["validation"]
        if data_args.max_eval_samples is not None:
            max_eval_samples = min(len(eval_dataset), data_args.max_eval_samples)
            eval_dataset = eval_dataset.select(range(max_eval_samples))

        logger.info(f"Final DEV size: {len(eval_dataset):,}")

    return train_dataset, eval_dataset

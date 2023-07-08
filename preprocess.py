import logging
from argparse import Namespace
from typing import Dict, Union, Literal

import numpy as np
from datasets import DatasetDict
import sys
import os

from prompt_format import PromptFormatter

sys.path.append(os.getcwd())  # noqa

logger = logging.getLogger(__name__)


def formatting_prompts_func(examples, prompt_formatter: PromptFormatter):
    output_text = []
    cols = list(examples.keys())
    for idx in range(len(examples[cols[0]])):
        args = {col: examples[col][idx] for col in cols}
        # TODO: fix messages to expected format of list of dicts where each dict has "role" and "content"
        messages = []
        prompt = prompt_formatter.get_training_prompt(messages)
        output_text.append(prompt)

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


def _is_suitable_samples(sample, response_token_ids, prompt_formatter: PromptFormatter, tokenizer, max_seq_length: int):
    """Find samples where, even after tokenization and truncation to the max seq length,
    the response prefix is still fully present."""
    # TODO: convert sample to messages to expected format of list of dicts where each dict has "role" and "content"
    messages = []
    prompt = prompt_formatter.get_training_prompt(messages)
    input_ids = tokenizer(prompt, truncation=True, max_length=max_seq_length, return_tensors="np").input_ids[0]
    return _check_if_response_in_prompt_ids(input_ids, response_token_ids)


def filter_on_prefix_present(datasets: Union[Dict, DatasetDict], prompt_formatter: PromptFormatter, tokenizer, data_args: Namespace):
    response_token_ids = prompt_formatter.assistant_token_ids(tokenizer)
    datasets = datasets.filter(lambda sample: _is_suitable_samples(sample,
                                                                   response_token_ids=response_token_ids,
                                                                   tokenizer=tokenizer,
                                                                   max_seq_length=data_args.max_seq_length)
                               )

    return datasets


def maybe_undersample_datasets(datasets: Union[Dict, DatasetDict], data_args: Namespace):
    train_dataset = None
    if "train" in datasets:
        train_dataset = datasets["train"]
        if data_args.max_train_samples is not None:
            max_train_samples = min(len(train_dataset), data_args.max_train_samples)
            train_dataset = train_dataset.select(range(max_train_samples))

        if hasattr(train_dataset, "__len__"):  # False for streaming=True datasets
            logger.info(f"Final TRAIN size: {len(train_dataset):,}")

    eval_dataset = None
    if "validation" in datasets:
        eval_dataset = datasets["validation"]
        if data_args.max_eval_samples is not None:
            max_eval_samples = min(len(eval_dataset), data_args.max_eval_samples)
            eval_dataset = eval_dataset.select(range(max_eval_samples))

        if hasattr(eval_dataset, "__len__"):
            logger.info(f"Final DEV size: {len(eval_dataset):,}")

    return train_dataset, eval_dataset

#!/usr/bin/env python
# coding=utf-8
# Copyright 2020 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for causal language modeling (GPT, GPT-2, CTRL, ...) on a text file or a dataset.

Here is the full list of checkpoints on the hub that can be fine-tuned by this script:
https://huggingface.co/models?filter=text-generation

Adapted by Bram Vanroy for LLM finetuning on instructions
"""
import dataclasses
import json

import logging
import re
import sys
from itertools import chain

import evaluate
from peft import PeftConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from transformers.testing_utils import CaptureLogger
from transformers.utils import is_peft_available
from trl.trainer.utils import PeftSavingCallback

from collator import DataCollatorForTurnBasedLM

# Setup logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
    force=True,
)

import math
import os
from functools import partial
from pathlib import Path

sys.path.append(os.getcwd())  # noqa

from data import build_data
from hyperopt_args import HyperOptArguments
from lora_config import build_lora_config
from preprocess import formatting_prompts_func, filter_on_prefix_present, maybe_undersample_datasets
from prompt_format import get_prompt_formatter

from config import build_config
from model import build_model
from tokenizer import build_tokenizer

from data_args import DataTrainingArguments
from model_args import ModelArguments
from trainer import SFTTrainer

import torch
import datasets

import transformers
from transformers import (
    EarlyStoppingCallback,
    HfArgumentParser,
    TrainingArguments,
    set_seed, default_data_collator, Trainer, is_torch_tpu_available, PreTrainedModel, AutoModelForCausalLM
)

from transformers.trainer_utils import get_last_checkpoint

logger = logging.getLogger(__name__)


def main():
    # See https://gist.github.com/BramVanroy/f78530673b1437ed0d6be7c61cdbdd7c
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments, HyperOptArguments))

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
        other_args = sys.argv[config_arg_idx + 1:]
        arg_names = {arg[2:] for arg in other_args if arg.startswith("--")}

        if "run_name" in arg_names or "run_name" in raw_config_json:
            run_name_specified = True

        required_args = [(act.option_strings[0], "dummy")
                         for act in parser._actions
                         if act.required and not any(act_s[2:] in arg_names for act_s in act.option_strings)]
        required_args = [arg for req_dummy_args in required_args for arg in req_dummy_args]  # Flatten

        cli_args = other_args + required_args
        cli_args = parser.parse_args_into_dataclasses(args=cli_args, look_for_args_file=False)

        all_args = []

        for cfg_dc, cli_dc in zip(config_args, cli_args):
            # Have to check explicitly for no_ for the automatically added negated boolean arguments
            # E.g. find_unused... vs no_find_unused...
            cli_d = {k: v for k, v in dataclasses.asdict(cli_dc).items() if k in arg_names or f"no_{k}" in arg_names}
            all_args.append(dataclasses.replace(cfg_dc, **cli_d))
        model_args, data_args, training_args, hyperopt_args = all_args
    else:
        model_args, data_args, training_args, hyperopt_args = parser.parse_args_into_dataclasses()

    # Normally, post_init of training_args sets run_name to output_dir (defaults to "results/" in our config file)
    # But if we overwrite output_dir with a CLI option, then we do not correctly update
    # run_name to the same value. Which in turn will lead to wandb to use the original "results/" as a run name
    # see: https://github.com/huggingface/transformers/blob/fe861e578f50dc9c06de33cd361d2f625017e624/src/transformers/integrations.py#L741-L742
    # Instead we explicitly have to set run_name to the output_dir again -- but of course only if the user
    # did not specifically specify run_name in the config or in the CLI
    if not run_name_specified:
        training_args.run_name = training_args.output_dir

    if training_args.do_eval and data_args.streaming and not data_args.use_presplit_validation:
        raise ValueError("When using 'streaming=True' it is not possible to automatically generate a split from the"
                         " training set. This is not supported by 'datasets'. Specify a validation set, disable"
                         " streaming, or enable 'use_presplit_validation'")

    if training_args.should_log:
        # The default of training_args.log_level is passive, so we set log level at info here to have that default.
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    prompt_formatter = get_prompt_formatter(data_args.template_name) if model_args.task == "instruct" else None

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

    config = build_config(model_args)
    tokenizer = build_tokenizer(model_args)
    model = build_model(config, tokenizer, model_args)

    peft_config = build_lora_config(
        model_args.lora_model_type,
        lora_alpha=model_args.lora_alpha,
        lora_dropout=model_args.lora_dropout,
        lora_r=model_args.lora_r
    ) if model_args.lora_model_type != "none" else None

    callbacks = []
    if is_peft_available() and peft_config is not None:
        if not isinstance(peft_config, PeftConfig):
            raise ValueError(
                "If you want to use the PeftModel, you need to pass a PeftConfig object to the SFTTrainer."
                f" and you passed a {type(peft_config)}."
            )

        if not isinstance(model, PeftModel):
            if not isinstance(model, PreTrainedModel):
                model = AutoModelForCausalLM.from_pretrained(
                    model,
                )

            if getattr(model, "is_loaded_in_8bit", False) or getattr(model, "is_loaded_in_4bit", False):
                model = prepare_model_for_kbit_training(model)

            model = get_peft_model(model, peft_config)

        callbacks.append(PeftSavingCallback)

    loaded_datasets = build_data(data_args, model_args)

    if model_args.task == "instruct":
        loaded_datasets = filter_on_prefix_present(loaded_datasets, prompt_formatter, tokenizer, data_args)
    elif model_args.task == "clm":
        if training_args.do_train:
            column_names = list(loaded_datasets["train"].features)
        else:
            column_names = list(loaded_datasets["validation"].features)
        text_column_name = "text" if "text" in column_names else column_names[0]

        # since this will be pickled to avoid _LazyModule error in Hasher force logger loading before tokenize_function
        tok_logger = transformers.utils.logging.get_logger("transformers.tokenization_utils_base")

        def tokenize_function(examples):
            with CaptureLogger(tok_logger) as cl:
                output = tokenizer(examples[text_column_name])
            # clm input could be much much longer than block_size
            if "Token indices sequence length is longer than the" in cl.out:
                tok_logger.warning(
                    "^^^^^^^^^^^^^^^^ Please ignore the warning above - this long input will be chunked into smaller bits"
                    " before being passed to the model."
                )
            return output

        with training_args.main_process_first(desc="dataset map tokenization"):
            if not data_args.streaming:
                loaded_datasets = loaded_datasets.map(
                    tokenize_function,
                    batched=True,
                    num_proc=data_args.preprocessing_num_workers,
                    remove_columns=column_names,
                    load_from_cache_file=not data_args.overwrite_cache,
                    desc="Running tokenizer on dataset",
                )
            else:
                loaded_datasets = loaded_datasets.map(
                    tokenize_function,
                    batched=True,
                    remove_columns=column_names,
                )
        if data_args.block_size is None:
            block_size = tokenizer.model_max_length
            if block_size > 1024:
                logger.warning(
                    "The chosen tokenizer supports a `model_max_length` that is longer than the default `block_size` value"
                    " of 1024. If you would like to use a longer `block_size` up to `tokenizer.model_max_length` you can"
                    " override this default with `--block_size xxx`."
                )
                block_size = 1024
        else:
            if data_args.block_size > tokenizer.model_max_length:
                logger.warning(
                    f"The block_size passed ({data_args.block_size}) is larger than the maximum length for the model"
                    f"({tokenizer.model_max_length}). Using block_size={tokenizer.model_max_length}."
                )
            block_size = min(data_args.block_size, tokenizer.model_max_length)

        # Main data processing function that will concatenate all texts from our dataset and generate chunks of block_size.
        def group_texts(examples):
            # Concatenate all texts.
            concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
            total_length = len(concatenated_examples[list(examples.keys())[0]])
            # We drop the small remainder, and if the total_length < block_size  we exclude this batch and return an empty dict.
            # We could add padding if the model supported it instead of this drop, you can customize this part to your needs.
            total_length = (total_length // block_size) * block_size
            # Split by chunks of max_len.
            result = {
                k: [t[i: i + block_size] for i in range(0, total_length, block_size)]
                for k, t in concatenated_examples.items()
            }
            result["labels"] = result["input_ids"].copy()
            return result

        # Note that with `batched=True`, this map processes 1,000 texts together, so group_texts throws away a remainder
        # for each of those groups of 1,000 texts. You can adjust that batch_size here but a higher value might be slower
        # to preprocess.
        #
        # To speed up this part, we use multiprocessing. See the documentation of the map method for more information:
        # https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasets.Dataset.map
        with training_args.main_process_first(desc="grouping texts together"):
            if not data_args.streaming:
                loaded_datasets = loaded_datasets.map(
                    group_texts,
                    batched=True,
                    num_proc=data_args.preprocessing_num_workers,
                    load_from_cache_file=not data_args.overwrite_cache,
                    desc=f"Grouping texts in chunks of {block_size}",
                )
            else:
                loaded_datasets = loaded_datasets.map(
                    group_texts,
                    batched=True,
                )

    train_dataset, eval_dataset = maybe_undersample_datasets(loaded_datasets, data_args)

    if training_args.do_train and train_dataset is None:
        raise ValueError("--do_train requires a train dataset")
    elif training_args.do_eval and eval_dataset is None:
        raise ValueError("--do_eval requires a validation dataset. If your dataset does"
                         " not have a dedicate validation set, and you did not specify an explicit"
                         " validation_file, and you also did not specify --do_train (so that a portion of"
                         " the training set could be used) then this error may occur.")

    # If you want to use early stopping, both arguments have to be specified. Throw error if just one is specified.
    if hyperopt_args.early_stopping_patience is not None and hyperopt_args.early_stopping_threshold is not None:
        callbacks.append(
            EarlyStoppingCallback(
                early_stopping_patience=hyperopt_args.early_stopping_patience,
                early_stopping_threshold=hyperopt_args.early_stopping_threshold,
            )
        )
        logger.info(f"Early stopping enabled (patience: {hyperopt_args.early_stopping_patience};"
                    f" threshold: {hyperopt_args.early_stopping_threshold})!")
    elif (hyperopt_args.early_stopping_patience is None or hyperopt_args.early_stopping_threshold is None) and not (
            hyperopt_args.early_stopping_patience is None and hyperopt_args.early_stopping_threshold is None
    ):
        raise ValueError(
            "Both 'early_stopping_patience' and 'early_stopping_threshold' must be given, or none of them."
            " If none are given, early stopping will not be used."
        )

    def preprocess_logits_for_metrics(logits, labels):
        if isinstance(logits, tuple):
            # Depending on the model and config, logits may contain extra tensors,
            # like past_key_values, but logits always come first
            logits = logits[0]
        return logits.argmax(dim=-1)

    metric = evaluate.load("accuracy")

    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        # preds have the same shape as the labels, after the argmax(-1) has been calculated
        # by preprocess_logits_for_metrics but we need to shift the labels
        labels = labels[:, 1:].reshape(-1)
        preds = preds[:, :-1].reshape(-1)
        return metric.compute(predictions=preds, references=labels)

    if model_args.task == "instruct":
        collator = DataCollatorForTurnBasedLM(prompt_formatter.user_token,
                                              prompt_formatter.assistant_token,
                                              tokenizer=tokenizer,
                                              mlm=False)
        trainer = SFTTrainer(
            model=model,
            args=training_args,
            peft_config=peft_config,
            train_dataset=train_dataset if training_args.do_train else None,
            eval_dataset=eval_dataset if training_args.do_eval else None,
            tokenizer=tokenizer,
            data_collator=collator,
            formatting_func=partial(formatting_prompts_func, template_name=data_args.template_name),
            max_seq_length=data_args.max_seq_length,
            callbacks=callbacks,
        )
    else:
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset if training_args.do_train else None,
            eval_dataset=eval_dataset if training_args.do_eval else None,
            tokenizer=tokenizer,
            # Data collator will default to DataCollatorWithPadding, so we change it.
            data_collator=default_data_collator,
            compute_metrics=compute_metrics if training_args.do_eval and not is_torch_tpu_available() else None,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics
            if training_args.do_eval and not is_torch_tpu_available()
            else None,
            callbacks=callbacks,
        )

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

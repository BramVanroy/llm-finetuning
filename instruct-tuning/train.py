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
# You can also adapt this script on your own causal language modeling task. Pointers for this are left as comments.

import logging
import math
import os
import sys
from functools import partial
from json import dump
from pathlib import Path

from data import build_data
from hyperopt import wandb_hp_space
from hyperopt_args import HyperOptArguments
from lora_config import build_lora_config
from preprocess import formatting_prompts_func, filter_on_prefix_present, maybe_undersample_datasets
from prompt_templates import get_lm_prefix

sys.path.append(os.getcwd())  # noqa

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
    set_seed
)

from transformers.trainer_utils import get_last_checkpoint
from trl import DataCollatorForCompletionOnlyLM

logger = logging.getLogger(__name__)


def main():
    # See https://gist.github.com/BramVanroy/f78530673b1437ed0d6be7c61cdbdd7c
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments, HyperOptArguments))

    try:
        # Assumes that the first .json file is the config file (if any)
        config_file = next(iter(arg for arg in sys.argv if arg.endswith(".json")))
    except StopIteration:
        config_file = None

    if config_file:
        config_args = parser.parse_json_file(json_file=os.path.abspath(config_file))
        config_arg_idx = sys.argv.index(config_file)
        other_args = sys.argv[config_arg_idx + 1:]

        arg_names = {arg[2:] for arg in other_args if arg.startswith("--")}
        required_args = [(act.option_strings[0], "dummy")
                         for act in parser._actions
                         if act.required and not any(act_s[2:] in arg_names for act_s in act.option_strings)]
        required_args = [arg for req_dummy_args in required_args for arg in req_dummy_args]  # Flatten
        cli_args = other_args + required_args
        cli_args = parser.parse_args_into_dataclasses(args=cli_args, look_for_args_file=False)

        all_args = []
        for cfg_dc, cli_dc in zip(config_args, cli_args):
            cli_d = {k: v for k, v in dataclasses.asdict(cli_dc).items() if k in arg_names}
            all_args.append(dataclasses.replace(cfg_dc, **cli_d))
        model_args, data_args, training_args, hyperopt_args = all_args
    else:
        model_args, data_args, training_args, hyperopt_args = parser.parse_args_into_dataclasses()

    if hyperopt_args.do_hparams_search:
        logger.error("Hyperparameter search currently not supported by trl. Disabling...")
        hyperopt_args.do_hparams_search = False

    training_args.save_safetensors = True

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

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}"
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

    config = build_config(model_args)
    tokenizer = build_tokenizer(model_args)
    model = build_model(config, tokenizer, model_args) if not hyperopt_args.do_hparams_search else None

    loaded_datasets = build_data(data_args, model_args)
    loaded_datasets = filter_on_prefix_present(loaded_datasets, tokenizer, data_args)
    train_dataset, eval_dataset = maybe_undersample_datasets(loaded_datasets, data_args)

    del loaded_datasets

    if training_args.do_train or hyperopt_args.do_hparams_search:
        if train_dataset is None:
            raise ValueError("--do_train and --do_hparams_search require a train dataset")
    elif training_args.do_eval or hyperopt_args.do_hparams_search:
        if eval_dataset is None:
            raise ValueError("--do_eval and --do_hparams_search require a validation dataset. If your dataset does"
                             " not have a dedicate validation set, and you did not specify an explicit"
                             " validation_file, and you also did not specify --do_train (so that a portion of"
                             " the training set could be used) then this error may occur.")
    elif not hyperopt_args.do_hparams_search:
        logger.info(
            "There is nothing to do. Please pass `do_train`, `do_eval`, and/or 'do_hparams_search'.")
        return

    callbacks = []
    # If you want to use early stopping, both arguments have to be specified. Throw error if just one is specified.
    if hyperopt_args.early_stopping_patience is not None and hyperopt_args.early_stopping_threshold is not None:
        callbacks.append(
            EarlyStoppingCallback(
                early_stopping_patience=hyperopt_args.early_stopping_patience,
                early_stopping_threshold=hyperopt_args.early_stopping_threshold,
            )
        )
    elif (hyperopt_args.early_stopping_patience is None or hyperopt_args.early_stopping_threshold is None) and not (
        hyperopt_args.early_stopping_patience is None and hyperopt_args.early_stopping_threshold is None
    ):
        raise ValueError(
            "Both 'early_stopping_patience' and 'early_stopping_threshold' must be given, or none of them."
            " If none are given, early stopping will not be used."
        )

    peft_config = build_lora_config(
        model_args.lora_model_type,
        lora_alpha=model_args.lora_alpha,
        lora_dropout=model_args.lora_dropout,
        lora_r=model_args.lora_r
    )

    collator = DataCollatorForCompletionOnlyLM(get_lm_prefix(data_args.template_name), tokenizer, mlm=False)

    # Initialize our Trainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        peft_config=peft_config,
        train_dataset=train_dataset if training_args.do_train or hyperopt_args.do_hparams_search else None,
        eval_dataset=eval_dataset if training_args.do_eval or hyperopt_args.do_hparams_search else None,
        tokenizer=tokenizer,
        data_collator=collator,
        model_init=partial(build_model, config, tokenizer, model_args) if hyperopt_args.do_hparams_search else None,
        formatting_func=partial(formatting_prompts_func, template_name=data_args.template_name),
        max_seq_length=data_args.max_seq_length,
    )

    for name, module in trainer.model.named_modules():
        if "norm" in name:
            module = module.to(torch.float32)

    if hyperopt_args.do_hparams_search:
        best_trial = trainer.hyperparameter_search(
            backend="wandb",
            hp_space=partial(wandb_hp_space, hyperopt_args=hyperopt_args),
            n_trials=hyperopt_args.hparam_max_trials,
        )

        logging.info(f"Best hyperparameter search run: {best_trial.run_id}")
        with Path(training_args.output_dir).joinpath("wandb_best_hparams.json").open("w", encoding="utf-8") as hp_out:
            best_trial.hyperparameters.pop("assignments", None)
            best_trial.hyperparameters["metric"] = "eval/loss"
            hparams_dump = {
                **best_trial.hyperparameters,
                "best_run": best_trial.run_id,
                "objective": best_trial.objective
            }
            dump(hparams_dump, hp_out, indent=4, sort_keys=True)

        for hparam, v in best_trial.hyperparameters.items():
            setattr(trainer.args, hparam, v)

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

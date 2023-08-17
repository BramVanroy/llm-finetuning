"""Taken from https://github.com/lvwerra/trl/blob/main/examples/stack_llama/scripts/merge_peft_adapter.py"""
from dataclasses import dataclass, field

import torch
from peft import PeftConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer, HfArgumentParser


@dataclass
class ScriptArguments:
    adapter_model_name: str = field(metadata={"help": "the location of the adapters"})
    base_model_name: str = field(metadata={"help": "the base model name to merge with"})
    output_name: str = field(metadata={"help": "where to save the output model"})


parser = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses()[0]

peft_config = PeftConfig.from_pretrained(script_args.adapter_model_name)
if peft_config.task_type == "SEQ_CLS":
    # peft is for reward model so load sequence classification
    model = AutoModelForSequenceClassification.from_pretrained(
        script_args.base_model_name, num_labels=1, torch_dtype=torch.bfloat16
    )
else:
    model = AutoModelForCausalLM.from_pretrained(
        script_args.base_model_name, return_dict=True, torch_dtype=torch.bfloat16, trust_remote_code=True
    )

tokenizer = AutoTokenizer.from_pretrained(script_args.base_model_name, trust_remote_code=True)

# Load the Lora model
model = PeftModel.from_pretrained(model, script_args.adapter_model_name)
model.eval()

model = model.merge_and_unload()

model.save_pretrained(f"{script_args.output_name}", safe_serialization=True)
tokenizer.save_pretrained(f"{script_args.output_name}")

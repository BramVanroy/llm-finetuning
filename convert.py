from dataclasses import dataclass, field
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, FlaxAutoModelForCausalLM, HfArgumentParser, TFAutoModelForCausalLM


@dataclass
class ScriptArguments:
    """
    The name of the Casual LM model we wish to fine with PPO
    """

    model_name: Optional[str] = field(default=None, metadata={"help": "the model directory with PyTorch weights"})
    convert_pt_safe: Optional[bool] = field(default=False, metadata={"help": "convert to safetensors for PyTorch"})


parser = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses()[0]
assert script_args.model_name is not None, "please provide a directory that contains a PyTorch model to convert"

if script_args.convert_pt_safe:
    try:
        pt_model = AutoModelForCausalLM.from_pretrained(
            script_args.model_name, return_dict=True, torch_dtype=torch.bfloat16, trust_remote_code=True
        )
        pt_model.save_pretrained(script_args.model_name, safe_serialization=True)
    except Exception as exc:
        print(f"Failed converting to PyTorch safetensors, {exc}")

try:
    tf_model = TFAutoModelForCausalLM.from_pretrained(
        script_args.model_name, return_dict=True, torch_dtype=torch.bfloat16, trust_remote_code=True, from_pt=True
    )
    tf_model.save_pretrained(script_args.model_name, safe_serialization=True)
except Exception as exc:
    print(f"Failed converting to Tensorflow, {exc}")

try:
    flax_model = FlaxAutoModelForCausalLM.from_pretrained(
        script_args.model_name, return_dict=True, torch_dtype=torch.bfloat16, trust_remote_code=True, from_pt=True
    )
    flax_model.save_pretrained(script_args.model_name, safe_serialization=True)
except Exception as exc:
    print(f"Failed converting to Flax, {exc}")

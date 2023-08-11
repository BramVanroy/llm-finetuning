from typing import Literal

from peft import LoraConfig


def _get_target_modules(lora_model_type: Literal["falcon"] = "falcon"):
    # If adding more, also update `model_type` in model_args.model_type and build_lora_config below
    if lora_model_type == "falcon":
        return [
            "query_key_value",
            "dense",
            "dense_h_to_4h",
            "dense_4h_to_h",
        ]
    else:
        raise ValueError(
            f"Model type '{lora_model_type}' not found. Currently supported model_types are" f" 'falcon'."
        )


def build_lora_config(
    lora_model_type: Literal["falcon"] = "falcon",
    lora_alpha: int = 8,
    lora_dropout: float = 0.0,
    lora_r: int = 8,
    task_type: str = "CAUSAL_LM",
):
    target_modules = _get_target_modules(lora_model_type)
    return LoraConfig(
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        r=lora_r,
        bias="none",
        task_type=task_type,
        target_modules=target_modules,
    )

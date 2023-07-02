from functools import lru_cache
from typing import Literal


def _format_alpaca_sample(instruction: str, input: str, output: str, **kwargs):
    if len(input) >= 2:
        text = f'''Hieronder staat een instructie `Instruction` die een taak beschrijft, gecombineerd met een invoer `Input` die verdere context biedt. Schrijf een antwoord na `Response:` dat het verzoek op de juiste manier voltooit of beantwoordt.

### Instruction:
{instruction}

### Input:
{input}

### Response:
{output}
'''
    else:
        text = f'''Hieronder staat een instructie `Instruction` die een taak beschrijft. Schrijf een antwoord na `Response:` dat het verzoek op de juiste manier voltooit of beantwoordt.

### Instruction:
{instruction}

### Response:
{output}'''
    return text


def format_sample(template_name: Literal["alpaca"] = "alpaca", **kwargs):
    # Note: if you add here, also add the option to data_args.template_name
    if template_name == "alpaca":
        return _format_alpaca_sample(**kwargs)
    else:
        raise ValueError(f"Template name '{template_name}' not found. Currently supported template_names are"
                         f" 'alpaca'.")


@lru_cache
def get_lm_prefix(template_name: Literal["alpaca"] = "alpaca"):
    if template_name == "alpaca":
        return "### Response:\n"
    else:
        raise ValueError(f"Template name '{template_name}' not found. Currently supported template_names are"
                         f" 'alpaca'.")

from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, List, Literal, Optional

import numpy as np


# Inspired by https://huggingface.co/spaces/HuggingFaceH4/starchat-playground/blob/main/dialogues.py
@dataclass(eq=True, frozen=True)
class PromptFormatter:
    system_message: str
    system_token: str = "<|system|>"
    user_token: str = "<|user|>"
    input_token: str = "<|input|>"
    assistant_token: str = "<|assistant|>"
    end_token: str = "<|end|>"
    separator: str = "\n"

    def get_training_prompt(self, messages: List[Dict[Literal["role", "content"], str]]) -> str:
        prompt = self.system_token + self.separator + self.system_message + self.end_token + "\n"
        for message in messages:
            if message["role"] == "user":
                prompt += self.user_token + self.separator + message["content"] + self.end_token + "\n"
            elif message["role"] == "input":
                prompt += self.input_token + self.separator + message["content"] + self.end_token + "\n"
            else:
                prompt += self.assistant_token + self.separator + message["content"] + self.end_token + "\n"
        return prompt

    def get_inference_prompt(self, messages: List[Dict[Literal["role", "content"], str]]) -> str:
        prompt = self.get_training_prompt(messages)
        prompt += self.assistant_token + self.separator
        return prompt

    @lru_cache
    def assistant_token_ids(self, tokenizer, return_tensors="np"):
        return tokenizer(self.assistant_token, add_special_tokens=False, return_tensors=return_tensors).input_ids[0]

    def is_sample_suitable(
        self, tokenizer, messages: List[Dict[Literal["role", "content"], str]], max_length: Optional[int] = None
    ) -> bool:
        """Check whether a sample is suitable for training. That means, whether it contains an assistant message
        prompt (tokens) AND whether those tokens are not the last ones of the prompt
        """
        prompt = self.get_training_prompt(messages)
        assistant_token_ids = self.assistant_token_ids(tokenizer)
        input_ids = tokenizer(prompt, truncation=True, max_length=max_length, return_tensors="np").input_ids[0]
        for idx in np.where(input_ids == assistant_token_ids[0])[0]:
            last_span_idx = idx + len(assistant_token_ids)
            if np.array_equiv(assistant_token_ids, input_ids[idx:last_span_idx]):
                # Check that the last index of the assistant prompt is not the last item,
                # otherwise there is nothing left to predict
                if last_span_idx < len(input_ids):
                    return True
        return False


@dataclass(eq=True, frozen=True)
class AlpacaPromptFormatter(PromptFormatter):
    system_message: str = (
        "Hieronder staat een instructie `Instruction` die een taak beschrijft, gecombineerd met een"
        " invoer `Input` die verdere context biedt. Schrijf een antwoord na `Response:` dat het"
        " verzoek op de juiste manier voltooit of beantwoordt."
    )
    system_token: str = ""
    user_token: str = "### Instruction:"
    input_token: str = "### Input:"
    assistant_token: str = "### Response:"


PROMPT_FORMATTERS = {"alpaca": AlpacaPromptFormatter}


def get_prompt_formatter(name: Literal["alpaca"], **kwargs):
    return PROMPT_FORMATTERS[name](**kwargs)

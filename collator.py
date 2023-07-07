from typing import List, Union, Any, Dict

import numpy as np
from transformers import DataCollatorForLanguageModeling


# Modified from https://github.com/lvwerra/trl/pull/456
class DataCollatorForTurnBasedLM(DataCollatorForLanguageModeling):
    def __init__(self, user_token: str, assistant_token: str, *args, mlm: bool = False, ignore_index: int = -100, **kwargs):
        super().__init__(*args, mlm=mlm, **kwargs)
        self.user_template = user_token
        self.assistant_template = assistant_token
        self.ignore_index = ignore_index

        self.human_token_ids = self.tokenizer(self.user_template).input_ids
        self.human_tokens_len = len(self.human_token_ids)
        self.assistant_token_ids = self.tokenizer(self.assistant_template).input_ids
        self.assistant_tokens_len = len(self.assistant_token_ids)

    def torch_call(self, examples: List[Union[List[int], Any, Dict[str, Any]]]) -> Dict[str, Any]:
        batch = super().torch_call(examples)

        for sample_idx in range(len(examples)):
            assistant_start_idxs = []
            human_start_idxs = []
            for start_hum_idx in np.where(batch["labels"][sample_idx] == self.human_token_ids[0])[0]:
                if self.human_token_ids == examples[sample_idx]["input_ids"][start_hum_idx : start_hum_idx + self.human_tokens_len]:
                    human_start_idxs.append(start_hum_idx)

            for start_ast_idx in np.where(batch["labels"][sample_idx] == self.assistant_token_ids[0])[0]:
                if self.assistant_token_ids == examples[sample_idx]["input_ids"][start_ast_idx: start_ast_idx + self.assistant_tokens_len]:
                    # +self.assistant_tokens_len because we need to ignore the assistant token itself, too
                    assistant_start_idxs.append(start_ast_idx+self.assistant_tokens_len)

            if len(human_start_idxs) != len(assistant_start_idxs):
                raise ValueError(f"Expected an equal amount of turns between assistant and human. Not the case in"
                                 f" sample {sample_idx:,} ({len(assistant_start_idxs)} != {len(human_start_idxs)})")

            for start_idx, end_idx in zip(human_start_idxs, assistant_start_idxs):
                batch["labels"][sample_idx, start_idx:end_idx] = self.ignore_index

        return batch



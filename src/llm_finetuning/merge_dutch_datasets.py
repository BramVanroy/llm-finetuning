from typing import Dict, List, Literal, TypedDict, Union

import datasets
from datasets import load_dataset


# Inspired by https://github.com/facebookresearch/llama/blob/main/example_chat_completion.py
Role = Literal["system", "user", "assistant"]


class Message(TypedDict):
    role: Role
    content: str


Dialog = List[Message]


def process_dolly():
    dolly_ds = load_dataset("BramVanroy/dolly-15k-dutch")["train"]

    def get_data(row) -> Dict[Literal["dialog", "source"], Union[str, Dialog]]:
        extra_input = " " + row["context"] if "context" in row and row["context"] else ""
        return {
            "dialog": [
                {
                    "role": "user",
                    "content": row["instruction"] + extra_input,
                },
                {
                    "role": "assistant",
                    "content": row["response"],
                },
            ],
            "source": "BramVanroy/dolly-15k-dutch",
        }

    dolly_ds = dolly_ds.map(get_data, remove_columns=dolly_ds.column_names)

    return dolly_ds


def process_alpaca():
    alpaca_ds = load_dataset("BramVanroy/alpaca-cleaned-dutch")["train"]

    def get_data(row) -> Dict[Literal["dialog", "source"], Union[str, Dialog]]:
        extra_input = " " + row["input"] if "input" in row and row["input"] else ""
        return {
            "dialog": [
                {
                    "role": "user",
                    "content": row["instruction"] + extra_input,
                },
                {
                    "role": "assistant",
                    "content": row["output"],
                },
            ],
            "source": "BramVanroy/alpaca-cleaned-dutch",
        }

    alpaca_ds = alpaca_ds.map(get_data, remove_columns=alpaca_ds.column_names)

    return alpaca_ds


def process_baize(
    dataset_name: str,
    system_prefix: str = "Het gesprek tussen de mens en de AI-assistent.",
    user_prefix: str = "[|Human|]",
    assistant_prefix: str = "[|AI|]",
):
    ds = load_dataset(dataset_name)["train"]

    def get_data(row) -> Dict[Literal["dialog", "source"], Union[str, Dialog]]:
        text = row["input"].replace(system_prefix, "")
        turns = [tstrip for t in text.split(user_prefix) if (tstrip := t.strip())]

        data = []
        for turn in turns:
            try:
                user, assistant = turn.split(assistant_prefix, 1)
            except ValueError:
                # If the data contains malformed data (likely mistranslated), then some errors may occur
                continue

            data.extend(
                [
                    {
                        "role": "user",
                        "content": user.strip(),
                    },
                    {
                        "role": "assistant",
                        "content": assistant.strip(),
                    },
                ]
            )

        return {"dialog": data, "source": dataset_name}

    ds = ds.map(get_data, remove_columns=ds.column_names)

    return ds


def main(push_to_hub: bool = True):
    dolly_ds = process_dolly()
    print(f"Finished processing Dolly. Size: {len(dolly_ds):,}")

    quora_ds = process_baize("BramVanroy/quora-chat-dutch")
    print(f"Finished processing Quora. Size: {len(quora_ds):,}")

    stackoverflow_ds = process_baize("BramVanroy/stackoverflow-chat-dutch")
    print(f"Finished processing Stack Overflow. Size: {len(stackoverflow_ds):,}")

    alpaca_ds = process_alpaca()
    print(f"Finished processing Alpaca. Size: {len(alpaca_ds):,}")

    dutch_chat_datasets = datasets.concatenate_datasets([dolly_ds, quora_ds, stackoverflow_ds, alpaca_ds])
    print(f"Total dataset size: {len(dutch_chat_datasets):,}")

    if push_to_hub:
        dutch_chat_datasets.push_to_hub("dutch_chat_datasets")


if __name__ == "__main__":
    main()

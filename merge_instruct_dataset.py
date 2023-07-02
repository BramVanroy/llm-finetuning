from typing import Optional, List
from datasets import load_dataset, concatenate_datasets


def merge(datasets: List[str], cols: List[str], repo_id: str, seed: Optional[int] = None):
    num_datasets = len(datasets)
    first_cols = [col.split(",")[0] for col in cols]
    # Map of dataset_idx to a dictionary that contains a mapping between its column names and its "new" column names
    # which are the column names of the first item
    col_map = {
        ds_idx: {
            col.split(",")[ds_idx]: first_cols[col_idx] for col_idx, col in enumerate(cols)
        }
        for ds_idx in range(num_datasets)
    }

    updated_datasets = []
    for ds_idx, ds_name in enumerate(datasets):
        ds = load_dataset(ds_name)["train"]
        print(ds_name, ds)
        ds = ds.add_column("dataset_source", [ds_name] * len(ds))
        ds = ds.rename_columns(col_map[ds_idx])
        ds = ds.select_columns(first_cols+["dataset_source"])
        updated_datasets.append(ds)

    merged_ds = concatenate_datasets(updated_datasets)

    if seed is not None:
        merged_ds = merged_ds.shuffle(seed=seed)

    merged_ds.push_to_hub(repo_id)


def main():
    import argparse
    cparser = argparse.ArgumentParser(description="Merge given datasets. Only merges the training set and discards the rest")
    cparser.add_argument("--datasets", help="datasets to merge, must be available on the HF hub", nargs="+")
    cparser.add_argument("--cols", help="a list of a comma-separated items of columns to merge across the datasets."
                                        " E.g., 'ds1_col1,ds2_column1 ds1_col2,ds2_column'", nargs="+")
    cparser.add_argument("--repo_id", help="repo_id of the Hugging Face repository where to upload the data to")
    cargs = cparser.parse_args()
    merge(**vars(cargs))


if __name__ == "__main__":
    main()

import json
from pathlib import Path
from typing import Callable

import pandas as pd
import numpy as np


def df_to_df_dataset(
    df: pd.DataFrame,
    size: int | None = 32,
    seed: int = 42,
):
    """
    Sample a subset of a DataFrame based on unique protocol IDs.

    Args:
        df (pd.DataFrame): The DataFrame to sample from.
        size (int|None): Number of selected protocol IDs. Defaults to 32.
            Set to `None` to select all protocol IDs.
        seed (int): Random seed for reproducibility. Defaults to 42.

    Returns:
        pd.DataFrame: A DataFrame containing the sampled subset.
    """
    np.random.seed(seed)
    prot_ids_dataset = np.random.choice(df["prot_id"].unique(), size or len(df))
    df_dataset = df[df["prot_id"].isin(prot_ids_dataset)]
    return df_dataset


def df_chunks_to_doccano(
    input_path: Path,
    output_path: Path,
    size: int | None = 32,
    seed: int = 42,
):
    """
    Generate a dataset of chunks in JSONL format for Doccano.

    Args:
        input_path (Path): Path to the input DataFrame of chunks in Parquet format.
        output_path (Path): Path to save the resulting JSONL dataset.
        size (int|None): Number of selected protocol IDs. Defaults to 32.
            Set to `None` to select all protocol IDs.
        seed (int): Random seed for reproducibility. Defaults to 42.
    """
    df_dataset = df_to_df_dataset(pd.read_parquet(input_path), size, seed)

    with open(output_path, "w") as f:
        for row in df_dataset.itertuples():
            line = json.dumps(
                {
                    "prot_id": row.prot_id,
                    "prot_info": row.prot_info,
                    "eml_id": row.eml_id,
                    "eml_filename": row.eml_filename,
                    "chk_id": row.chk_id,
                    "text": row.chk_text,
                    "label": [],
                }
            )
            f.write(line + "\n")


def df_chunks_to_batch(
    input_path: Path,
    output_path: Path,
    body_fn: Callable,
    model: str,
    size: int | None = 32,
    seed: int = 42,
):
    """
    Generate a dataset of chunks in JSONL format for OpenAI Batch API.

    Args:
        input_path (Path): Path to the input DataFrame of chunks in Parquet format.
        output_path (Path): Path to save the resulting JSONL dataset.
        body_fn (callable): A function to generate the body of the request.
        model (str): The model to use for the request.
        size (int|None): Number of selected protocol IDs. Defaults to 32.
            Set to `None` to select all protocol IDs.
        seed (int): Random seed for reproducibility. Defaults to 42.
    """
    df_dataset = df_to_df_dataset(pd.read_parquet(input_path), size, seed)

    with open(output_path, "w") as f:
        for row in df_dataset.itertuples():
            line = json.dumps(
                {
                    "custom_id": row.chk_id,
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": body_fn(row.chk_text, model),
                }
            )
            f.write(line + "\n")


def doccano_to_df(input_path: Path) -> pd.DataFrame:
    """
    Convert a Doccano JSONL file to a DataFrame.

    Args:
        input_path (Path): Path to the Doccano JSONL file.

    Returns:
        pd.DataFrame: A DataFrame containing the chunk IDs and labels.
    """
    df = pd.DataFrame(columns=["chk_id", "label"])

    with open(input_path, "r") as f:
        for line in f:
            data = json.loads(line)
            assert (
                len(data["label"]) == 1
            ), f"Invalid label: {data['label']}, chk_id: {data['chk_id']}"
            df.loc[len(df)] = [data["chk_id"], data["label"]]

    return df


def batch_response_to_df(input_path: Path) -> pd.DataFrame:
    """
    Convert a batch response JSONL file to a DataFrame.

    Args:
        input_path (Path): Path to the batch response JSONL file.

    Returns:
        pd.DataFrame: A DataFrame containing the chunk IDs and response content.
    """
    df = pd.DataFrame(columns=["chk_id", "content"])

    with open(input_path, "r") as f:
        for line in f:
            data = json.loads(line)
            if data["error"] is not None:
                print(f"Error while parsing {data['id']}. {data['error']}")
                continue

            response = data["response"]
            if response["status_code"] != 200:
                print(
                    f"Error while parsing {data['id']}."
                    f"Status code: {response['status_code']}"
                )
                continue

            choice = response["body"]["choices"][0]
            if choice["finish_reason"] != "stop":
                print(
                    f"Error while parsing {data['id']}."
                    f"Finish reason: {choice['finish_reason']}"
                )
                continue

            try:
                content = json.loads(choice["message"]["content"])
                df.loc[len(df)] = [data["custom_id"], content]

            except json.decoder.JSONDecodeError:
                print(f"Error while parsing {data['id']}. Content is not JSON.")

    return df


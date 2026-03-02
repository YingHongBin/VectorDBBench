import os
import random

from hakesclient.index.build import (
    init_hakes_params,
    build_dataset,
    train_hakes_params,
    recenter_ivf,
)
import numpy as np
import torch
import polars as pl

INDEX_DIR = "/home/hongbin/workspace/HAKES/index"
COLLECTION_NAME = "poc"
DATA_PATH = "./dataset/cohere/cohere_medium_1m/shuffle_train.parquet"

def init_index(collection_name):
    os.makedirs(os.path.join(INDEX_DIR, COLLECTION_NAME), exist_ok=True)
    os.makedirs(os.path.join(INDEX_DIR, COLLECTION_NAME, "checkpoint_0"), exist_ok=True)
    dir = os.path.join(INDEX_DIR, collection_name, "checkpoint_0")

    seed = 0
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Load data from parquet file
    print(f"Loading data from {DATA_PATH}...")
    df = pl.read_parquet(DATA_PATH)
    data = np.stack(df["emb"].to_list()).astype(np.float32)
    print(f"Loaded {data.shape[0]} vectors with dimension {data.shape[1]}")
    
    # Normalize for cosine similarity (IP metric)
    data = data / np.linalg.norm(data, axis=1, keepdims=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    index = init_hakes_params(data, 384, 100, "ip")
    index.set_fixed_assignment(True)
    index.save_as_hakes_index(os.path.join(dir, "findex.bin"))
    sample_ratio = 0.1
    dataset = build_dataset(data, sample_ratio=sample_ratio, nn=50)
    train_hakes_params(
        model=index,
        dataset=dataset,
        epochs=3,
        batch_size=128,
        lr_params={"vt": 1e-4, "pq": 1e-4, "ivf": 0},
        loss_weight={
            "vt": "rescale",
            "pq": 1,
            "ivf": 0,
        },
        temperature=1,
        loss_method="hakes",
        device=device,
    )
    recenter_ivf(index, data, sample_ratio)
    index.save_as_hakes_index(os.path.join(dir, "uindex.bin"))

if __name__ == "__main__":
    init_index(COLLECTION_NAME)

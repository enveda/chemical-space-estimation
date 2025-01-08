import os
import pandas as pd
from scipy.sparse import csr_matrix, vstack
from tqdm import tqdm
from itertools import chain
from scipy.sparse import save_npz
import numpy as np

DATA_DIR = "../data/"

def process_file_to_csr(file_path, row_map, col_map):
    """Read a file and convert it to a CSR matrix with global indices."""
    df = pd.read_parquet(file_path)
    expanded_df = (
        df.explode(["entropy_row_uid", "entropy_score"])
        .reset_index(drop=True)
        .fillna(0)
    )

    # Map row_uid and entropy_row_uid to global indices
    expanded_df["global_row"] = expanded_df["row_uid"].map(row_map)
    expanded_df["global_col"] = expanded_df["entropy_row_uid"].map(col_map)

    # Create sparse matrix
    row = expanded_df["global_row"].values
    col = expanded_df["global_col"].values
    data = expanded_df["entropy_score"].values
    shape = (len(row_map), len(col_map))

    return csr_matrix((data, (row, col)), shape=shape)


def main():
    # Directory containing files
    base_dir = f"{DATA_DIR}/processed/enpkg_with_ms_data_cleaned_entropy_scores/"
    files = os.listdir(base_dir)
    file_paths = [os.path.join(base_dir, f) for f in files]

    # Collect all unique row and column indices for global mapping
    row_ids = set()
    col_ids = set()

    row_uid_df = pd.DataFrame()

    for file_path in tqdm(file_paths, desc="Collecting IDs"):
        df = pd.read_parquet(file_path)
        row_ids.update(df["row_uid"].unique())
        col_ids.update(chain.from_iterable(df["entropy_row_uid"]))

        row_uid_df = pd.concat([row_uid_df, df[["row_uid"]]])

    # Create global row and column mappings
    row_map = {row_id: i for i, row_id in enumerate(sorted(row_ids))}
    col_map = {col_id: i for i, col_id in enumerate(sorted(col_ids))}

    # Initialize the combined sparse matrix
    combined_sparse_matrix = csr_matrix((len(row_map), len(col_map)))

    # Process files and build the combined matrix
    for file_path in tqdm(file_paths, desc="Processing Files"):
        sparse_matrix = process_file_to_csr(file_path, row_map, col_map)
        combined_sparse_matrix += sparse_matrix

    # Save sparse matrix
    save_npz(
        f"{DATA_DIR}/processed/entropy_scores_matrix.npz",
        combined_sparse_matrix,
    )

    # Save row and column names separately
    row_names = np.array(list(row_map.keys()), dtype=object)
    col_names = np.array(list(col_map.keys()), dtype=object)
    np.savez(
        f"{DATA_DIR}/processed/entropy_scores_metadata.npz",
        row_names=row_names,
        col_names=col_names,
    )


if __name__ == "__main__":
    main()

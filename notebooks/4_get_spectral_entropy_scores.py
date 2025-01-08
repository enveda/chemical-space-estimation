import pandas as pd
import numpy as np
import os
import multiprocessing

# Note that this is internal code that is a wrapper around the ms_entropy
# (https://msentropy.readthedocs.io/en/latest/classical_entropy_similarity.html)
# package. Users wanting to replicate this code can use the ms_entropy package

# the ppm filter returns the indices of the library that are within the ppm tolerance
# by using this function
# def call_ppm_filter(self, query_df, ref_df):
    # ref_inds = []
    # for i in range(len(query_df)):
    #     qmz = query_df.iloc[i][query_precursor_mz_col]
    #     ppm = 1e6 * ((ref_df[reference_precursor_mz_col] - qmz) / qmz)
    #     ref_inds.append(np.where(np.abs(ppm) < ppm_tol)[0])

    # return ref_inds

from spectral_inference_zoo.library_search import (
    SpectralEntropyRank,
    PPMFilter,
    LibrarySearch,
)

DATA_DIR = "../data/"

def process_chunk(j, data, lib_searcher, output_dir):
    """Process a single chunk of data."""
    # check if file already exists
    if os.path.exists(f"{output_dir}/part_{j}.pq"):
        return f"{output_dir}/part_{j}.pq"
    else:

        # Get predictions
        current_data = data.iloc[j * 1000 : (j + 1) * 1000]
        preds = lib_searcher.run(current_data)
        preds.rename({c: f"entropy_{c}" for c in preds.columns}, axis=1, inplace=True)

        # Merge predictions with current data
        current_data = current_data.merge(preds, left_index=True, right_index=True)
        output_path = f"{output_dir}/part_{j}.pq"
        current_data.drop(
            [
                "entropy_smiles_2d",
                "entropy_ms2_precursor_mz",
                "entropy_normalized_mzs",
                "entropy_normalized_intensities",
            ],
            axis=1,
            inplace=True,
        )
        current_data.to_parquet(output_path)

        return output_path


def main():
    # get data
    data = pd.read_parquet(
        f"{DATA_DIR}/processed/enpkg_with_ms_data_cleaned.pq"
    )

    # Preprocess data
    data["ion_mode"] = data.charge.apply(lambda x: "positive" if x > 0 else "negative")
    data.rename(
        {
            "mz_list": "normalized_mzs",
            "i_list": "normalized_intensities",
            "precursor_mz": "ms2_precursor_mz",
            "smiles": "smiles_2d",
        },
        axis=1,
        inplace=True,
    )

    # Create model object
    ranker = SpectralEntropyRank(
        mz_col="normalized_mzs",
        intensity_col="normalized_intensities",
    )
    filter = PPMFilter(
        reference_precursor_mz_col="ms2_precursor_mz",
        query_precursor_mz_col="ms2_precursor_mz",
        ppm_tol=10,
    )
    lib_searcher = LibrarySearch(
        library_ranker=ranker,
        library_filter=filter,
        top_k=len(data),
        return_fields=[
            "row_uid",
            "smiles_2d",
            "ms2_precursor_mz",
            "normalized_mzs",
            "normalized_intensities",
        ],
    )

    # Get predictions
    lib_searcher.prepare(data)
    output_dir = f"{DATA_DIR}/processed/enpkg_with_ms_data_cleaned_entropy_scores/"
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    num_chunks = int(np.ceil(len(data) / 1000))

    # Create a pool of workers
    with multiprocessing.Pool(processes=10) as pool:
        # Distribute the work
        args = [(j, data, lib_searcher, output_dir) for j in range(num_chunks)]
        results = pool.starmap(process_chunk, args)


if __name__ == "__main__":
    main()

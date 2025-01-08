import os
import urllib
import pandas as pd
from tqdm import tqdm
from matchms.importing import load_from_mgf
import seaborn as sns
from rdkit import Chem
from functools import cache

from rdkit.Chem.Scaffolds.MurckoScaffold import MurckoScaffoldSmiles

DATA_DIR = "../data"
os.makedirs(DATA_DIR, exist_ok=True)

DEFAULT_COLORMAP = {
    "white": "#FFFFFF",
    "black": "#000000",
    "grey": "#808080",
    "deep-pink": "#F10A84",
    "yellow-green": "#BCD20B",
    "slate-blue": "#6942D9",
    "khakhi": "#F6E547",
    "cornflower-blue": "#5C7CFC",
    "orange-red": "#FA5F0D",
    "olive-drab": "#6A8D3E",
    "crimson": "#E93848",
    "sky-blue": "#80BDE9",
}

DEFAULT_PALETTE_RAINBOW = sns.color_palette(
    [v for k, v in DEFAULT_COLORMAP.items() if k not in ["white", "black"]]
)
DEFAULT_PALETTE_RAINBOW_PLOTLY = [
    f"#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}"
    for r, g, b in DEFAULT_PALETTE_RAINBOW
]


def mfg2df(file_path: str) -> pd.DataFrame:
    """Converting mgf file to pandas DataFrame"""
    specs = load_from_mgf(file_path)
    ms2_specs = [spec for spec in specs]

    dfs = []

    for spec in ms2_specs:
        mz = spec.peaks.to_numpy[:, 0]
        i = spec.peaks.to_numpy[:, 1]

        spec_data = pd.DataFrame([spec.metadata])
        spec_data["mz_list"] = [mz]
        spec_data["i_list"] = [i]

        dfs.append(spec_data)

    return pd.concat(dfs, ignore_index=True)


def download_massive_data(df: pd.DataFrame) -> None:
    """Download ENPKG data from MASSIVE Database."""
    FOLDER_DIR = f"{DATA_DIR}/raw"
    for massive_idx, file_name in tqdm(df[["massive_idx", "mgf_file"]].values):
        fname = file_name.split(".")[0]
        pq_file_name = f"{massive_idx}_{fname}.pq"

        if os.path.exists(os.path.join(FOLDER_DIR, pq_file_name)):
            continue

        if massive_idx == "MSV000093464":  # Korean plant collection
            ftp_file_path = f"https://massive.ucsd.edu/ProteoSAFe/DownloadResultFile?file=f.MSV000093464/peak/MSV000086161/{file_name}&forceDownload=true"
        elif massive_idx == "MSV000087728":  # PF library
            if "pos" in file_name:
                ftp_file_path = f"https://massive.ucsd.edu/ProteoSAFe/DownloadResultFile?file=f.MSV000087728/updates/2021-12-03_pmallard_cff635d5/peak/individual_mgf_files/{file_name}&forceDownload=true"
            else:
                ftp_file_path = f"https://massive.ucsd.edu/ProteoSAFe/DownloadResultFile?file=f.MSV000087728/updates/2022-07-06_pmallard_5ffd2d04/peak/individual_mgf_neg/{file_name}&forceDownload=true"
        elif massive_idx == "MSV000088521":  # GNPS Waltheria indica
            ftp_file_path = f"https://massive.ucsd.edu/ProteoSAFe/DownloadResultFile?file=f.MSV000088521/updates/2023-05-18_Arnaud_Gaudry_1_4ac30a51/peak/{file_name}&forceDownload=true"

        # Download file
        urllib.request.urlretrieve(
            ftp_file_path, os.path.join(FOLDER_DIR, f"{massive_idx}_{file_name}")
        )
        feature_df = mfg2df(os.path.join(FOLDER_DIR, f"{massive_idx}_{file_name}"))
        feature_df["massive_idx"] = massive_idx
        feature_df["file_name"] = file_name

        # Save as parquet
        feature_df.to_parquet(os.path.join(FOLDER_DIR, pq_file_name))
        # all_features.append(feature_df)

        # Remove downloaded file
        os.remove(os.path.join(DATA_DIR, f"{massive_idx}_{file_name}"))


def get_ms_data(df: pd.DataFrame) -> None:
    """Compile a MS tabel for the complete dataset.

    This function requires a lot of ram memory.
    """
    FOLDER_DIR = f"{DATA_DIR}/raw"
    if len(os.listdir(FOLDER_DIR)) == 0:
        download_massive_data(df)

    df[["data_type", "massive_idx", "mgf_file", "file_type", "scan_num"]] = df[
        "massive_id"
    ].str.split(":", expand=True)
    df["main_info_file"] = df["massive_idx"] + "_" + df["mgf_file"]

    """Subsetting the spectra to the scan numbers."""
    all_feature_counts = set()

    for fname in tqdm(df["main_info_file"].unique()):
        pq_file_name = fname.replace(".mgf", ".pq")
        feature_df = pd.read_parquet(os.path.join(DATA_DIR, pq_file_name))

        k = df["main_info_file"] == fname
        scan_nums = df[k]["scan_num"].unique().tolist()

        # Subset to scan number
        m = feature_df["scans"].isin(scan_nums)
        feature_df = feature_df[m]

        feature_df["massive_idx"] = fname.split("_")[0]

        feature_df.to_parquet(f"{FOLDER_DIR}/{fname.split('.')[0]}_trimed.pq")

    print("Total number of features", len(all_feature_counts))

    """Combining all the features."""
    features_df = []

    for file in os.listdir(FOLDER_DIR):
        if not file.endswith(".pq"):
            continue
        features_df.append(pd.read_parquet(f"{FOLDER_DIR}/{file}"))

    print("Number of non-empty files", len(features_df))

    combined_featues = pd.concat(features_df)

    cols = [
        "feature_id",
        "scans",
        "charge",
        "retention_time",
        "ms_level",
        "precursor_mz",
        "massive_idx",
        "file_name",
    ]
    combined_featues.drop_duplicates(subset=cols, inplace=True)

    combined_featues.to_parquet(os.path.join(FOLDER_DIR, "all_enpkg_ms_features.pq"))


def load_lotus_data():
    """Load data from LOTUS database (https://zenodo.org/records/7534062)."""
    df = pd.read_csv(
        f"{DATA_DIR}/raw/validated_referenced_structure_organism_pairs.tsv.gz",
        sep="\t",
        usecols=["organismCleaned", "structureCleaned_smiles2D"],
    )
    df.columns = ["plant", "smiles"]
    df["source"] = "lotus"
    df.drop_duplicates(inplace=True)

    df.to_csv(
        f"{DATA_DIR}/processed/lotus_data.tsv.gz",
        sep="\t",
        index=False,
        compression="gzip",
    )


def load_coconut_data():
    """Load data from COCONUT database (https://coconut.naturalproducts.net/download)."""
    df = pd.read_csv(
        f"{DATA_DIR}/raw/coconut_complete-10-2024.csv.gz", low_memory=False
    )

    df = df[["canonical_smiles", "organisms"]]

    updated_data = []

    for smiles, plants in tqdm(df.values):
        if pd.isna(plants):
            continue

        for plant in plants.split("|"):
            updated_data.append({"plant": plant, "smiles": smiles})

    update_df = pd.DataFrame(updated_data)
    update_df["source"] = "coconut"

    update_df.to_csv(
        f"{DATA_DIR}/processed/coconut_data.tsv.gz",
        index=False,
        sep="\t",
        compression="gzip",
    )


def load_ref_data():
    """Load data from reference database."""
    if os.path.exists(f"{DATA_DIR}/processed/lotus_data.tsv.gz"):
        lotus_data = pd.read_csv(
            f"{DATA_DIR}/processed/lotus_data.tsv.gz", sep="\t", compression="gzip"
        )
    else:
        load_lotus_data()
        lotus_data = pd.read_csv(
            f"{DATA_DIR}/processed/lotus_data.tsv", sep="\t", compression="gzip"
        )

    print(f"Lotus data: {lotus_data.shape}")

    if os.path.exists(f"{DATA_DIR}/processed/coconut_data.tsv"):
        coconut_data = pd.read_csv(
            f"{DATA_DIR}/processed/coconut_data.tsv.gz", sep="\t", compression="gzip"
        )
    else:
        load_coconut_data()
        coconut_data = pd.read_csv(
            f"{DATA_DIR}/processed/coconut_data.tsv.gz", sep="\t", compression="gzip"
        )

    print(f"Coconut data: {coconut_data.shape}")

    ref_data = pd.concat([lotus_data, coconut_data], ignore_index=True)
    return ref_data


@cache
def canonicalize_smiles(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        return Chem.MolToSmiles(mol)
    except:
        return pd.NA


@cache
def get_murcko_scaffold(smiles):
    try:
        scaffold = MurckoScaffoldSmiles(smiles)
        return scaffold
    except:
        return pd.NA

"""
PTB-XL dataset download utility.

Downloads the dataset from PhysioNet as a zip archive and extracts it.
Run this on the training server before starting experiments.
"""
import os
import argparse
import subprocess
import zipfile


PTBXL_URL = (
    "https://physionet.org/static/published-projects/ptb-xl/"
    "ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3.zip"
)


def download_ptbxl(data_dir: str = "data/raw"):
    """Download and extract PTB-XL from PhysioNet.

    Args:
        data_dir: Parent directory for the dataset. The archive extracts
            into a subdirectory automatically.
    """
    os.makedirs(data_dir, exist_ok=True)
    zip_path = os.path.join(data_dir, "ptb-xl-1.0.3.zip")

    if os.path.exists(os.path.join(data_dir, "ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3")):
        print("PTB-XL already downloaded, skipping.")
        return

    print(f"Downloading PTB-XL (~3GB) to {zip_path} ...")
    subprocess.run(
        ["wget", "-c", "-O", zip_path, PTBXL_URL],
        check=True,
    )

    print("Extracting archive ...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(data_dir)

    os.remove(zip_path)
    print("Done. Dataset extracted to", data_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/raw")
    args = parser.parse_args()
    download_ptbxl(args.data_dir)

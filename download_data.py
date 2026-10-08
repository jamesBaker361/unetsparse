# Downloads and unzips the raw datasets into {data_root}/{dataset}/raw.
#   celeba: CelebAMask-HQ.zip (3.15 GB) from the HF mirror liusq/CelebAMask-HQ
#           (images 1024px, 19-class masks 512px, attributes)
#   awa2:   AwA2-data.zip (13 GB) from cvml.ista.ac.at - resumable download
# Already-extracted datasets are skipped.

import os
import argparse
import zipfile

import requests

from common import DEFAULT_DATA_ROOT, raw_dir

AWA2_URL = "https://cvml.ista.ac.at/AwA2/AwA2-data.zip"

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="all", choices=["all", "celeba", "awa2"])
parser.add_argument("--data_root", type=str, default=DEFAULT_DATA_ROOT)
parser.add_argument("--keep_zip", action="store_true")


def extract(zip_path: str, dest: str, keep_zip: bool):
    print("extracting", zip_path)
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(dest)
    open(os.path.join(dest, ".extracted"), "w").close()
    if not keep_zip:
        os.remove(zip_path)


def download_celeba(root: str, keep_zip: bool):
    from huggingface_hub import hf_hub_download
    dest = raw_dir(root, "celeba")
    if os.path.exists(os.path.join(dest, ".extracted")):
        print("celeba already extracted")
        return
    os.makedirs(dest, exist_ok=True)
    zip_path = hf_hub_download(repo_id="liusq/CelebAMask-HQ", filename="CelebAMask-HQ.zip",
                               repo_type="dataset", local_dir=dest)
    extract(zip_path, dest, keep_zip)


def download_awa2(root: str, keep_zip: bool):
    dest = raw_dir(root, "awa2")
    if os.path.exists(os.path.join(dest, ".extracted")):
        print("awa2 already extracted")
        return
    os.makedirs(dest, exist_ok=True)
    zip_path = os.path.join(dest, "AwA2-data.zip")
    done = os.path.getsize(zip_path) if os.path.exists(zip_path) else 0
    headers = {"Range": f"bytes={done}-"} if done else {}
    with requests.get(AWA2_URL, stream=True, headers=headers, timeout=60) as r:
        if r.status_code == 416:  # range past the end: already complete
            pass
        else:
            r.raise_for_status()
            mode = "ab" if r.status_code == 206 else "wb"
            total = int(r.headers.get("Content-Length", 0)) + (done if mode == "ab" else 0)
            with open(zip_path, mode) as f:
                for i, chunk in enumerate(r.iter_content(chunk_size=1 << 22)):
                    f.write(chunk)
                    if i % 256 == 0:
                        print(f"awa2 {f.tell() / 1e9:.2f} / {total / 1e9:.2f} GB", flush=True)
    extract(zip_path, dest, keep_zip)


def main(args):
    if args.dataset in ("all", "celeba"):
        download_celeba(args.data_root, args.keep_zip)
    if args.dataset in ("all", "awa2"):
        download_awa2(args.data_root, args.keep_zip)


if __name__ == "__main__":
    main(parser.parse_args())

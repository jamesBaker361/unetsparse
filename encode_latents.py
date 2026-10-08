# Encodes every image of a dataset (center crop -> 256px, plus its horizontal
# flip) into one input space and writes {root}/{dataset}/latents_{space}.npy,
# (N, 2, C, S, S) float16. The UNets train on these cached tensors.

import os
import argparse

import numpy as np
import torch

from common import (DEFAULT_DATA_ROOT, DATASETS, SPACES, Space, dataset_dir, meta_path, latents_path,
                    load_json, load_square, pil_to_tensor)

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, required=True, choices=DATASETS)
parser.add_argument("--space", type=str, required=True, choices=SPACES)
parser.add_argument("--data_root", type=str, default=DEFAULT_DATA_ROOT)
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--num_workers", type=int, default=4)


class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, root: str, paths: list):
        self.root, self.paths = root, paths

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        return pil_to_tensor(load_square(os.path.join(self.root, self.paths[i])))


def main(args):
    out_path = latents_path(args.data_root, args.dataset, args.space)
    if os.path.exists(out_path):
        print("already exists:", out_path)
        return
    device = "cuda" if torch.cuda.is_available() else "cpu"
    meta = load_json(meta_path(args.data_root, args.dataset))
    space = Space(args.space, device)
    loader = torch.utils.data.DataLoader(ImageDataset(dataset_dir(args.data_root, args.dataset), meta["images"]),
                                         batch_size=args.batch_size, num_workers=args.num_workers)
    n = len(meta["images"])
    tmp = out_path + ".tmp.npy"
    out = np.lib.format.open_memmap(tmp, mode="w+", dtype=np.float16,
                                    shape=(n, 2, space.channels, space.size, space.size))
    start = 0
    for b, x in enumerate(loader):
        z = space.encode(torch.cat([x, x.flip(-1)]))
        k = len(x)
        out[start:start + k, 0] = z[:k].cpu().numpy().astype(np.float16)
        out[start:start + k, 1] = z[k:].cpu().numpy().astype(np.float16)
        start += k
        if b % 50 == 0:
            print(f"{start}/{n} std={z.std().item():.3f} mean={z.mean().item():.3f}", flush=True)
    out.flush()
    del out
    os.replace(tmp, out_path)
    sample = np.load(out_path, mmap_mode="r")[:2000].astype(np.float32)
    print(f"wrote {out_path}; latent mean {sample.mean():.3f} std {sample.std():.3f}")


if __name__ == "__main__":
    main(parser.parse_args())

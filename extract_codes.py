# Runs real images through one arm's UNet at a single small t and saves, for
# every patch of every SAE site (8x8 = 64 patches per image, (b, h, w) order):
#   {split}_{site}_idx.npy / _val.npy   top-k SAE codes (int16 / float16, (n_patches, k))
#   {split}_{site}_raw.npy              the site's activation before the SAE (float16, (n_patches, C))
#   {split}_input.npy                   the clean input latent cut into the same 8x8 patches
#                                       ((n_patches, C * f * f)), an "input space" baseline
#   {split}_images.npy                  image index of each image, in order
# to {root}/codes/{dataset}_{space}_{arm}_t{t}. x_t = (1 - t) x_0 + t eps with
# a fixed eps per image (common.image_noise), the same for every arm.
#   --arm inline:  joint UNet, SAEs inline (later sites see earlier sites' reconstructions)
#   --arm posthoc: plain UNet, post-hoc SAEs on the side
# train = a random --n_train_images subset of the train split, test = the whole test split.

import os
import argparse

import numpy as np
import torch

from common import (DEFAULT_DATA_ROOT, DATASETS, SPACES, PATCH_GRID, load_unet_and_saes, SAEHooks, noisy,
                    predict_v, image_noise, latents_path, meta_path, load_json, save_json, split_indices,
                    run_dir, codes_dir)

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, required=True, choices=DATASETS)
parser.add_argument("--space", type=str, required=True, choices=SPACES)
parser.add_argument("--arm", type=str, required=True, choices=["inline", "posthoc"])
parser.add_argument("--data_root", type=str, default=DEFAULT_DATA_ROOT)
parser.add_argument("--t", type=float, default=0.1)
parser.add_argument("--n_train_images", type=int, default=5000)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--no_raw", action="store_true", help="skip the raw activations")


def input_patches(x0: torch.Tensor) -> torch.Tensor:
    b, c, s, _ = x0.shape
    f = s // PATCH_GRID
    return x0.reshape(b, c, PATCH_GRID, f, PATCH_GRID, f).permute(0, 2, 4, 1, 3, 5).reshape(b * PATCH_GRID ** 2, -1)


@torch.no_grad()
def extract(args, unet, hooks, sites, latents, indices, device) -> dict:
    out = {"input": []}
    for s in sites:
        out.update({f"{s}_idx": [], f"{s}_val": [], f"{s}_raw": []})
    for start in range(0, len(indices), args.batch_size):
        idx = indices[start:start + args.batch_size]
        x0 = torch.from_numpy(np.array(latents[idx, 0], dtype=np.float32)).to(device)
        eps = image_noise(idx, x0.shape[1:], args.seed).to(device)
        t = torch.full((len(idx),), args.t, device=device)
        predict_v(unet, noisy(x0, eps, t), t)
        out["input"].append(input_patches(x0).half().cpu().numpy())
        for s in sites:
            x, _, z, _ = hooks.records[s]
            top = torch.topk(z.float(), hooks.saes[s].k, dim=-1)
            out[f"{s}_idx"].append(top.indices.to(torch.int16).cpu().numpy())
            out[f"{s}_val"].append(top.values.half().cpu().numpy())
            if not args.no_raw:
                out[f"{s}_raw"].append(x.half().cpu().numpy())
        print(f"{start + len(idx)}/{len(indices)}", flush=True)
    return {k: np.concatenate(v) for k, v in out.items() if v}


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = codes_dir(args.data_root, args.dataset, args.space, args.arm, args.t)
    os.makedirs(out_dir, exist_ok=True)
    unet, saes, sae_cfg = load_unet_and_saes(run_dir(args.data_root, args.dataset, args.space, args.arm),
                                             args.space, saes_too=True, device=device)
    sites = sae_cfg["sites"]
    hooks = SAEHooks(unet, saes, sites, mode="inline" if args.arm == "inline" else "side")
    meta = load_json(meta_path(args.data_root, args.dataset))
    latents = np.load(latents_path(args.data_root, args.dataset, args.space), mmap_mode="r")

    rng = np.random.default_rng(args.seed)
    train = split_indices(meta, "train")
    train = np.sort(rng.choice(train, min(args.n_train_images, len(train)), replace=False))
    for split, indices in (("train", train), ("test", split_indices(meta, "test"))):
        arrays = extract(args, unet, hooks, sites, latents, indices, device)
        np.save(os.path.join(out_dir, f"{split}_images.npy"), indices)
        for name, arr in arrays.items():
            np.save(os.path.join(out_dir, f"{split}_{name}.npy"), arr)
    save_json(os.path.join(out_dir, "info.json"),
              {**sae_cfg, "t": args.t, "seed": args.seed, "n_latents": saes[sites[0]].n_latents,
               "dataset": args.dataset, "space": args.space})
    print("wrote", out_dir)


if __name__ == "__main__":
    main(parser.parse_args())

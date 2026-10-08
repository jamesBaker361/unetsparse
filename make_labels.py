# Builds {root}/{dataset}/meta.json (image list, train/test split, classes)
# and {root}/{dataset}/labels.npy, (N, 64, 64) uint8 label maps aligned with
# the center-cropped 256px images every space encodes.
#
#   celeba: 19 classes (0 = background) from the CelebAMask-HQ part masks,
#           composed in g_mask.py's overwrite order at 512px, then each 8x8
#           pixel block takes its majority class.
#   awa2:   0 = background, 1..50 = species. SAM3 is prompted with the species
#           name on the center crop; the union of its instance masks is
#           area-downsampled to 64x64 and thresholded at 0.5. Per-image results
#           are cached in {root}/awa2/sam/{i}.npz so the SAM3 pass can be sharded
#           (--shard / --n_shards); labels.npy is written once every image has
#           its cache. meta["found"] is False where SAM3 found nothing - those
#           images are left out of probing (common.split_indices).

import os
import argparse
from multiprocessing import Pool

import numpy as np
from PIL import Image

from common import (DEFAULT_DATA_ROOT, CELEBA_CLASSES, LABEL_SIZE, raw_dir, meta_path, labels_path,
                    dataset_dir, find_dir, find_file, load_json, save_json, center_crop)

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, required=True, choices=["celeba", "awa2"])
parser.add_argument("--data_root", type=str, default=DEFAULT_DATA_ROOT)
parser.add_argument("--test_frac", type=float, default=0.1)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--num_workers", type=int, default=4)
parser.add_argument("--shard", type=int, default=0)
parser.add_argument("--n_shards", type=int, default=1)
parser.add_argument("--sam_size", type=int, default=1008, help="crop is resized to this before SAM3")


def random_split(n: int, test_frac: float, seed: int) -> list:
    rng = np.random.default_rng(seed)
    test = set(rng.choice(n, int(round(test_frac * n)), replace=False).tolist())
    return ["test" if i in test else "train" for i in range(n)]


def block_majority(label: np.ndarray, n_classes: int, out: int = LABEL_SIZE) -> np.ndarray:
    '''(H, W) int label map -> (out, out) majority label of each (H/out)x(W/out) block.'''
    f = label.shape[0] // out
    blocks = label.reshape(out, f, out, f).transpose(0, 2, 1, 3).reshape(out, out, f * f)
    counts = (blocks[..., None] == np.arange(n_classes)).sum(2)
    return counts.argmax(-1).astype(np.uint8)


# ----------------------------------------------------------------------------- celeba

def celeba_label(args) -> np.ndarray:
    i, mask_root = args
    folder = os.path.join(mask_root, str(i // 2000))
    label = np.zeros((512, 512), dtype=np.uint8)
    for c, name in enumerate(CELEBA_CLASSES[1:], start=1):
        path = os.path.join(folder, f"{i:05d}_{name}.png")
        if os.path.exists(path):
            m = np.asarray(Image.open(path).convert("L").resize((512, 512), Image.NEAREST)) > 127
            label[m] = c
    return block_majority(label, len(CELEBA_CLASSES))


def make_celeba(args):
    raw = raw_dir(args.data_root, "celeba")
    img_dir = find_dir(raw, "CelebA-HQ-img")
    mask_root = find_dir(raw, "CelebAMask-HQ-mask-anno")
    n = 30000
    images = [os.path.relpath(os.path.join(img_dir, f"{i}.jpg"), dataset_dir(args.data_root, "celeba"))
              for i in range(n)]
    meta = {"dataset": "celeba", "images": images, "split": random_split(n, args.test_frac, args.seed),
            "classes": CELEBA_CLASSES}
    save_json(meta_path(args.data_root, "celeba"), meta)
    with Pool(args.num_workers) as pool:
        labels = pool.map(celeba_label, [(i, mask_root) for i in range(n)], chunksize=64)
    np.save(labels_path(args.data_root, "celeba"), np.stack(labels))
    print("wrote", labels_path(args.data_root, "celeba"))


# ----------------------------------------------------------------------------- awa2

def awa2_meta(args) -> dict:
    path = meta_path(args.data_root, "awa2")
    if os.path.exists(path):
        return load_json(path)
    raw = raw_dir(args.data_root, "awa2")
    jpeg_dir = find_dir(raw, "JPEGImages")
    with open(find_file(raw, "classes.txt")) as f:
        species = [line.split()[1] for line in f if line.strip()]
    images, image_species = [], []
    for s, name in enumerate(species):
        for fn in sorted(os.listdir(os.path.join(jpeg_dir, name))):
            if fn.lower().endswith((".jpg", ".jpeg", ".png")):
                images.append(os.path.relpath(os.path.join(jpeg_dir, name, fn), dataset_dir(args.data_root, "awa2")))
                image_species.append(s)
    meta = {"dataset": "awa2", "images": images, "split": random_split(len(images), args.test_frac, args.seed),
            "species": species, "image_species": image_species,
            "classes": ["background"] + [s.replace("+", " ") for s in species]}
    save_json(path, meta)
    return meta


def sam_cache(root: str, i: int) -> str:
    return os.path.join(root, "awa2", "sam", f"{i}.npz")


def run_sam(args, meta: dict):
    import torch
    from sam3_repo.sam3.model_builder import build_sam3_image_model
    from sam3_repo.sam3.model.sam3_image_processor import Sam3Processor

    todo = [i for i in range(len(meta["images"]))
            if i % args.n_shards == args.shard and not os.path.exists(sam_cache(args.data_root, i))]
    print(f"shard {args.shard}/{args.n_shards}: {len(todo)} images to mask")
    if not todo:
        return
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = Sam3Processor(build_sam3_image_model(device=device), device=device)
    os.makedirs(os.path.dirname(sam_cache(args.data_root, 0)), exist_ok=True)
    for n, i in enumerate(todo):
        image = center_crop(Image.open(os.path.join(dataset_dir(args.data_root, "awa2"), meta["images"][i]))
                            .convert("RGB")).resize((args.sam_size, args.sam_size), Image.BICUBIC)
        query = meta["species"][meta["image_species"][i]].replace("+", " ")
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
            state = processor.set_image(image)
            out = processor.set_text_prompt(state=state, prompt=query)
        masks, scores = out["masks"], out["scores"]
        if len(scores) == 0:
            mask, score = np.zeros((args.sam_size, args.sam_size), dtype=bool), 0.0
        else:
            mask = np.any(masks.squeeze(1).cpu().numpy(), axis=0)
            score = float(scores.float().max().cpu())
        frac = np.asarray(Image.fromarray(mask.astype(np.uint8) * 255).resize((LABEL_SIZE, LABEL_SIZE),
                                                                              Image.BOX)) / 255.0
        np.savez_compressed(sam_cache(args.data_root, i), mask=frac >= 0.5, score=score)
        if n % 500 == 0:
            print(f"{n}/{len(todo)} {query} score={score:.3f} area={mask.mean():.3f}", flush=True)


def make_awa2(args):
    meta = awa2_meta(args)
    run_sam(args, meta)
    n = len(meta["images"])
    missing = [i for i in range(n) if not os.path.exists(sam_cache(args.data_root, i))]
    if missing:
        print(f"{len(missing)} images still need SAM3 masks - labels.npy not written yet")
        return
    labels = np.zeros((n, LABEL_SIZE, LABEL_SIZE), dtype=np.uint8)
    found, scores = [], []
    for i in range(n):
        with np.load(sam_cache(args.data_root, i)) as d:
            labels[i][d["mask"]] = meta["image_species"][i] + 1
            found.append(bool(d["mask"].any()))
            scores.append(float(d["score"]))
    meta["found"], meta["sam_scores"] = found, scores
    save_json(meta_path(args.data_root, "awa2"), meta)
    np.save(labels_path(args.data_root, "awa2"), labels)
    print(f"wrote {labels_path(args.data_root, 'awa2')}; SAM3 found the animal in {np.mean(found):.3f} of images")


if __name__ == "__main__":
    args = parser.parse_args()
    make_celeba(args) if args.dataset == "celeba" else make_awa2(args)

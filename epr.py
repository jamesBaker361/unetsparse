# EPR (Editing Precision Ratio, CASL arXiv:2601.15441 Eq. 14) of each arm's
# class latents, measured on generated images with the trained image
# classifier (train_classifier.py).
#
# For every class c (celeba: 18 parts; awa2: 50 species) and site, the latent
# is the one probe.py selected for c (--selection, SAE codes, positive weight).
# On --n_images test images where c is absent (celeba: below the classifier's
# presence threshold; awa2: other species):
#   1. x_t = (1 - t) x_0 + t eps at --t_edit (same per-image eps as extract_codes.py)
#   2. Euler-denoise to t=0 twice: unedited, and with strength * a_j * d_j added
#      to the site's output at every patch and step (d_j = the latent's unit
#      decoder direction, a_j = its mean activation when active on the train codes)
#   3. decode both, classifier logits, delta = edited - unedited
#   epr_target_change    = mean over images of |delta[c]|
#   epr_nontarget_change = mean over images of mean_{k != c} |delta[k]|
#   EPR                  = epr_target_change / (epr_nontarget_change + 1e-8)
# plus the signed target gain, how often the edited image is classified as c
# (celeba: sigmoid > 0.5, awa2: argmax), and PSNR vs the unedited image.
# --n_random random live latents per class x site are injected the same way
# as a control (latent_kind = "random").
# Rows go to {out_dir}/epr.csv; example panels to {out_dir}/epr_panels/.

import os
import csv
import argparse

import numpy as np
import torch

from common import (DEFAULT_DATA_ROOT, DATASETS, SPACES, Space, load_unet_and_saes, SAEHooks, noisy, predict_v,
                    image_noise, latents_path, labels_path, meta_path, load_json, split_indices, run_dir, codes_dir,
                    tensor_to_pil, image_grid)
from train_classifier import load_classifier, preprocess, presence, classifier_dir
from probe import append_rows

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, required=True, choices=DATASETS)
parser.add_argument("--space", type=str, required=True, choices=SPACES)
parser.add_argument("--arms", type=str, default="inline,posthoc")
parser.add_argument("--sites", type=str, default=None, help="default: every site in sae_config.json")
parser.add_argument("--data_root", type=str, default=DEFAULT_DATA_ROOT)
parser.add_argument("--t", type=float, default=0.1, help="t of the probe run whose latents are used")
parser.add_argument("--coverage", type=float, default=0.5, help="coverage of the probe run whose latents are used")
parser.add_argument("--selection", type=str, default="f1", choices=["f1", "bce"])
parser.add_argument("--probe_csv", type=str, default="outputs/probe_binary.csv")
parser.add_argument("--strengths", type=str, default="1,4")
parser.add_argument("--t_edit", type=float, default=0.5)
parser.add_argument("--sample_steps", type=int, default=50, help="Euler steps for the full t=1..0 range")
parser.add_argument("--n_images", type=int, default=32)
parser.add_argument("--n_random", type=int, default=1)
parser.add_argument("--min_active", type=int, default=100, help="random latents must fire on >= this many train patches")
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--n_panel", type=int, default=4, help="images per example panel (0 = none)")
parser.add_argument("--out_dir", type=str, default="outputs")


def probe_latents(args) -> dict:
    '''(arm, site, task) -> latent from probe_binary.csv; the last matching row wins (latest run).'''
    out = {}
    with open(args.probe_csv) as f:
        for r in csv.DictReader(f):
            if (r["dataset"] == args.dataset and r["space"] == args.space and r["features"] == "sae"
                    and r["selection"] == args.selection and np.isclose(float(r["t"]), args.t)
                    and np.isclose(float(r["coverage"]), args.coverage)):
                out[(r["arm"], r["site"], r["task"])] = int(r["latent"])
    return out


def mean_active(codes: str, site: str, n_latents: int):
    '''Per latent: mean value when active and number of active train patches.'''
    idx = np.load(os.path.join(codes, f"train_{site}_idx.npy"), mmap_mode="r").reshape(-1).astype(np.int64)
    val = np.load(os.path.join(codes, f"train_{site}_val.npy"), mmap_mode="r").reshape(-1).astype(np.float64)
    keep = val > 0
    counts = np.bincount(idx[keep], minlength=n_latents)
    sums = np.bincount(idx[keep], weights=val[keep], minlength=n_latents)
    return sums / np.maximum(counts, 1), counts


def base_images(args, meta, labels, names, min_frac) -> dict:
    '''task -> sorted test image indices where the class is absent (seeded per task, shared by every arm).'''
    test = split_indices(meta, "test")
    if args.dataset == "celeba":
        present = presence(labels[test], len(meta["classes"]), min_frac)
    else:
        species = np.asarray(meta["image_species"])[test]
    out = {}
    for ti, task in enumerate(names):
        absent = test[~present[:, ti]] if args.dataset == "celeba" else test[species != ti]
        pool = absent if len(absent) >= args.n_images else test
        rng = np.random.default_rng(args.seed * 1000 + ti)
        out[task] = np.sort(rng.choice(pool, min(args.n_images, len(pool)), replace=False))
    return out


@torch.no_grad()
def generate(args, unet, hooks, space, classifier, latents, indices, device, inject=None):
    '''Edited (or unedited) denoise from t_edit for every image -> (logits (n, n_out), uint8 images (n, 3, H, W)).'''
    n_steps = max(1, int(round(args.t_edit * args.sample_steps)))
    ts = torch.linspace(args.t_edit, 0.0, n_steps + 1, device=device)
    logits, images = [], []
    for s in range(0, len(indices), args.batch_size):
        idx = indices[s:s + args.batch_size]
        x0 = torch.from_numpy(np.array(latents[idx, 0], dtype=np.float32)).to(device)
        eps = image_noise(idx, x0.shape[1:], args.seed).to(device)
        x = noisy(x0, eps, ts[0].expand(len(idx)))
        hooks.inject = inject or {}
        for i in range(n_steps):
            x = x + (ts[i + 1] - ts[i]) * predict_v(unet, x, ts[i].expand(len(idx))).float()
        hooks.inject = {}
        img = space.decode(x)
        logits.append(classifier(preprocess(img)).float().cpu())
        images.append(((img + 1) * 127.5).round().clamp(0, 255).byte().cpu())
    return torch.cat(logits), torch.cat(images)


def psnr(a: torch.Tensor, b: torch.Tensor) -> np.ndarray:
    mse = ((a.float() - b.float()) ** 2).flatten(1).mean(1).clamp_min(1e-10)
    return (10 * torch.log10(255.0 ** 2 / mse)).numpy()


def classified(logits: torch.Tensor, ti: int, multilabel: bool) -> float:
    return float((logits[:, ti] > 0).float().mean() if multilabel else (logits.argmax(1) == ti).float().mean())


def save_panel(path, base, edited):
    to = lambda x: tensor_to_pil(x.float() / 127.5 - 1)
    images = [to(x) for x in base] + [to(x) for x in edited]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    image_grid(images, cols=len(base)).save(path)


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    meta = load_json(meta_path(args.data_root, args.dataset))
    labels = np.load(labels_path(args.data_root, args.dataset), mmap_mode="r")
    latents = np.load(latents_path(args.data_root, args.dataset, args.space), mmap_mode="r")
    cls_cfg = load_json(os.path.join(classifier_dir(args.data_root, args.dataset), "classifier_config.json"))
    names, multilabel = cls_cfg["names"], cls_cfg["multilabel"]
    classifier = load_classifier(args.data_root, args.dataset, device)
    space = Space(args.space, device)
    chosen = probe_latents(args)
    bases = base_images(args, meta, labels, names, cls_cfg.get("min_frac", 0.001))
    strengths = [float(s) for s in args.strengths.split(",")]
    all_bases = np.unique(np.concatenate(list(bases.values())))

    for arm in args.arms.split(","):
        unet, saes, sae_cfg = load_unet_and_saes(run_dir(args.data_root, args.dataset, args.space, arm), args.space,
                                                 saes_too=True, device=device)
        sites = args.sites.split(",") if args.sites else sae_cfg["sites"]
        hooks = SAEHooks(unet, saes, sites, mode="inline" if arm == "inline" else "side")
        codes = codes_dir(args.data_root, args.dataset, args.space, arm, args.t)
        base_logits, base_imgs = generate(args, unet, hooks, space, classifier, latents, all_bases, device)
        pos = {int(i): k for k, i in enumerate(all_bases)}
        rows = []
        for site in sites:
            n_latents = saes[site].n_latents
            act, counts = mean_active(codes, site, n_latents)
            dictionary = saes[site].sae.get_dictionary().float()  # (n_latents, C), unit rows
            live = np.nonzero(counts >= args.min_active)[0]
            for ti, task in enumerate(names):
                if (arm, site, task) not in chosen:
                    print(f"no probe latent for {arm} {site} {task} - skipped")
                    continue
                rng = np.random.default_rng(args.seed * 7919 + ti)
                kinds = [("probe", chosen[(arm, site, task)])]
                kinds += [("random", int(j)) for j in rng.choice(live, min(args.n_random, len(live)), replace=False)]
                idx = bases[task]
                rows_of = torch.tensor([pos[int(i)] for i in idx])
                b_logits, b_imgs = base_logits[rows_of], base_imgs[rows_of]
                others = [k for k in range(len(names)) if k != ti]
                for kind, j in kinds:
                    for strength in strengths:
                        vec = strength * float(act[j]) * dictionary[j]
                        e_logits, e_imgs = generate(args, unet, hooks, space, classifier, latents, idx, device,
                                                    inject={site: vec})
                        delta = e_logits - b_logits
                        d_t = delta[:, ti].abs()
                        d_n = delta[:, others].abs().mean(1)
                        rows.append({
                            "dataset": args.dataset, "space": args.space, "arm": arm, "site": site, "task": task,
                            "latent_kind": kind, "latent": j, "strength": strength, "inject_value": float(act[j]),
                            "selection": args.selection, "t_edit": args.t_edit, "n_images": len(idx),
                            "epr_target_change": float(d_t.mean()), "epr_nontarget_change": float(d_n.mean()),
                            "EPR": float(d_t.mean() / (d_n.mean() + 1e-8)),
                            "target_gain": float(delta[:, ti].mean()),
                            "classified_as": classified(e_logits, ti, multilabel),
                            "classified_as_base": classified(b_logits, ti, multilabel),
                            "psnr_vs_unedited": float(psnr(e_imgs, b_imgs).mean())})
                        if kind == "probe" and strength == max(strengths) and args.n_panel > 0:
                            save_panel(os.path.join(args.out_dir, "epr_panels", f"{args.dataset}_{args.space}",
                                                    f"{arm}_{site}_{task.replace(' ', '_')}_s{strength:g}.png"),
                                       b_imgs[:args.n_panel], e_imgs[:args.n_panel])
                r = rows[-len(kinds) * len(strengths)]
                print(f"{arm} {site} {task}: probe latent {r['latent']} EPR {r['EPR']:.2f} "
                      f"gain {r['target_gain']:.2f} classified {r['classified_as']:.2f}", flush=True)
        append_rows(os.path.join(args.out_dir, "epr.csv"), rows)
        hooks.remove()
    print("wrote", os.path.join(args.out_dir, "epr.csv"))


if __name__ == "__main__":
    main(parser.parse_args())

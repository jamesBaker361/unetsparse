# Compares the arms' SAE features (extract_codes.py output) on patch-level tasks.
#
# Patch labels: each 8x8 patch covers an 8x8 block of the 64x64 label map;
# its class fractions decide the labels (--coverage, default 0.5).
#   binary:     one-vs-rest per mask class, positive if the class covers
#               >= coverage of the patch. celeba: the 18 part classes;
#               awa2: "animal" (any species). Per-latent 1D ridge logistic
#               probes are fit on the train patches; the latent with the
#               lowest train BCE and the latent with the highest train F1 are
#               each scored on the test patches (F1/precision/recall at 0.5,
#               AUROC, BCE, loss explained vs the test prevalence).
#   multiclass: patches whose majority class covers >= coverage (others are
#               dropped). celeba: 19 classes; awa2: background + 50 species.
#               Multinomial logistic regression on the whole code of the patch
#               (latents scaled by their train max; dense features standardized).
#               Test accuracy, balanced accuracy, macro F1.
#
# Feature sets (--features):
#   sae         the arm's SAE codes
#   random_sae  a randomly initialized TopK SAE of the same size on the raw
#               activations (b_dec = their train mean) - a no-learning control
#   raw         the site's activation before the SAE (per-dimension probes)
#   input       the clean input latent patches (arm/site independent; run once)
# Rows are appended to {out_dir}/probe_binary.csv and probe_multiclass.csv.

import os
import csv
import fcntl
import argparse

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score, balanced_accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler

from common import (DEFAULT_DATA_ROOT, DATASETS, SPACES, PATCH_GRID, LABEL_SIZE, BlockSAE, labels_path,
                    meta_path, load_json, codes_dir)
from sparse_probe import (fit_sparse_1d_ridge_logistic, sparse_per_latent_confusion, fit_dense_1d_ridge_logistic,
                          precision_recall_f1, baseline_bce, latent_column, scaled_csr)

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, required=True, choices=DATASETS)
parser.add_argument("--space", type=str, required=True, choices=SPACES)
parser.add_argument("--arms", type=str, default="inline,posthoc")
parser.add_argument("--sites", type=str, default=None, help="default: every site in the codes' info.json")
parser.add_argument("--data_root", type=str, default=DEFAULT_DATA_ROOT)
parser.add_argument("--t", type=float, default=0.1)
parser.add_argument("--features", type=str, default="sae,random_sae,raw,input")
parser.add_argument("--tasks", type=str, default="binary,multiclass")
parser.add_argument("--coverage", type=float, default=0.5)
parser.add_argument("--min_pos", type=int, default=50, help="skip binary tasks with fewer train positives")
parser.add_argument("--ridge", type=float, default=1e-8)
parser.add_argument("--mc_C", type=float, default=1.0)
parser.add_argument("--mc_max_iter", type=int, default=500)
parser.add_argument("--mc_max_train", type=int, default=200000, help="subsample train patches for multiclass")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--out_dir", type=str, default="outputs")


# ----------------------------------------------------------------------------- labels

def patch_fractions(labels: np.ndarray, images: np.ndarray, n_classes: int) -> np.ndarray:
    '''(n_images * 64, n_classes): fraction of each patch covered by each class.'''
    f = LABEL_SIZE // PATCH_GRID
    lab = labels[images].reshape(len(images), PATCH_GRID, f, PATCH_GRID, f)
    out = np.zeros((len(images), PATCH_GRID, PATCH_GRID, n_classes), dtype=np.float32)
    for c in np.unique(lab):
        out[..., c] = (lab == c).mean((2, 4))
    return out.reshape(-1, n_classes)


def binary_tasks(dataset: str, classes: list, frac: np.ndarray, coverage: float) -> dict:
    if dataset == "celeba":
        return {name: frac[:, c] >= coverage for c, name in enumerate(classes) if c > 0}
    return {"animal": (1.0 - frac[:, 0]) >= coverage}


def multiclass_labels(frac: np.ndarray, coverage: float):
    '''Majority class per patch and a mask of the patches it covers >= coverage of.'''
    return frac.argmax(1), frac.max(1) >= coverage


# ----------------------------------------------------------------------------- features

class Features:
    '''One feature set at one site: sparse (idx, val, n_dirs) or dense X, for train and test.'''

    def __init__(self, name, train, test, n_dirs=None):
        self.name, self.train, self.test, self.n_dirs = name, train, test, n_dirs
        self.sparse = n_dirs is not None


def random_sae_codes(raw_train, raw_test, n_latents, k, seed):
    torch.manual_seed(seed)
    sae = BlockSAE(raw_train.shape[1], n_latents // raw_train.shape[1], k)
    with torch.no_grad():
        sae.b_dec.copy_(torch.from_numpy(raw_train.astype(np.float32).mean(0)))
    sae.eval()

    @torch.no_grad()
    def encode(raw):
        idx, val = [], []
        for s in range(0, len(raw), 8192):
            _, z, _ = sae(torch.from_numpy(raw[s:s + 8192].astype(np.float32)))
            top = torch.topk(z, k, dim=-1)
            idx.append(top.indices.to(torch.int16).numpy())
            val.append(top.values.numpy())
        return np.concatenate(idx), np.concatenate(val)
    return encode(raw_train), encode(raw_test)


def load_features(directory: str, site: str, info: dict, wanted: list, seed: int) -> list:
    def ld(split, name):
        return np.load(os.path.join(directory, f"{split}_{name}.npy"), mmap_mode="r")
    out = []
    if "sae" in wanted:
        out.append(Features("sae", (ld("train", f"{site}_idx"), ld("train", f"{site}_val")),
                            (ld("test", f"{site}_idx"), ld("test", f"{site}_val")), info["n_latents"]))
    has_raw = os.path.exists(os.path.join(directory, f"train_{site}_raw.npy"))
    if ("raw" in wanted or "random_sae" in wanted) and has_raw:
        raw_train, raw_test = np.asarray(ld("train", f"{site}_raw")), np.asarray(ld("test", f"{site}_raw"))
        if "raw" in wanted:
            out.append(Features("raw", raw_train, raw_test))
        if "random_sae" in wanted:
            tr, te = random_sae_codes(raw_train, raw_test, info["n_latents"], info["k"], seed)
            out.append(Features("random_sae", tr, te, info["n_latents"]))
    return out


# ----------------------------------------------------------------------------- binary

def column(feat: Features, split: str, j: int) -> np.ndarray:
    data = feat.train if split == "train" else feat.test
    return latent_column(*data, j) if feat.sparse else np.asarray(data[:, j], dtype=np.float64)


def test_metrics(z: np.ndarray, y: np.ndarray, w: float, b: float) -> dict:
    from scipy.special import expit
    p = np.clip(expit(w * z + b), 1e-12, 1 - 1e-12)
    pred = p >= 0.5
    tp, fp, fn = (pred & y).sum(), (pred & ~y).sum(), (~pred & y).sum()
    precision, recall, f1 = precision_recall_f1(tp, fp, fn)
    bce = float(-(y * np.log(p) + (~y) * np.log(1 - p)).mean())
    base = baseline_bce(y)
    auroc = roc_auc_score(y, w * z) if 0 < y.sum() < len(y) else float("nan")
    return {"test_f1": float(f1), "test_precision": float(precision), "test_recall": float(recall),
            "test_auroc": float(auroc), "test_bce": bce, "test_loss_explained": 1.0 - bce / base}


def run_binary(feat: Features, tasks_train: dict, tasks_test: dict, args) -> list:
    rows = []
    for task, y_train in tasks_train.items():
        y_test = tasks_test[task]
        if y_train.sum() < args.min_pos or y_test.sum() == 0:
            continue
        if feat.sparse:
            w, b, loss = fit_sparse_1d_ridge_logistic(*feat.train, y_train, feat.n_dirs, args.ridge)
            tp, fp, fn = sparse_per_latent_confusion(*feat.train, y_train, w, b)
        else:
            w, b, loss, tp, fp, fn = fit_dense_1d_ridge_logistic(np.asarray(feat.train), y_train, args.ridge)
        _, _, f1 = precision_recall_f1(tp, fp, fn)
        base = baseline_bce(y_train)
        for rule, j in (("bce", int(np.argmin(loss))), ("f1", int(np.argmax(f1)))):
            row = {"task": task, "selection": rule, "latent": j, "n_train_pos": int(y_train.sum()),
                   "n_test_pos": int(y_test.sum()), "train_bce": float(loss[j]),
                   "train_loss_explained": float(1 - loss[j] / base), "train_f1": float(f1[j]),
                   "w": float(w[j]), "b": float(b[j])}
            row.update(test_metrics(column(feat, "test", j), y_test, w[j], b[j]))
            rows.append(row)
        print(f"  {feat.name} {task}: best-f1 latent test f1 {rows[-1]['test_f1']:.3f}", flush=True)
    return rows


# ----------------------------------------------------------------------------- multiclass

def design(feat: Features):
    if feat.sparse:
        Xtr, scale = scaled_csr(*feat.train, feat.n_dirs)
        Xte, _ = scaled_csr(*feat.test, feat.n_dirs, scale)
        return Xtr, Xte
    scaler = StandardScaler().fit(np.asarray(feat.train, dtype=np.float32))
    return (scaler.transform(np.asarray(feat.train, dtype=np.float32)),
            scaler.transform(np.asarray(feat.test, dtype=np.float32)))


def run_multiclass(feat: Features, y_train, keep_train, y_test, keep_test, args) -> dict:
    Xtr, Xte = design(feat)
    tr = np.nonzero(keep_train)[0]
    if len(tr) > args.mc_max_train:
        tr = np.sort(np.random.default_rng(args.seed).choice(tr, args.mc_max_train, replace=False))
    te = np.nonzero(keep_test)[0]
    clf = LogisticRegression(C=args.mc_C, max_iter=args.mc_max_iter)
    clf.fit(Xtr[tr], y_train[tr])
    pred = clf.predict(Xte[te])
    row = {"n_train": len(tr), "n_test": len(te), "n_classes": len(np.unique(y_train[tr])),
           "train_accuracy": float(clf.score(Xtr[tr], y_train[tr])),
           "test_accuracy": float(accuracy_score(y_test[te], pred)),
           "test_balanced_accuracy": float(balanced_accuracy_score(y_test[te], pred)),
           "test_macro_f1": float(f1_score(y_test[te], pred, average="macro"))}
    print(f"  {feat.name} multiclass: acc {row['test_accuracy']:.3f} macro-f1 {row['test_macro_f1']:.3f}",
          flush=True)
    return row


# ----------------------------------------------------------------------------- output

def append_rows(path: str, rows: list):
    if not rows:
        return
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a+", newline="") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        f.seek(0)
        header = next(csv.reader(f), None)
        writer = csv.DictWriter(f, fieldnames=header or list(rows[0].keys()), extrasaction="ignore")
        if header is None:
            writer.writeheader()
        writer.writerows(rows)
        fcntl.flock(f, fcntl.LOCK_UN)


def main(args):
    meta = load_json(meta_path(args.data_root, args.dataset))
    labels = np.load(labels_path(args.data_root, args.dataset), mmap_mode="r")
    classes = meta["classes"]
    wanted = args.features.split(",")
    tasks = args.tasks.split(",")
    binary_rows, mc_rows = [], []
    did_input = False

    for arm in args.arms.split(","):
        directory = codes_dir(args.data_root, args.dataset, args.space, arm, args.t)
        info = load_json(os.path.join(directory, "info.json"))
        images = {s: np.load(os.path.join(directory, f"{s}_images.npy")) for s in ("train", "test")}
        frac = {s: patch_fractions(labels, images[s], len(classes)) for s in images}
        bin_tr = binary_tasks(args.dataset, classes, frac["train"], args.coverage)
        bin_te = binary_tasks(args.dataset, classes, frac["test"], args.coverage)
        (mc_y_tr, mc_k_tr), (mc_y_te, mc_k_te) = (multiclass_labels(frac[s], args.coverage) for s in ("train", "test"))

        jobs = [(site, f) for site in (args.sites.split(",") if args.sites else info["sites"])
                for f in load_features(directory, site, info, wanted, args.seed)]
        if "input" in wanted and not did_input:
            jobs.append(("input", Features("input", np.load(os.path.join(directory, "train_input.npy")),
                                           np.load(os.path.join(directory, "test_input.npy")))))
            did_input = True
        for site, feat in jobs:
            key = {"dataset": args.dataset, "space": args.space, "arm": "-" if feat.name == "input" else arm,
                   "site": site, "features": feat.name, "t": args.t, "coverage": args.coverage}
            print(key, flush=True)
            if "binary" in tasks:
                binary_rows += [{**key, **r} for r in run_binary(feat, bin_tr, bin_te, args)]
            if "multiclass" in tasks:
                mc_rows.append({**key, **run_multiclass(feat, mc_y_tr, mc_k_tr, mc_y_te, mc_k_te, args)})

    append_rows(os.path.join(args.out_dir, "probe_binary.csv"), binary_rows)
    append_rows(os.path.join(args.out_dir, "probe_multiclass.csv"), mc_rows)
    print("wrote", args.out_dir)


if __name__ == "__main__":
    main(parser.parse_args())

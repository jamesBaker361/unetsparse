# Trains the image classifier EPR (epr.py) reads logits from, one per dataset,
# on the same center-cropped 256px train images the UNets see:
#   celeba: 18-way multi-label part presence (hat, eyeglasses, earring, ...),
#           class c present if it covers >= --min_frac of the 64x64 label map
#   awa2:   50-way species
# ImageNet-pretrained ResNet-50, fine-tuned. Generated images are 256px VAE
# decodes or 64px pixel-space samples upsampled, so training randomly
# round-trips images through 64px (--p_lowres) to cover that domain too.
# Writes {root}/runs/classifier_{dataset}/classifier.pt + classifier_config.json.

import os
import argparse
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from common import (DEFAULT_DATA_ROOT, DATASETS, dataset_dir, meta_path, labels_path, load_json, save_json,
                    load_square, pil_to_tensor)

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, required=True, choices=DATASETS)
parser.add_argument("--data_root", type=str, default=DEFAULT_DATA_ROOT)
parser.add_argument("--epochs", type=int, default=5)
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--lr", type=float, default=1e-4)
parser.add_argument("--min_frac", type=float, default=0.001)
parser.add_argument("--p_lowres", type=float, default=0.3)
parser.add_argument("--num_workers", type=int, default=4)


def classifier_dir(root: str, dataset: str) -> str:
    return os.path.join(root, "runs", f"classifier_{dataset}")


def presence(labels: np.ndarray, n_classes: int, min_frac: float) -> np.ndarray:
    '''(N, n_classes - 1) bool: class 1..n_classes-1 covers >= min_frac of the label map.'''
    n = labels.shape[0]
    flat = labels.reshape(n, -1)
    counts = np.stack([(flat == c).sum(1) for c in range(1, n_classes)], 1)
    return counts >= min_frac * flat.shape[1]


def targets(dataset: str, meta: dict, labels: np.ndarray, min_frac: float) -> np.ndarray:
    if dataset == "celeba":
        return presence(labels, len(meta["classes"]), min_frac).astype(np.float32)
    return np.asarray(meta["image_species"], dtype=np.int64)


def make_classifier(n_out: int, pretrained: bool = True) -> nn.Module:
    from torchvision.models import resnet50, ResNet50_Weights
    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2 if pretrained else None)
    model.fc = nn.Linear(model.fc.in_features, n_out)
    return model


def preprocess(x: torch.Tensor) -> torch.Tensor:
    '''[-1, 1] images (any size) -> ImageNet-normalized 256px.'''
    if x.shape[-1] != 256:
        x = F.interpolate(x, size=(256, 256), mode="bilinear", align_corners=False)
    x = (x + 1) / 2
    return (x - IMAGENET_MEAN.to(x.device)) / IMAGENET_STD.to(x.device)


def load_classifier(root: str, dataset: str, device) -> nn.Module:
    cfg = load_json(os.path.join(classifier_dir(root, dataset), "classifier_config.json"))
    model = make_classifier(cfg["n_out"], pretrained=False)
    model.load_state_dict(torch.load(os.path.join(classifier_dir(root, dataset), "classifier.pt"),
                                     map_location="cpu", weights_only=True))
    return model.to(device).eval().requires_grad_(False)


class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, root, paths, y, train: bool, p_lowres: float):
        self.root, self.paths, self.y, self.train, self.p_lowres = root, paths, y, train, p_lowres

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        x = pil_to_tensor(load_square(os.path.join(self.root, self.paths[i])))
        if self.train:
            if np.random.rand() < 0.5:
                x = x.flip(-1)
            if np.random.rand() < self.p_lowres:
                x = F.interpolate(F.interpolate(x[None], size=(64, 64), mode="bilinear", antialias=True),
                                  size=(256, 256), mode="bilinear")[0]
        return x, self.y[i]


@torch.no_grad()
def evaluate(model, loader, dataset, device) -> dict:
    model.eval()
    logits, ys = [], []
    for x, y in loader:
        logits.append(model(preprocess(x.to(device))).float().cpu())
        ys.append(y)
    model.train()
    logits, ys = torch.cat(logits), torch.cat(ys)
    if dataset == "awa2":
        return {"accuracy": float((logits.argmax(1) == ys).float().mean())}
    from sklearn.metrics import average_precision_score
    ys = ys.numpy()
    aps = [average_precision_score(ys[:, c], logits[:, c].numpy()) for c in range(ys.shape[1]) if 0 < ys[:, c].sum() < len(ys)]
    return {"mAP": float(np.mean(aps)), "accuracy": float(((logits > 0).numpy() == (ys > 0.5)).mean())}


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    meta = load_json(meta_path(args.data_root, args.dataset))
    labels = np.load(labels_path(args.data_root, args.dataset), mmap_mode="r")
    y = targets(args.dataset, meta, labels, args.min_frac)
    n_out = y.shape[1] if args.dataset == "celeba" else len(meta["species"])
    split = np.array(meta["split"])
    root = dataset_dir(args.data_root, args.dataset)

    def loader(name, train):
        idx = np.nonzero(split == name)[0]
        ds = ImageDataset(root, [meta["images"][i] for i in idx], torch.from_numpy(y[idx]), train, args.p_lowres)
        return torch.utils.data.DataLoader(ds, batch_size=args.batch_size, shuffle=train, drop_last=train,
                                           num_workers=args.num_workers)
    train_loader, test_loader = loader("train", True), loader("test", False)

    model = make_classifier(n_out).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    loss_fn = nn.BCEWithLogitsLoss() if args.dataset == "celeba" else nn.CrossEntropyLoss()
    use_amp = device == "cuda"
    scaler = torch.amp.GradScaler(enabled=use_amp)
    for epoch in range(1, args.epochs + 1):
        start = time.time()
        for x, t in train_loader:
            with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                loss = loss_fn(model(preprocess(x.to(device))).float(), t.to(device))
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        print(f"epoch {epoch} loss {loss.item():.4f} test {evaluate(model, test_loader, args.dataset, device)} "
              f"({time.time() - start:.0f}s)", flush=True)

    out = classifier_dir(args.data_root, args.dataset)
    os.makedirs(out, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(out, "classifier.pt"))
    names = meta["classes"][1:]
    save_json(os.path.join(out, "classifier_config.json"),
              {"n_out": n_out, "names": names, "multilabel": args.dataset == "celeba", "min_frac": args.min_frac,
               "test": evaluate(model, test_loader, args.dataset, device)})
    print("wrote", out)


if __name__ == "__main__":
    main(parser.parse_args())

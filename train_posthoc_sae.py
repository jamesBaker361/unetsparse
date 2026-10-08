# Post-hoc arm: trains a TopK SAE at each site of a fully trained plain UNet
# (train_unet.py --sae_mode none) on its frozen activations. The activations
# come from the same training-image / logit-normal-t / random-flip
# distribution the inline arm's SAEs see, and the SAEs use the same
# config and losses (nmse + aux_coef * auxk). b_dec starts at the mean
# activation of the first batch.
# Writes saes.pt + sae_config.json (pointing at the UNet's run dir) to
# {root}/runs/{dataset}_{space}_posthoc.

import os
import time

import numpy as np
import torch

from experiment_helpers.gpu_details import print_details
from experiment_helpers.init_helpers import default_parser, repo_api_init, DEFAULT_SAVE_DIR, DEFAULT_REPO_ID
from experiment_helpers.saving_helpers import save_and_load_functions

from common import (DEFAULT_DATA_ROOT, DATASETS, SPACES, SITES, make_saes, SAEHooks, LatentDataset, sample_t,
                    noisy, predict_v, latents_path, meta_path, load_json, save_json, run_dir, default_repo_id,
                    load_unet_and_saes)

parser = default_parser({"batch_size": 64, "gradient_accumulation_steps": 1, "epochs": 100, "val_interval": 10,
                         "project_name": "unetsparse", "lr": 1e-4})
parser.add_argument("--dataset", type=str, required=True, choices=DATASETS)
parser.add_argument("--space", type=str, required=True, choices=SPACES)
parser.add_argument("--data_root", type=str, default=DEFAULT_DATA_ROOT)
parser.add_argument("--unet_dir", type=str, default=None, help="default {root}/runs/{dataset}_{space}_none")
parser.add_argument("--sites", type=str, default=",".join(SITES))
parser.add_argument("--expansion", type=int, default=16)
parser.add_argument("--k", type=int, default=32)
parser.add_argument("--aux_coef", type=float, default=1 / 32)
parser.add_argument("--dead_steps", type=int, default=1000)
parser.add_argument("--k_aux", type=int, default=256)
parser.add_argument("--lr_warmup_steps", type=int, default=1000)
parser.add_argument("--max_grad_norm", type=float, default=1.0)
parser.add_argument("--num_workers", type=int, default=4)


def main(args):
    if args.save_dir == DEFAULT_SAVE_DIR:
        args.save_dir = run_dir(args.data_root, args.dataset, args.space, "posthoc")
    if args.repo_id == DEFAULT_REPO_ID:
        args.repo_id = default_repo_id(args.dataset, args.space, "posthoc")
    args.unet_dir = args.unet_dir or run_dir(args.data_root, args.dataset, args.space, "none")
    sites = args.sites.split(",")
    api, accelerator, device = repo_api_init(args)

    unet, _, _ = load_unet_and_saes(args.unet_dir, args.space, device=device)
    saes = make_saes(unet, sites, args.expansion, args.k)
    os.makedirs(args.save_dir, exist_ok=True)
    save_json(os.path.join(args.save_dir, "sae_config.json"),
              {"sites": sites, "expansion": args.expansion, "k": args.k, "arm": "posthoc",
               "unet_dir": os.path.abspath(args.unet_dir)})
    save, load = save_and_load_functions({"saes.pt": saes}, args.save_dir, api, args.repo_id)
    start_epoch = load(args.load_hf)
    saes.to(device)

    meta = load_json(meta_path(args.data_root, args.dataset))
    train_idx = np.nonzero(np.array(meta["split"]) == "train")[0]
    loader = torch.utils.data.DataLoader(LatentDataset(latents_path(args.data_root, args.dataset, args.space),
                                                       train_idx),
                                         batch_size=args.batch_size, shuffle=True, drop_last=True,
                                         num_workers=args.num_workers, persistent_workers=args.num_workers > 0)
    optimizer = torch.optim.AdamW(saes.parameters(), lr=args.lr, weight_decay=0.0)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s: min(1.0, (s + 1) / args.lr_warmup_steps))
    optimizer, loader, scheduler = accelerator.prepare(optimizer, loader, scheduler)
    hooks = SAEHooks(unet, None, sites, mode="capture")

    global_step = (start_epoch - 1) * len(loader)
    for epoch in range(start_epoch, args.epochs + 1):
        start = time.time()
        for x0 in loader:
            eps = torch.randn_like(x0)
            t = sample_t(len(x0), x0.device)
            with torch.no_grad(), accelerator.autocast():
                predict_v(unet, noisy(x0, eps, t), t)
            if global_step == 0:
                with torch.no_grad():
                    for s in sites:
                        saes[s].b_dec.copy_(hooks.records[s].float().mean(0))
            loss, logs = 0.0, {}
            for s in sites:
                x = hooks.records[s].float()
                pre, z, xh = saes[s](x)
                l = saes[s].losses(x, pre, z, xh, args.dead_steps, args.k_aux)
                loss = loss + l["nmse"] + args.aux_coef * l["aux"]
                logs.update({f"{s}/nmse": l["nmse"].item(), f"{s}/aux": float(l["aux"]),
                             f"{s}/dead_frac": l["dead_frac"], f"{s}/l0": l["l0"]})
            accelerator.backward(loss)
            accelerator.clip_grad_norm_(saes.parameters(), args.max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            global_step += 1
            logs.update({"loss": loss.item(), "lr": scheduler.get_last_lr()[0], "epoch": epoch})
            accelerator.log(logs, step=global_step)
        print(f"epoch {epoch} step {global_step} loss {loss.item():.4f} ({time.time() - start:.0f}s)", flush=True)
        if accelerator.is_main_process and (epoch % args.val_interval == 0 or epoch == args.epochs):
            save(epoch + 1)
    accelerator.end_training()


if __name__ == "__main__":
    print_details()
    args = parser.parse_args()
    print(args)
    main(args)
    print("all done!")

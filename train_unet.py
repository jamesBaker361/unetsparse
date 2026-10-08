# Trains an unconditional rectified-flow UNet on one dataset's cached latents.
#   --sae_mode none:   plain UNet (later gets post-hoc SAEs, train_posthoc_sae.py)
#   --sae_mode inline: joint arm - a TopK SAE at each site whose reconstruction
#                      replaces the activation. For the first --sae_warmup_steps
#                      the SAEs train on the side (detached activations, UNet
#                      unchanged) so they are reasonable when switched in.
# loss = flow MSE + sae_coef * sum_sites (nmse + aux_coef * auxk)
# Checkpoints (unet.pt, saes.pt, configs) go to --save_dir, default
# {root}/runs/{dataset}_{space}_{sae_mode}, and are uploaded to --repo_id.

import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from experiment_helpers.gpu_details import print_details
from experiment_helpers.init_helpers import default_parser, repo_api_init, DEFAULT_SAVE_DIR, DEFAULT_REPO_ID
from experiment_helpers.saving_helpers import save_and_load_functions

from common import (DEFAULT_DATA_ROOT, DATASETS, SPACES, SITES, Space, make_unet, make_saes, SAEHooks,
                    LatentDataset, sample_t, noisy, predict_v, euler_sample, latents_path, meta_path,
                    load_json, save_json, split_indices, run_dir, default_repo_id, tensor_to_pil, image_grid)

parser = default_parser({"batch_size": 64, "gradient_accumulation_steps": 1, "epochs": 200, "val_interval": 10,
                         "project_name": "unetsparse", "lr": 1e-4})
parser.add_argument("--dataset", type=str, required=True, choices=DATASETS)
parser.add_argument("--space", type=str, required=True, choices=SPACES)
parser.add_argument("--data_root", type=str, default=DEFAULT_DATA_ROOT)
parser.add_argument("--sae_mode", type=str, default="none", choices=["none", "inline"])
parser.add_argument("--sites", type=str, default=",".join(SITES))
parser.add_argument("--expansion", type=int, default=16)
parser.add_argument("--k", type=int, default=32)
parser.add_argument("--sae_coef", type=float, default=1.0)
parser.add_argument("--aux_coef", type=float, default=1 / 32)
parser.add_argument("--dead_steps", type=int, default=1000)
parser.add_argument("--k_aux", type=int, default=256)
parser.add_argument("--sae_warmup_steps", type=int, default=5000)
parser.add_argument("--layers_per_block", type=int, default=2)
parser.add_argument("--attention_head_dim", type=int, default=64)
parser.add_argument("--lr_warmup_steps", type=int, default=1000)
parser.add_argument("--max_grad_norm", type=float, default=1.0)
parser.add_argument("--num_workers", type=int, default=4)
parser.add_argument("--n_samples", type=int, default=16)
parser.add_argument("--sample_steps", type=int, default=50)


def save_samples(unet, space: Space, args, epoch: int, accelerator):
    unet.eval()
    z = euler_sample(unet, args.n_samples, space.channels, space.size, args.sample_steps, accelerator.device)
    images = [tensor_to_pil(x) for x in space.decode(z)]
    unet.train()
    grid = image_grid(images, cols=int(np.ceil(np.sqrt(len(images)))))
    grid.save(os.path.join(args.save_dir, f"samples_epoch{epoch}.png"))
    try:
        import wandb
        accelerator.log({"samples": wandb.Image(grid)})
    except Exception as e:
        print("could not log samples", e)


def main(args):
    if args.save_dir == DEFAULT_SAVE_DIR:
        args.save_dir = run_dir(args.data_root, args.dataset, args.space, args.sae_mode)
    if args.repo_id == DEFAULT_REPO_ID:
        args.repo_id = default_repo_id(args.dataset, args.space, args.sae_mode)
    sites = args.sites.split(",")
    api, accelerator, device = repo_api_init(args)

    meta = load_json(meta_path(args.data_root, args.dataset))
    train_idx = np.nonzero(np.array(meta["split"]) == "train")[0]
    dataset = LatentDataset(latents_path(args.data_root, args.dataset, args.space), train_idx)
    loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True,
                                         num_workers=args.num_workers, persistent_workers=args.num_workers > 0)

    unet_cfg = {"layers_per_block": args.layers_per_block, "attention_head_dim": args.attention_head_dim}
    unet = make_unet(args.space, **unet_cfg)
    model_dict = {"unet.pt": unet}
    saes = None
    if args.sae_mode == "inline":
        saes = make_saes(unet, sites, args.expansion, args.k)
        model_dict["saes.pt"] = saes
    os.makedirs(args.save_dir, exist_ok=True)
    save_json(os.path.join(args.save_dir, "unet_config.json"), unet_cfg)
    save_json(os.path.join(args.save_dir, "sae_config.json"),
              {"sites": sites, "expansion": args.expansion, "k": args.k, "arm": args.sae_mode})

    save, load = save_and_load_functions(model_dict, args.save_dir, api, args.repo_id)
    start_epoch = load(args.load_hf)
    print(f"unet params {sum(p.numel() for p in unet.parameters()) / 1e6:.1f}M, start epoch {start_epoch}")

    params = list(unet.parameters()) + (list(saes.parameters()) if saes is not None else [])
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s: min(1.0, (s + 1) / args.lr_warmup_steps))
    if saes is not None:
        # the hooks call the SAEs directly, so a DDP wrapper around them would never sync their grads
        assert accelerator.num_processes == 1, "the inline arm is single-GPU only"
        saes.to(device)
    unet, optimizer, loader, scheduler = accelerator.prepare(unet, optimizer, loader, scheduler)
    raw_unet = accelerator.unwrap_model(unet)
    raw_saes = saes
    hooks = SAEHooks(raw_unet, raw_saes, sites, mode="side") if saes is not None else None
    space = Space(args.space, device) if accelerator.is_main_process else None

    global_step = (start_epoch - 1) * len(loader) // args.gradient_accumulation_steps
    for epoch in range(start_epoch, args.epochs + 1):
        start = time.time()
        for x0 in loader:
            if hooks is not None:
                hooks.mode = "inline" if global_step >= args.sae_warmup_steps else "side"
            with accelerator.accumulate(unet):
                eps = torch.randn_like(x0)
                t = sample_t(len(x0), x0.device)
                with accelerator.autocast():
                    v = predict_v(unet, noisy(x0, eps, t), t)
                flow = F.mse_loss(v.float(), (eps - x0).float())
                loss = flow
                logs = {"flow_loss": flow.item()}
                if hooks is not None:
                    for s in sites:
                        x, pre, z, xh = hooks.records[s]
                        l = raw_saes[s].losses(x, pre, z, xh, args.dead_steps, args.k_aux)
                        loss = loss + args.sae_coef * (l["nmse"] + args.aux_coef * l["aux"])
                        logs.update({f"{s}/nmse": l["nmse"].item(), f"{s}/aux": float(l["aux"]),
                                     f"{s}/dead_frac": l["dead_frac"], f"{s}/l0": l["l0"]})
                    logs["sae_inline"] = float(hooks.mode == "inline")
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(params, args.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
            if accelerator.sync_gradients:
                global_step += 1
                logs.update({"loss": loss.item(), "lr": scheduler.get_last_lr()[0], "epoch": epoch})
                accelerator.log(logs, step=global_step)
        print(f"epoch {epoch} step {global_step} loss {loss.item():.4f} ({time.time() - start:.0f}s)", flush=True)
        if accelerator.is_main_process and (epoch % args.val_interval == 0 or epoch == args.epochs):
            save_samples(raw_unet, space, args, epoch, accelerator)
            save(epoch + 1)
    accelerator.end_training()


if __name__ == "__main__":
    print_details()
    args = parser.parse_args()
    print(args)
    main(args)
    print("all done!")

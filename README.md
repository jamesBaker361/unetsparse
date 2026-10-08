# unetsparse

Do sparse autoencoders trained **jointly** inside an unconditional rectified-flow UNet learn better features than SAEs trained **post hoc** on a finished UNet? Features are compared on patch-level binary and multiclass probes.

| | |
|---|---|
| Datasets | CelebAMask-HQ (19 part masks), AwA2 (50 species; animal masks from SAM3 prompted with the species name) |
| Input spaces | `sdvae` = `stabilityai/sd-vae-ft-mse` (4x32x32), `flux` = `REPA-E/e2e-flux-vae` (16x32x32), `pixel` = 64x64 RGB |
| Arms | `inline`: TopK SAEs replace the activations during UNet training. `posthoc`: TopK SAEs trained on the frozen plain UNet (`none`) |
| SAE sites | `down` (last down block), `mid`, `up` (first up block, before its upsampler); every UNet is built so all three are 8x8 x 512 |
| SAE | `overcomplete` TopKSAE, expansion 16, k=32, learned decoder bias, AuxK dead-latent loss; identical in both arms |
| Probing | real images noised to t=0.1 with a fixed eps per image; one sample = one 8x8 patch |

## Pipeline

Run from the repo root. Each `scripts/*.sh` submits one stage via `runpygpu_chip.sh`.

| Stage | Script | Output under `$UNETSPARSE_DATA` |
|---|---|---|
| 0 | `python download_data.py` (login node) | `{dataset}/raw/` |
| 1 | `scripts/01_data.sh` → `make_labels.py` | `{dataset}/meta.json`, `{dataset}/labels.npy` (64x64 label maps) |
| 2 | `scripts/02_encode.sh` → `encode_latents.py` | `{dataset}/latents_{space}.npy` (image + flip) |
| 3 | `scripts/03_train_unets.sh` → `train_unet.py --sae_mode none / inline` | `runs/{dataset}_{space}_{none,inline}/` |
| 4 | `scripts/04_train_posthoc.sh` → `train_posthoc_sae.py` | `runs/{dataset}_{space}_posthoc/` |
| 5 | `scripts/05_extract.sh` → `extract_codes.py` | `codes/{dataset}_{space}_{arm}_t0.1/` |
| 6 | `scripts/06_probe.sh` → `probe.py` | `outputs/probe_binary.csv`, `outputs/probe_multiclass.csv` |

The probes compare four feature sets at each site:
- `sae`: the arm's SAE codes.
- `random_sae`: a randomly initialized SAE of the same size, applied to the raw activations.
- `raw`: the activations before the SAE.
- `input`: the input-latent patches. These don't depend on the arm or site.

What the probes compute:
- **Binary:** one-vs-rest per mask class. A patch is positive when the class covers at least 50% of it. A per-latent 1D ridge logistic probe is fit for every latent, and the best latent is chosen by train BCE and by train F1. Each chosen latent is then scored on test patches.
- **Multiclass:** multinomial logistic regression on the patch's whole code. CelebA uses 19 classes; AwA2 uses 51 (background + 50 species).

## Setup

```bash
git submodule update --init
pip install -e sam3_repo   # needs HF access to facebook/sam3
pip install torch torchvision diffusers accelerate huggingface_hub scikit-learn scipy einops wandb requests pillow
export UNETSPARSE_DATA=/umbc/rs/pi_donengel/users/jbaker15/unetsparse_data   # default
```

`common.py` imports only `overcomplete.sae`. `overcomplete`'s top-level `__init__` needs timm and cv2, and it breaks on matplotlib >= 3.9.

Training checkpoints use `experiment_helpers` (wandb + uploads to `jlbaker361/unetsparse_{dataset}_{space}_{arm}` unless `--repo_id` is given). The inline arm is single-GPU only.

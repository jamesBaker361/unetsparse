# Shared pieces for every stage of the pipeline:
#   paths/layout of the data root, the three input spaces (SD VAE, REPA-E
#   flux VAE, 64x64 pixels), the unconditional UNet each space is trained
#   with, the TopK SAE wrapper + forward hooks that put SAEs at the three
#   8x8 sites ("down" = last down block, "mid" = mid block, "up" = first up
#   block before its upsampler), and the rectified-flow utilities.
#
# Data root layout ({root}/{dataset}/...):
#   raw/                       downloaded + unzipped archives (download_data.py)
#   meta.json                  image list, train/test split, class names (make_labels.py)
#   labels.npy                 (N, 64, 64) uint8 label maps (make_labels.py)
#   latents_{space}.npy        (N, 2, C, S, S) float16, [:, 0] original, [:, 1] h-flipped (encode_latents.py)
# Runs go to {root}/runs/{dataset}_{space}_{arm} and probe codes to
# {root}/codes/{dataset}_{space}_{arm}_t{t}.

import os
import sys
import json
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
# Only overcomplete.sae is needed. Its top-level __init__ also imports the
# visualization/model code (timm, cv2, matplotlib's removed cm.get_cmap), so
# register a bare package for it instead of running that __init__.
if "overcomplete" not in sys.modules:
    import types
    _pkg = types.ModuleType("overcomplete")
    _pkg.__path__ = [os.path.join(REPO_ROOT, "overcomplete", "overcomplete")]
    sys.modules["overcomplete"] = _pkg
from overcomplete.sae.topk_sae import TopKSAE  # noqa: E402

DEFAULT_DATA_ROOT = os.environ.get("UNETSPARSE_DATA", "/umbc/rs/pi_donengel/users/jbaker15/unetsparse_data")

DATASETS = ("celeba", "awa2")
SPACES = ("sdvae", "flux", "pixel")
ARMS = ("none", "inline", "posthoc")  # none = plain UNet, inline = joint SAE, posthoc = SAE on the plain UNet
SITES = ("down", "mid", "up")

IMAGE_SIZE = 256  # what the VAEs encode
LABEL_SIZE = 64   # label map resolution
PATCH_GRID = 8    # spatial size of every SAE site

# CelebAMask-HQ classes in the overwrite order of the official g_mask.py
# (later classes paint over earlier ones); index 0 is background
CELEBA_CLASSES = ["background", "skin", "nose", "eye_g", "l_eye", "r_eye", "l_brow", "r_brow", "l_ear",
                  "r_ear", "mouth", "u_lip", "l_lip", "hair", "hat", "ear_r", "neck_l", "neck", "cloth"]

SPACE_CFG = {
    "sdvae": {"repo": "stabilityai/sd-vae-ft-mse", "channels": 4, "size": 32,
              "block_out_channels": (128, 256, 512)},
    "flux": {"repo": "REPA-E/e2e-flux-vae", "channels": 16, "size": 32,
             "block_out_channels": (128, 256, 512)},
    "pixel": {"repo": None, "channels": 3, "size": 64,
              "block_out_channels": (128, 256, 256, 512)},
}


# ----------------------------------------------------------------------------- paths

def dataset_dir(root: str, dataset: str) -> str:
    return os.path.join(root, dataset)


def raw_dir(root: str, dataset: str) -> str:
    return os.path.join(root, dataset, "raw")


def meta_path(root: str, dataset: str) -> str:
    return os.path.join(root, dataset, "meta.json")


def labels_path(root: str, dataset: str) -> str:
    return os.path.join(root, dataset, "labels.npy")


def latents_path(root: str, dataset: str, space: str) -> str:
    return os.path.join(root, dataset, f"latents_{space}.npy")


def run_dir(root: str, dataset: str, space: str, arm: str) -> str:
    return os.path.join(root, "runs", f"{dataset}_{space}_{arm}")


def codes_dir(root: str, dataset: str, space: str, arm: str, t: float) -> str:
    return os.path.join(root, "codes", f"{dataset}_{space}_{arm}_t{t:g}")


def default_repo_id(dataset: str, space: str, arm: str) -> str:
    return f"jlbaker361/unetsparse_{dataset}_{space}_{arm}"


def find_dir(root: str, name: str) -> str:
    '''First directory called `name` under root (archives differ in their top-level folder).'''
    for dirpath, dirnames, _ in os.walk(root):
        if name in dirnames:
            return os.path.join(dirpath, name)
    raise FileNotFoundError(f"no directory named {name} under {root}")


def find_file(root: str, name: str) -> str:
    for dirpath, _, filenames in os.walk(root):
        if name in filenames:
            return os.path.join(dirpath, name)
    raise FileNotFoundError(f"no file named {name} under {root}")


def load_json(path: str):
    with open(path) as f:
        return json.load(f)


def save_json(path: str, obj):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1)
    os.replace(tmp, path)


def split_indices(meta: dict, split: str) -> np.ndarray:
    '''Image indices of a split; AwA2 images where SAM3 found no animal are dropped.'''
    keep = np.array([s == split for s in meta["split"]])
    if "found" in meta:
        keep &= np.array(meta["found"], dtype=bool)
    return np.nonzero(keep)[0]


# ----------------------------------------------------------------------------- images

def center_crop(image: Image.Image) -> Image.Image:
    w, h = image.size
    s = min(w, h)
    left, top = (w - s) // 2, (h - s) // 2
    return image.crop((left, top, left + s, top + s))


def load_square(path: str, size: int = IMAGE_SIZE) -> Image.Image:
    return center_crop(Image.open(path).convert("RGB")).resize((size, size), Image.BICUBIC)


def pil_to_tensor(image: Image.Image) -> torch.Tensor:
    '''(3, H, W) float in [-1, 1].'''
    return torch.from_numpy(np.asarray(image, dtype=np.float32) / 127.5 - 1.0).permute(2, 0, 1)


def tensor_to_pil(x: torch.Tensor) -> Image.Image:
    x = ((x.float().clamp(-1, 1) + 1) * 127.5).round().byte().permute(1, 2, 0).cpu().numpy()
    return Image.fromarray(x)


def image_grid(images: list, cols: int) -> Image.Image:
    w, h = images[0].size
    rows = math.ceil(len(images) / cols)
    grid = Image.new("RGB", (cols * w, rows * h))
    for i, im in enumerate(images):
        grid.paste(im, ((i % cols) * w, (i // cols) * h))
    return grid


# ----------------------------------------------------------------------------- input spaces

class Space:
    '''
    Maps [-1, 1] images (B, 3, 256, 256) to the normalized tensor a UNet is
    trained on and back. VAEs use the posterior mean and their config's
    scaling (and shift, for the flux VAE) factors; "pixel" is the image
    downscaled to 64x64.
    '''

    def __init__(self, name: str, device="cpu"):
        self.name = name
        self.cfg = SPACE_CFG[name]
        self.channels = self.cfg["channels"]
        self.size = self.cfg["size"]
        self.device = device
        self.vae = None
        if self.cfg["repo"] is not None:
            from diffusers import AutoencoderKL
            # fp32: the flux VAE's config asks for force_upcast anyway
            self.vae = AutoencoderKL.from_pretrained(self.cfg["repo"]).to(device).eval().requires_grad_(False)
            self.scale = float(self.vae.config.scaling_factor)
            self.shift = float(getattr(self.vae.config, "shift_factor", None) or 0.0)

    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(self.device, torch.float32)
        if self.vae is None:
            return F.interpolate(x, size=(self.size, self.size), mode="bilinear", antialias=True,
                                 align_corners=False)
        mean = self.vae.encode(x).latent_dist.mean
        return (mean - self.shift) * self.scale

    @torch.no_grad()
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        z = z.to(self.device, torch.float32)
        if self.vae is None:
            return z.clamp(-1, 1)
        return self.vae.decode(z / self.scale + self.shift).sample.clamp(-1, 1)


# ----------------------------------------------------------------------------- UNet

def make_unet(space: str, layers_per_block: int = 2, attention_head_dim: int = 64,
              block_out_channels: tuple = None):
    '''
    Unconditional diffusers UNet2DModel whose lowest resolution is exactly
    PATCH_GRID: n_down = log2(size / 8) downsamples (2 for 32x32 latents,
    3 for 64x64 pixels). Blocks at <= 16x16 have self-attention.
    '''
    from diffusers import UNet2DModel
    cfg = SPACE_CFG[space]
    size, channels = cfg["size"], cfg["channels"]
    block_out_channels = tuple(block_out_channels or cfg["block_out_channels"])
    n_down = int(round(math.log2(size // PATCH_GRID)))
    assert len(block_out_channels) == n_down + 1, f"{space} needs {n_down + 1} blocks for an 8x8 bottleneck"
    resolutions = [size // 2 ** i for i in range(n_down + 1)]
    down = ["AttnDownBlock2D" if r <= 16 else "DownBlock2D" for r in resolutions]
    up = ["AttnUpBlock2D" if r <= 16 else "UpBlock2D" for r in reversed(resolutions)]
    return UNet2DModel(sample_size=size, in_channels=channels, out_channels=channels,
                       layers_per_block=layers_per_block, block_out_channels=block_out_channels,
                       down_block_types=tuple(down), up_block_types=tuple(up),
                       attention_head_dim=attention_head_dim, norm_num_groups=32)


def sae_site_modules(unet) -> dict:
    '''
    The module whose output is each SAE site's (B, C, 8, 8) activation:
    the last resnet/attention of the last down block (no downsampler), the
    mid block, and the last resnet/attention of the first up block (its
    output before the upsampler).
    '''
    def last(block):
        mods = getattr(block, "attentions", None) or block.resnets
        return mods[-1]
    return {"down": last(unet.down_blocks[-1]), "mid": unet.mid_block, "up": last(unet.up_blocks[0])}


def site_dim(unet, site: str) -> int:
    return {"down": unet.config.block_out_channels[-1], "mid": unet.config.block_out_channels[-1],
            "up": unet.config.block_out_channels[-1]}[site]


# ----------------------------------------------------------------------------- SAE

class BlockSAE(nn.Module):
    '''
    overcomplete TopKSAE (linear encoder, l2-normalized dictionary) plus a
    learned decoder bias b_dec that is subtracted before encoding and added
    back after decoding. Also tracks how many steps each latent has gone
    without firing, for the AuxK dead-latent loss (Gao et al. 2024).

    NOTE: overcomplete's DictionaryLayer fuses the dictionary when .eval()
    is called, so call .eval() only after loading weights and moving to the
    final device.
    '''

    def __init__(self, d_in: int, expansion: int = 16, k: int = 32):
        super().__init__()
        self.d_in, self.n_latents, self.k = d_in, d_in * expansion, k
        self.sae = TopKSAE(d_in, nb_concepts=self.n_latents, top_k=k)
        self.b_dec = nn.Parameter(torch.zeros(d_in))
        self.register_buffer("steps_since_fired", torch.zeros(self.n_latents, dtype=torch.long))

    def forward(self, x: torch.Tensor):
        '''x: (N, d_in) -> pre_codes, codes (top-k, relu), reconstruction.'''
        pre, z = self.sae.encode(x - self.b_dec)
        return pre, z, self.sae.decode(z) + self.b_dec

    @torch.no_grad()
    def update_dead(self, z: torch.Tensor):
        fired = (z > 0).any(0)
        self.steps_since_fired += 1
        self.steps_since_fired[fired] = 0

    def losses(self, x, pre, z, xh, dead_steps: int = 1000, k_aux: int = 256) -> dict:
        '''
        nmse: ||x - xh||^2 / ||x - mean(x)||^2. The denominator stays in the
        graph, so in the inline arm the UNet can't lower it by shrinking its
        activations. aux: dead latents (no firing for dead_steps steps)
        predicting the detached residual, normalized the same way.
        '''
        x, xh = x.float(), xh.float()
        var = (x - x.mean(0, keepdim=True)).pow(2).sum(-1).mean().clamp_min(1e-8)
        nmse = (x - xh).pow(2).sum(-1).mean() / var
        self.update_dead(z)
        dead = self.steps_since_fired > dead_steps
        n_dead = int(dead.sum())
        aux = nmse.new_zeros(())
        if n_dead > 0:
            pre_dead = torch.relu(pre.float()).masked_fill(~dead, 0.0)
            top = torch.topk(pre_dead, min(k_aux, n_dead), dim=-1)
            z_aux = torch.zeros_like(pre_dead).scatter(-1, top.indices, top.values)
            resid = (x - xh).detach()
            e_hat = z_aux @ self.sae.get_dictionary().float()
            aux = (resid - e_hat).pow(2).sum(-1).mean() / resid.pow(2).sum(-1).mean().clamp_min(1e-8)
        return {"nmse": nmse, "aux": aux, "dead_frac": n_dead / self.n_latents,
                "l0": float((z > 0).float().sum(-1).mean())}


def make_saes(unet, sites, expansion: int, k: int) -> nn.ModuleDict:
    return nn.ModuleDict({s: BlockSAE(site_dim(unet, s), expansion, k) for s in sites})


class SAEHooks:
    '''
    Forward hooks on the SAE sites. mode:
      "inline":  the SAE reconstruction replaces the activation (joint arm)
      "side":    the SAE runs on the detached activation, the UNet is unchanged
                 (joint-arm warmup, and probing the post-hoc arm)
      "capture": only record the activation (post-hoc SAE training)
    After each forward, self.records[site] is (x, pre, z, xh), or x in capture mode,
    with x of shape (B*8*8, C) in (b, h, w) order.
    '''

    def __init__(self, unet, saes: nn.ModuleDict = None, sites=SITES, mode: str = "inline"):
        self.saes, self.mode, self.records = saes, mode, {}
        modules = sae_site_modules(unet)
        self.handles = [modules[s].register_forward_hook(self._make(s)) for s in sites]

    def _make(self, site):
        def hook(module, inputs, out):
            b, c, h, w = out.shape
            x = out.permute(0, 2, 3, 1).reshape(-1, c)
            if self.mode == "capture":
                self.records[site] = x.detach()
                return None
            xin = x if self.mode == "inline" else x.detach()
            pre, z, xh = self.saes[site](xin)
            self.records[site] = (xin, pre, z, xh)
            if self.mode == "inline":
                return xh.to(out.dtype).reshape(b, h, w, c).permute(0, 3, 1, 2)
            return None
        return hook

    def remove(self):
        for h in self.handles:
            h.remove()


# ----------------------------------------------------------------------------- rectified flow
# x_t = (1 - t) x_0 + t eps, t=0 clean, t=1 noise; the UNet predicts v = eps - x_0.

def sample_t(n: int, device, mean: float = 0.0, std: float = 1.0) -> torch.Tensor:
    '''Logit-normal timesteps (SD3).'''
    return torch.sigmoid(torch.randn(n, device=device) * std + mean)


def noisy(x0: torch.Tensor, eps: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    t = t.view(-1, *([1] * (x0.dim() - 1)))
    return (1 - t) * x0 + t * eps


def predict_v(unet, xt: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    return unet(xt, t * 1000).sample


@torch.no_grad()
def euler_sample(unet, n: int, channels: int, size: int, steps: int, device, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, channels, size, size, generator=g).to(device)
    ts = torch.linspace(1.0, 0.0, steps + 1, device=device)
    for i in range(steps):
        t = ts[i].expand(n)
        x = x + (ts[i + 1] - ts[i]) * predict_v(unet, x, t).float()
    return x


def image_noise(indices, shape, seed: int) -> torch.Tensor:
    '''Deterministic per-image noise, so every arm probes with the same eps.'''
    return torch.stack([torch.randn(*shape, generator=torch.Generator().manual_seed(seed * 1_000_003 + int(i)))
                        for i in indices])


class LatentDataset(torch.utils.data.Dataset):
    '''Cached latents; a random horizontal flip is picked per item.'''

    def __init__(self, path: str, indices: np.ndarray, flip: bool = True):
        self.path, self.indices, self.flip = path, np.asarray(indices), flip
        self.data = None

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        if self.data is None:  # open lazily, once per worker
            self.data = np.load(self.path, mmap_mode="r")
        v = np.random.randint(2) if self.flip else 0
        return torch.from_numpy(np.array(self.data[self.indices[i], v], dtype=np.float32))


def load_unet_and_saes(directory: str, space: str, saes_too: bool = False, device="cpu"):
    '''
    Loads a run dir. A post-hoc run's sae_config.json names the plain UNet's
    run dir ("unet_dir"); the SAE sites/expansion/k come from sae_config.json.
    Returns (unet, saes or None, sae_config).
    '''
    sae_cfg_path = os.path.join(directory, "sae_config.json")
    sae_cfg = load_json(sae_cfg_path) if os.path.exists(sae_cfg_path) else {}
    unet_dir = sae_cfg.get("unet_dir", directory)
    unet = make_unet(space, **load_json(os.path.join(unet_dir, "unet_config.json")))
    unet.load_state_dict(torch.load(os.path.join(unet_dir, "unet.pt"), map_location="cpu", weights_only=True))
    saes = None
    if saes_too:
        saes = make_saes(unet, sae_cfg["sites"], sae_cfg["expansion"], sae_cfg["k"])
        saes.load_state_dict(torch.load(os.path.join(directory, "saes.pt"), map_location="cpu", weights_only=True))
        saes = saes.to(device).eval().requires_grad_(False)  # eval after .to: fuses the dictionary on device
    return unet.to(device).eval().requires_grad_(False), saes, sae_cfg

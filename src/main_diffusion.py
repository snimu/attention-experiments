"""
Experiment with different Attention mechanisms for diffusion models. 
Code taken from the annotated diffusion blog (https://huggingface.co/blog/annotated-diffusion)
and modified for my own purposes.
"""

import argparse
import itertools
import math
import os
from functools import partial
from inspect import isfunction
from typing import Any, Union
from pathlib import Path
from time import perf_counter

import numpy as np
import polars as pl
import torch
import torch.nn.functional as F
from torch.optim import Adam
from datasets import load_dataset
from einops import rearrange, reduce
from einops.layers.torch import Rearrange
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms import (
    CenterCrop,
    Compose,
    Lambda,
    Resize,
    ToPILImage,
    ToTensor,
)
from tqdm.auto import tqdm

import attention
import feature_maps

#############
## HELPERS ##
#############

def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x


def Upsample(dim, dim_out=None):
    return nn.Sequential(
        nn.Upsample(scale_factor=2, mode="nearest"),
        nn.Conv2d(dim, default(dim_out, dim), 3, padding=1),
    )


def Downsample(dim, dim_out=None):
    # No More Strided Convolutions or Pooling
    return nn.Sequential(
        Rearrange("b c (h p1) (w p2) -> b (c p1 p2) h w", p1=2, p2=2),
        nn.Conv2d(dim * 4, default(dim_out, dim), 1),
    )


#####################
## BUILDING BLOCKS ##
#####################

class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings
    

class WeightStandardizedConv2d(nn.Conv2d):
    """
    https://arxiv.org/abs/1903.10520
    weight standardization purportedly works synergistically with group normalization
    """

    def forward(self, x):
        eps = 1e-5 if x.dtype == torch.float32 else 1e-3

        weight = self.weight
        mean = reduce(weight, "o ... -> o 1 1 1", "mean")
        var = reduce(weight, "o ... -> o 1 1 1", partial(torch.var, unbiased=False))
        normalized_weight = (weight - mean) / (var + eps).rsqrt()

        return F.conv2d(
            x,
            normalized_weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )


class Block(nn.Module):
    def __init__(self, dim, dim_out, groups=8):
        super().__init__()
        self.proj = WeightStandardizedConv2d(dim, dim_out, 3, padding=1)
        self.norm = nn.GroupNorm(groups, dim_out)
        self.act = nn.SiLU()

    def forward(self, x, scale_shift=None):
        x = self.proj(x)
        x = self.norm(x)

        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        x = self.act(x)
        return x


class ResnetBlock(nn.Module):
    """https://arxiv.org/abs/1512.03385"""

    def __init__(self, dim, dim_out, *, time_emb_dim=None, groups=8):
        super().__init__()
        self.mlp = (
            nn.Sequential(nn.SiLU(), nn.Linear(time_emb_dim, dim_out * 2))
            if exists(time_emb_dim)
            else None
        )

        self.block1 = Block(dim, dim_out, groups=groups)
        self.block2 = Block(dim_out, dim_out, groups=groups)
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb=None):
        scale_shift = None
        if exists(self.mlp) and exists(time_emb):
            time_emb = self.mlp(time_emb)
            time_emb = rearrange(time_emb, "b c -> b c 1 1")
            scale_shift = time_emb.chunk(2, dim=1)

        h = self.block1(x, scale_shift=scale_shift)
        h = self.block2(h)
        return h + self.res_conv(x)


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.GroupNorm(1, dim)

    def forward(self, x):
        x = self.norm(x)
        return self.fn(x)


##########
## UNET ##
##########
    

ATTENTION_TYPES = Union[
    attention.VanillaConv, 
    attention.LinearConv,
    attention.HydraConv,
    attention.HerculesConv,
    attention.ZeusConv,
]
ATTENTION_CONSTRUCTOR_TYPES = Union[
    type(attention.VanillaConv), 
    type(attention.LinearConv),
    type(attention.HydraConv),
    type(attention.HerculesConv),
    type(attention.ZeusConv),
]
    
class Unet(nn.Module):
    def __init__(
        self,
        dim,
        in_attn_constructor: ATTENTION_CONSTRUCTOR_TYPES,
        mid_attn_constructor: ATTENTION_CONSTRUCTOR_TYPES,
        out_attn_constructor: ATTENTION_CONSTRUCTOR_TYPES,
        in_attn_settings: dict[str, Any],
        mid_attn_settings: dict[str, Any],
        out_attn_settings: dict[str, Any],
        init_dim=None,
        out_dim=None,
        dim_mults=(1, 2, 4, 8),
        channels=3,
        self_condition=False,
        resnet_block_groups=4,
    ):
        super().__init__()

        # determine dimensions
        self.channels = channels
        self.self_condition = self_condition
        input_channels = channels * (2 if self_condition else 1)

        init_dim = default(init_dim, dim)
        self.init_conv = nn.Conv2d(input_channels, init_dim, 1, padding=0) # changed to 1 and 0 from 7,3

        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        block_klass = partial(ResnetBlock, groups=resnet_block_groups)

        # time embeddings
        time_dim = dim * 4

        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(dim),
            nn.Linear(dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim),
        )

        # layers
        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(
                nn.ModuleList(
                    [
                        block_klass(dim_in, dim_in, time_emb_dim=time_dim),
                        block_klass(dim_in, dim_in, time_emb_dim=time_dim),
                        in_attn_constructor(dim_in, norm=nn.GroupNorm(1, dim_in), **in_attn_settings),
                        Downsample(dim_in, dim_out)
                        if not is_last
                        else nn.Conv2d(dim_in, dim_out, 3, padding=1),
                    ]
                )
            )

        mid_dim = dims[-1]
        self.mid_block1 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)
        self.mid_attn = mid_attn_constructor(mid_dim, norm=nn.GroupNorm(1, mid_dim), **mid_attn_settings)
        self.mid_block2 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = ind == (len(in_out) - 1)

            self.ups.append(
                nn.ModuleList(
                    [
                        block_klass(dim_out + dim_in, dim_out, time_emb_dim=time_dim),
                        block_klass(dim_out + dim_in, dim_out, time_emb_dim=time_dim),
                        out_attn_constructor(dim_out, norm=nn.GroupNorm(1, dim_out), **out_attn_settings),
                        Upsample(dim_out, dim_in)
                        if not is_last
                        else nn.Conv2d(dim_out, dim_in, 3, padding=1),
                    ]
                )
            )

        self.out_dim = default(out_dim, channels)

        self.final_res_block = block_klass(dim * 2, dim, time_emb_dim=time_dim)
        self.final_conv = nn.Conv2d(dim, self.out_dim, 1)

    def forward(self, x, time, x_self_cond=None):
        if self.self_condition:
            x_self_cond = default(x_self_cond, lambda: torch.zeros_like(x))
            x = torch.cat((x_self_cond, x), dim=1)

        x = self.init_conv(x)
        r = x.clone()

        t = self.time_mlp(time)

        h = []

        for block1, block2, attn, downsample in self.downs:
            x = block1(x, t)
            h.append(x)

            x = block2(x, t)
            x = attn(x)
            h.append(x)

            x = downsample(x)

        x = self.mid_block1(x, t)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t)

        for block1, block2, attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            x = block1(x, t)

            x = torch.cat((x, h.pop()), dim=1)
            x = block2(x, t)
            x = attn(x)

            x = upsample(x)

        x = torch.cat((x, r), dim=1)

        x = self.final_res_block(x, t)
        return self.final_conv(x)


##################
## DATA HELPERS ##
##################
    
class Transform:
    def __init__(self, image_size=128):
        self.transform = Compose([
            Resize(image_size),
            CenterCrop(image_size),
            ToTensor(), # turn into torch Tensor of shape CHW, divide by 255
            Lambda(lambda t: (t * 2) - 1),
        ])
        self.reverse_transform = Compose([
            Lambda(lambda t: (t + 1) / 2),
            Lambda(lambda t: t.permute(1, 2, 0)), # CHW to HWC
            Lambda(lambda t: t * 255.),
            Lambda(lambda t: t.numpy().astype(np.uint8)),
            ToPILImage(),
        ])

    def __call__(self, img):
        return self.transform(img).unsqueeze(0)
    
    def inverse(self, t):
        return self.reverse_transform(t.squeeze())
    

#######################
## FORWARD DIFFUSION ##
#######################
    
def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule as proposed in https://arxiv.org/abs/2102.09672
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)


def linear_beta_schedule(timesteps):
    beta_start = 0.0001
    beta_end = 0.02
    return torch.linspace(beta_start, beta_end, timesteps)


def quadratic_beta_schedule(timesteps):
    beta_start = 0.0001
    beta_end = 0.02
    return torch.linspace(beta_start**0.5, beta_end**0.5, timesteps) ** 2


def sigmoid_beta_schedule(timesteps):
    beta_start = 0.0001
    beta_end = 0.02
    betas = torch.linspace(-6, 6, timesteps)
    return torch.sigmoid(betas) * (beta_end - beta_start) + beta_start


timesteps = 300

# define beta schedule
betas = linear_beta_schedule(timesteps=timesteps)

# define alphas 
alphas = 1. - betas
alphas_cumprod = torch.cumprod(alphas, axis=0)
alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
sqrt_recip_alphas = torch.sqrt(1.0 / alphas)

# calculations for diffusion q(x_t | x_{t-1}) and others
sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)

# calculations for posterior q(x_{t-1} | x_t, x_0)
posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)


def extract(a, t, x_shape):
    batch_size = t.shape[0]
    out = a.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)
    

# forward diffusion (using the nice property)
def q_sample(x_start, t, noise=None):
    if noise is None:
        noise = torch.randn_like(x_start)

    sqrt_alphas_cumprod_t = extract(sqrt_alphas_cumprod, t, x_start.shape)
    sqrt_one_minus_alphas_cumprod_t = extract(
        sqrt_one_minus_alphas_cumprod, t, x_start.shape
    )

    return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise


############
## LOSSES ##
############

def p_losses(denoise_model, x_start, t, noise=None, loss_type="l1"):
    if noise is None:
        noise = torch.randn_like(x_start)

    x_noisy = q_sample(x_start=x_start, t=t, noise=noise)
    predicted_noise = denoise_model(x_noisy, t)

    if loss_type == 'l1':
        loss = F.l1_loss(noise, predicted_noise)
    elif loss_type == 'l2':
        loss = F.mse_loss(noise, predicted_noise)
    elif loss_type == "huber":
        loss = F.smooth_l1_loss(noise, predicted_noise)
    else:
        raise NotImplementedError()

    return loss

#############
## DATASET ##
#############

# TODO: make this more modular
dataset = load_dataset("fashion_mnist")
image_size = 28
channels = 1
batch_size = 128

# define image transformations (e.g. using torchvision)
transform = Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Lambda(lambda t: (t * 2) - 1)
])

# define function
def transforms(examples):
   examples["pixel_values"] = [transform(image.convert("L")) for image in examples["image"]]
   del examples["image"]

   return examples

transformed_dataset = dataset.with_transform(transforms).remove_columns("label")

# create dataloader
dataloader = DataLoader(transformed_dataset["train"], batch_size=batch_size, shuffle=True)


##############
## SAMPLING ##
##############

@torch.no_grad()
def p_sample(model, x, t, t_index):
    betas_t = extract(betas, t, x.shape)
    sqrt_one_minus_alphas_cumprod_t = extract(
        sqrt_one_minus_alphas_cumprod, t, x.shape
    )
    sqrt_recip_alphas_t = extract(sqrt_recip_alphas, t, x.shape)
    
    # Equation 11 in the paper
    # Use our model (noise predictor) to predict the mean
    model_mean = sqrt_recip_alphas_t * (
        x - betas_t * model(x, t) / sqrt_one_minus_alphas_cumprod_t
    )

    if t_index == 0:
        return model_mean
    else:
        posterior_variance_t = extract(posterior_variance, t, x.shape)
        noise = torch.randn_like(x)
        # Algorithm 2 line 4:
        return model_mean + torch.sqrt(posterior_variance_t) * noise 

# Algorithm 2 (including returning all images)
@torch.no_grad()
def p_sample_loop(model, shape):
    device = next(model.parameters()).device

    b = shape[0]
    # start from pure noise (for each example in the batch)
    img = torch.randn(shape, device=device)
    imgs = []

    for i in tqdm(reversed(range(0, timesteps)), desc='sampling loop time step', total=timesteps):
        img = p_sample(model, img, torch.full((b,), i, device=device, dtype=torch.long), i)
        imgs.append(img.cpu().numpy())
    return imgs

@torch.no_grad()
def sample(model, image_size, batch_size=16, channels=3):
    return p_sample_loop(model, shape=(batch_size, channels, image_size, image_size))


##############
## TRAINING ##
##############

results_folder = Path("./results")
results_folder.mkdir(exist_ok = True)
save_and_sample_every = 1000

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def train(
        model: Unet, 
        epochs: int = 6, 
        device: str | torch.device = "cpu", 
        dtype: torch.dtype = torch.float32,
) -> tuple[list[float], float]:
    optimizer = Adam(model.parameters(), lr=1e-3)
    model = model.to(device=device, dtype=dtype)
    # model = torch.compile(model)
    model.train()

    losses = []
    times_taken = []

    for epoch in range(epochs):
        for step, batch in enumerate(dataloader):
            optimizer.zero_grad()

            batch_size = batch["pixel_values"].shape[0]
            batch = batch["pixel_values"].to(device, dtype)

            # Algorithm 1 line 3: sample t uniformally for every example in the batch
            t = torch.randint(0, timesteps, (batch_size,), device=device, dtype=dtype).long()

            start = perf_counter()
            loss = p_losses(model, batch, t, loss_type="huber")
            time_taken = perf_counter() - start

            if step % 1 == 0:  # approximately 400 steps per epoch
                print(f"loss={loss.item()}, {epoch=}, {step=}, {time_taken=}")
            losses.append(loss.item())
            times_taken.append(time_taken)

            loss.backward()
            optimizer.step()

    return losses, torch.tensor(times_taken).mean().item()


##################
## MY OWN STUFF ##
##################

def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        "-s",
        "--save",
        action="store_true",
        help="Whether to save the results of the tests.",
    )
    parser.add_argument(
        "--append",
        action="store_true",
        help="Whether to append to the results of previous tests, instead of overwriting them (default).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=6,
        help="The number of epochs to train for.",
    )
    parser.add_argument(
        "--num_tries",
        type=int,
        default=10,
        help="The number of tries per test setting.",
    )
    parser.add_argument(
        "--in_attn", 
        type=str, 
        default="all", 
        choices = ["all", "linear", "vanilla", "hydra", "hercules", "zeus"],
        help="The attention mechanism to use for the input attention."
    )
    parser.add_argument(
        "--mid_attn", 
        type=str, 
        default="all", 
        choices = ["all", "linear", "vanilla", "hydra", "hercules", "zeus"],
        help="The attention mechanism to use for the middle attention."
    )
    parser.add_argument(
        "--out_attn", 
        type=str, 
        default="all", 
        choices = ["all", "linear", "vanilla", "hydra", "hercules", "zeus"],
        help="The attention mechanism to use for the output attention."
    )

    args = parser.parse_args()
    return args


get_attn_constructor = {
    "linear": attention.LinearConv,
    "vanilla": attention.VanillaConv,
    "hydra": attention.HydraConv,
    "hercules": attention.HerculesConv,
    "zeus": attention.ZeusConv,
}
get_attn_name = {v: k for k, v in get_attn_constructor.items()}

get_attn_settings = {
    "linear": {},
    "vanilla": {},
    "hydra": {
        "feature_map_qkv": feature_maps.identity,
        "feature_map_attn": feature_maps.cos_sim, 
        "device": DEVICE,
    },
    "hercules": {
        "feature_map_qkv": feature_maps.tanh,
        "feature_map_attn": feature_maps.cos_sim,
        "device": DEVICE,
    },
    "zeus": {
        "feature_map_qkv": feature_maps.tanh,
        "feature_map_attn": feature_maps.sigmoid,
        "device": DEVICE,
    },
}


def get_attn_constructor_and_settings(attn: str) -> tuple[ATTENTION_CONSTRUCTOR_TYPES, dict[str, Any]]:
    if attn == "all": 
        return [
            get_attn_constructor[a] for a in ["linear", "vanilla", "hydra", "hercules", "zeus"]
        ], [
            get_attn_settings[a] for a in ["linear", "vanilla", "hydra", "hercules", "zeus"]
        ]
    else:
        return [get_attn_constructor[attn]], [get_attn_settings[attn]]


def tests(args: argparse.Namespace) -> None:
    """Run some tests."""

    in_attn_constructors, in_attn_settings = get_attn_constructor_and_settings(args.in_attn)
    mid_attn_constructors, mid_attn_settings = get_attn_constructor_and_settings(args.mid_attn)
    out_attn_constructors, out_attn_settings = get_attn_constructor_and_settings(args.out_attn)

    for experiment_num, ((in_ac, in_set), (mid_ac, mid_set), (out_ac, out_set)) in enumerate(
        itertools.product(
            zip(in_attn_constructors, in_attn_settings), 
            zip(mid_attn_constructors, mid_attn_settings),
            zip(out_attn_constructors, out_attn_settings),
        )
    ):
        for trial_num in range(args.num_tries):
            in_attn_name = get_attn_name.get(in_ac, "all")
            mid_attn_name = get_attn_name.get(mid_ac, "all")
            out_attn_name = get_attn_name.get(out_ac, "all")

            print(f"\n\nTraining with {in_attn_name}, {mid_attn_name}, {out_attn_name}...")
            print(f"  Trial {trial_num} of {args.num_tries}...\n")

            model = Unet(
                dim=image_size,
                channels=channels,
                dim_mults=(1, 2, 4,),
                in_attn_constructor=in_ac,
                mid_attn_constructor=mid_ac,
                out_attn_constructor=out_ac,
                in_attn_settings=in_set,
                mid_attn_settings=mid_set,
                out_attn_settings=out_set,
            )

            losses, avg_time_taken = train(
                model=model, 
                epochs=args.epochs, 
                device=DEVICE, 
                dtype=torch.float32,
            )

            results = {
                "in_attn": in_attn_name,
                "mid_attn": mid_attn_name,
                "out_attn": out_attn_name,
                "epochs": args.epochs,
                "last_loss": losses[-1],
                "best_loss": min(losses),
                "losses": str(losses),
                "avg_time_taken": avg_time_taken,
            }
            df = pl.DataFrame(results)
            
            if args.save:
                if not os.path.exists('results_diffusion.csv') or (not args.append and experiment_num == 0):
                    df.write_csv('results_diffusion.csv')
                else:
                    with open('results_diffusion.csv', 'ab') as f:
                        df.write_csv(f, include_header=False)
            
            print(f"DONE ({in_attn_name}, {mid_attn_name}, {out_attn_name}, {trial_num}/{args.num_tries})\n\n)")

    # Print final results
    df = pl.read_csv('results_diffusion.csv')
    df = df.sort(by="best_loss")
    print("\n\nSorted Results:\n\n")
    print(str(df.columns[:6]))
    for row in df.iter_rows():
        print(str(row[:6]))


if __name__ == "__main__":
    args = get_args()
    tests(args)

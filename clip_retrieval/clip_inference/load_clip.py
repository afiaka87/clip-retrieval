"""Module for loading CLIP and (optionally) prior diffusion models from dalle2_pytorch"""

from functools import lru_cache

import clip
import torch
import torch.nn.functional as F
from dalle2_pytorch import OpenAIClipAdapter
from torchvision.transforms import (
    CenterCrop,
    Compose,
    InterpolationMode,
    Normalize,
    Resize,
)
from torchvision.transforms import functional as torchvision_functional


def clip_transform(clip_size) -> torch.Tensor:
    return Compose(
        [
            Resize(clip_size, interpolation=InterpolationMode.BICUBIC),
            CenterCrop(clip_size),
            Normalize(
                (0.48145466, 0.4578275, 0.40821073),
                (0.26862954, 0.26130258, 0.27577711),
            ),
        ]
    )


def vae_preprocess(image) -> torch.Tensor:
    image_tensor = torchvision_functional.resize(
        image, 512, interpolation=InterpolationMode.LANCZOS
    )
    image_tensor = torchvision_functional.center_crop(image_tensor, (512, 512))
    image_tensor = torchvision_functional.to_tensor(image_tensor)
    return image_tensor


def load_open_clip(clip_model, use_jit=True, device="cuda"):
    import open_clip  # pylint: disable=import-outside-toplevel

    pretrained = dict(open_clip.list_pretrained())
    checkpoint = pretrained[clip_model]
    model, _, preprocess = open_clip.create_model_and_transforms(
        clip_model, pretrained=checkpoint, device=device, jit=use_jit
    )
    return model, preprocess


@lru_cache(maxsize=None)
def load_clip(clip_model="ViT-B/32", use_jit=True):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if clip_model.startswith("open_clip:"):
        clip_model = clip_model[len("open_clip:") :]
        return load_open_clip(clip_model, use_jit, device)
    else:
        model, preprocess = clip.load(clip_model, device=device, jit=use_jit)
    return model, preprocess


def l2norm(t):
    return F.normalize(t, dim=-1)


def load_prior(model_path, device="cuda") -> torch.nn.Module:
    from dalle2_pytorch.dalle2_pytorch import DiffusionPrior, DiffusionPriorNetwork

    prior_network = DiffusionPriorNetwork(
        dim=768,
        depth=12,
        num_timesteps=1000,
        max_text_len=77,
        num_time_embeds=1,
        num_image_embeds=1,
        num_text_embeds=1,
        dim_head=64,
        heads=12,
        ff_mult=4,
        norm_out=True,
        attn_dropout=0.05,
        ff_dropout=0.05,
        final_proj=True,
        normformer=True,
        rotary_emb=True,
    )

    diffusion_prior = DiffusionPrior(
        clip=OpenAIClipAdapter(name="ViT-L/14"),
        net=prior_network,
        image_embed_dim=768,
        image_size=224,
        image_channels=3,
        timesteps=1000,
        sample_timesteps=64,
        cond_drop_prob=0.1,
        loss_type="l2",
        predict_x_start=True,
        beta_schedule="cosine",
        condition_on_text_encodings=True,
    ).to(device)
    state_dict = torch.load(model_path, map_location="cpu")
    diffusion_prior.load_state_dict(state_dict, strict=False)
    del state_dict
    return diffusion_prior

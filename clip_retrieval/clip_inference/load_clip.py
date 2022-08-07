"""Module for loading CLIP and (optionally) prior diffusion models from dalle2_pytorch"""

from functools import lru_cache

import torch
import torch.nn.functional as F
import clip

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


def load_prior(model_path, device="cpu"):
    from dalle2_pytorch import DiffusionPrior, DiffusionPriorNetwork, OpenAIClipAdapter

    prior_network = DiffusionPriorNetwork(
        dim=768,
        depth=24,
        dim_head=64,
        heads=32,
        normformer=True,
        attn_dropout=5e-2,
        ff_dropout=5e-2,
        num_time_embeds=1,
        num_image_embeds=1,
        num_text_embeds=1,
        num_timesteps=1000,
        ff_mult=4,
    )
    diffusion_prior = DiffusionPrior(
        net=prior_network,
        clip=OpenAIClipAdapter("ViT-L/14"),
        image_embed_dim=768,
        timesteps=1000,
        cond_drop_prob=0.0,  # TODO
        loss_type="l2",
        condition_on_text_encodings=False,  # TODO
    )
    diffusion_prior.to(device)
    model_state_dict = torch.load(model_path)
    if "ema_model" in model_state_dict:
        print("Loading EMA Model")
        diffusion_prior.load_state_dict(model_state_dict["ema_model"], strict=True)
        torch.save(diffusion_prior.state_dict(), "ema_prior_aes_finetune.pth")
    elif "ema" in model_path:
        diffusion_prior.load_state_dict(model_state_dict, strict=False)
    else:
        print("Loading Standard Model")
        diffusion_prior.load_state_dict(model_state_dict["model"], strict=False)
    del model_state_dict
    return diffusion_prior

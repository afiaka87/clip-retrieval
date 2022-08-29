"""mapper module transform images and text to embeddings"""
import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np
from torchvision.transforms import functional as torchvision_functional
from PIL import Image as PILImage
import pytest
import torch
from diffusers import AutoencoderKL


VAE_MODEL_WEIGHTS = (
    "/home/claym/Projects/cog-stable-diffusion/diffusers-cache/vae"  # TODO
)

vae = AutoencoderKL.from_pretrained(
    VAE_MODEL_WEIGHTS,
    # revision="fp16",
    # torch_dtype=torch.float16,
    local_files_only=True,  # TODO support diffusers api auth
)
vae.eval()
vae.to("cpu")
print(f"VAE loaded")


@torch.no_grad()
def test_batch_encode():
    image_tensor = torch.randn(4, 3, 512, 512)
    vae_latents = vae.encode(image_tensor).sample()
    vae_latents_npy = vae_latents.cpu().numpy()
    np.save("vae_latents.npy", vae_latents_npy)
    assert vae_latents.shape == (4, 4, 64, 64)


@torch.no_grad()
def test_batch_decode():
    vae_latents = torch.randn(1, 4, 64, 64)
    image_tensor = vae.decode(vae_latents)
    assert image_tensor.shape == (4, 3, 512, 512)


@torch.no_grad()
def test_pil_to_vae():
    pil_image = (
        PILImage.open(
            "/home/claym/Projects/clip-retrieval/tests/test_clip_inference/test_images/416_264.jpg"
        )
        .convert("RGB")
        .resize((512, 512))
    )
    image_tensor = torchvision_functional.to_tensor(pil_image)
    image_tensor = image_tensor.unsqueeze(0)  # add batch dimension
    vae_latents = vae.encode(image_tensor).sample()
    assert vae_latents.shape == (1, 4, 64, 64)


@torch.no_grad()
def test_decode_from_file():
    vae_latents_npy = np.load("vae_latents.npy")
    vae_latents = torch.from_numpy(vae_latents_npy)
    assert vae_latents.shape == (4, 4, 64, 64)  # batch size, channels, height, width
    image_tensor = vae.decode(vae_latents)

    assert image_tensor.shape == (4, 3, 512, 512)  # batch size, channels, height, width
    # save the batch as pil images
    for i in range(4):
        image_tensor_i = image_tensor[i]
        pil_image = torchvision_functional.to_pil_image(image_tensor_i)
        pil_image.save(f"vae_decoded_{i:03d}.jpg")

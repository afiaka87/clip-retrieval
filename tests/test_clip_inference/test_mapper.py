import pytest
import pickle
import os

from clip_retrieval.clip_inference.mapper import ClipMapper


@pytest.mark.parametrize("model", ["ViT-B/32", "open_clip:ViT-B-32-quickgelu"])
def test_mapper(model):
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    mapper = ClipMapper(
        enable_image=True,
        enable_text=False,
        enable_metadata=False,
        enable_unclip=False,
        enable_vae=False,
        use_mclip=False,
        clip_model=model,
        use_jit=True,
        mclip_model="",
        unclip_model=None,
        vae_model=None,
    )
    current_dir = os.path.dirname(os.path.abspath(__file__))
    tensor_files = [i for i in os.listdir(current_dir + "/test_tensors")]
    for tensor_file in tensor_files:
        with open(current_dir + "/test_tensors/{}".format(tensor_file), "rb") as f:
            tensor = pickle.load(f)
            sample = mapper(tensor)
            assert sample["image_embs"].shape[0] == tensor["image_tensor"].shape[0]
        pass

def test_enable_vae(model):
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    mapper = ClipMapper(
        enable_image=True,
        enable_text=True,
        enable_metadata=True,
        enable_unclip=False,
        enable_vae=True,
        use_mclip=False,
        clip_model=model,
        use_jit=True,
        mclip_model="",
        unclip_model=None,
        vae_model="/home/claym/Projects/cog-stable-diffusion/diffusers-cache/vae",
    )
    current_dir = os.path.dirname(os.path.abspath(__file__))
    tensor_files = [i for i in os.listdir(current_dir + "/test_tensors")]
    for tensor_file in tensor_files:
        with open(current_dir + "/test_tensors/{}".format(tensor_file), "rb") as f:
            tensor = pickle.load(f)
            sample = mapper(tensor)
            assert sample["vae_embs"].shape[0] == tensor["image_tensor"].shape[0]
        pass

def test_enable_unclip(model):
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    mapper = ClipMapper(
        enable_image=False,
        enable_text=True,
        enable_metadata=False,
        enable_unclip=True,
        enable_vae=False,
        use_mclip=False,
        clip_model=model,
        use_jit=True,
        mclip_model="",
        unclip_model=None,
        vae_model="/home/claym/Projects/cog-stable-diffusion/diffusers-cache/vae",
    )
    current_dir = os.path.dirname(os.path.abspath(__file__))
    tensor_files = [i for i in os.listdir(current_dir + "/test_tensors")]
    for tensor_file in tensor_files:
        with open(current_dir + "/test_tensors/{}".format(tensor_file), "rb") as f:
            tensor = pickle.load(f)
            sample = mapper(tensor)
            assert sample["vae_embs"].shape[0] == tensor["image_tensor"].shape[0]
        pass


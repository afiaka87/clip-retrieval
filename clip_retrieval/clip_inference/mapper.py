"""mapper module transform images and text to embeddings"""

import torch
from sentence_transformers import SentenceTransformer

from .load_clip import load_clip, load_prior


def normalized(a, axis=-1, order=2):
    import numpy as np  # pylint: disable=import-outside-toplevel

    l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    l2[l2 == 0] = 1
    return a / np.expand_dims(l2, axis)


class ClipMapper:
    """transforms images and texts into clip embeddings"""

    def __init__(
        self,
        enable_image,
        enable_text,
        enable_metadata,
        enable_inverted,
        use_mclip,
        clip_model,
        use_jit,
        mclip_model,
    ):
        self.enable_image = enable_image
        self.enable_text = enable_text
        self.enable_metadata = enable_metadata
        self.enable_inverted = enable_inverted
        self.use_mclip = use_mclip
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        model, _ = load_clip(clip_model, use_jit)
        self.model_img = model.encode_image
        self.model_txt = model.encode_text

        prior_diffusion = None
        if self.enable_inverted:
            assert (
                clip_model == "ViT-L/14"
            ), "ViT-L/14 is the only supported model for inverted text"
            assert not use_mclip, "You cannot use mclip with inverted text"
            prior_diffusion = load_prior(
                model_path="prior_aes_finetune.pth", device=self.device
            )
            self.model_txt_inverted = prior_diffusion.sample
        else:
            self.model_txt_inverted = None
        if use_mclip:
            print("\nLoading MCLIP model for text embedding\n")
            mclip = SentenceTransformer(mclip_model)
            self.model_txt = mclip.encode

    def check_item(self, item):
        if self.enable_image:
            assert item["image_tensor"] is not None, "image_tensor is required"
            assert item["image_filename"] is not None, "image_filename is required"
        if self.enable_text or self.enable_inverted:
            assert item["text"] is not None, "text is required"
            assert item["text_tokens"] is not None, "text_tokens is required"
        if self.enable_metadata:
            assert item["metadata"] is not None, "metadata is required"

    @torch.no_grad()
    def __call__(self, item):
        self.check_item(
            item
        )  # Run assertions at the beginning to make sure that all required fields are present
        output = {}
        if self.enable_image:
            image_features = self.model_img(item["image_tensor"].to(self.device))
            image_features /= image_features.norm(dim=-1, keepdim=True)
            output["image_embs"] = image_features.cpu().numpy()
            output["image_filename"] = item["image_filename"]
        if self.enable_text:
            if self.use_mclip:
                output["text_embs"] = normalized(self.model_txt(item["text"]))
            else:
                text_features = self.model_txt(item["text_tokens"].to(self.device))
                text_features /= text_features.norm(dim=-1, keepdim=True)
                output["text_embs"] = text_features.cpu().numpy()
        if self.enable_inverted:
            inverted_features = self.model_txt_inverted(item["text_tokens"].to(self.device))
            inverted_features /= inverted_features.norm(dim=-1, keepdim=True)
            output["inverted_embs"] = inverted_features.cpu().numpy()
        if self.enable_inverted or self.enable_text:  # If either of these are enabled, we should 
            output["text"] = item["text"]
        if self.enable_metadata:
            output["metadata"] = item["metadata"]
        
        return output
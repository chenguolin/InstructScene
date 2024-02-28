from typing import *
from PIL.Image import Image as PILImage

import torch
from torch import nn

from transformers import CLIPTokenizer, CLIPTextModelWithProjection
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection


class CLIPTextEncoder(nn.Module):
    def __init__(self,
        name="openai/clip-vit-base-patch32",
        max_length=77, device="cpu"
    ):
        super().__init__()

        self.tokenizer = CLIPTokenizer.from_pretrained(name)
        self.text_encoder = CLIPTextModelWithProjection.from_pretrained(name).to(device).eval()

        self.text_emb_dim = self.text_encoder.config.hidden_size
        self.max_length = max_length
        self.device = device

        assert self.max_length == self.tokenizer.model_max_length

    @torch.no_grad()
    def forward(self, prompt: Union[str, List[str]], norm=True):
        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids

        text_encoder_output = self.text_encoder(
            text_input_ids.to(self.device)
        )

        text_last_hidden_state = text_encoder_output.last_hidden_state.float()  # (num_prompts, max_length, text_emb_dim)
        text_embeds = text_encoder_output.text_embeds.float()  # (num_prompts, text_emb_dim)
        if norm:
            text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)  # L2 normalize

        return text_last_hidden_state, text_embeds


class CLIPImageEncoder(nn.Module):
    def __init__(self,
        name="openai/clip-vit-base-patch32",
        device="cpu"
    ):
        super().__init__()

        self.image_processor = CLIPImageProcessor()
        self.image_encoder = CLIPVisionModelWithProjection.from_pretrained(name).to(device).eval()

        self.image_emb_dim = self.image_encoder.config.hidden_size
        self.device = device

    @torch.no_grad()
    def forward(self, image: Union[PILImage, List[PILImage]], norm=True):
        image = self.image_processor(images=image, return_tensors="pt").pixel_values.to(self.device)
        image_embeds = self.image_encoder(image).image_embeds.float()  # (num_images, image_emb_dim)
        if norm:
            image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)

        return image_embeds

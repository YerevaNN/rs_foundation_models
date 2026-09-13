from __future__ import annotations

from typing import Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .vision_transformer import MultiLevelNeck


PANOPTICON_BAND_IDS: dict[str, int] = {
    "B1": 443,
    "B01": 443,
    "B2": 493,
    "B02": 493,
    "B3": 559,
    "B03": 559,
    "B4": 664,
    "B04": 664,
    "B5": 704,
    "B05": 704,
    "B6": 740,
    "B06": 740,
    "B7": 783,
    "B07": 783,
    "B8": 832,
    "B08": 832,
    "B8A": 864,
    "B9": 945,
    "B09": 945,
    "B10": 1373,
    "B11": 1610,
    "B12": 2186,
    "VV": -1,
    "VH": -2,
    "HH": -3,
    "HV": -4,
}


def get_panopticon_channel_ids(
    bands: Sequence[str],
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    ids: list[int] = []
    for band in bands:
        normalized_band = band.upper()
        if normalized_band not in PANOPTICON_BAND_IDS:
            raise ValueError(f"Unsupported Panopticon band `{band}`")
        ids.append(PANOPTICON_BAND_IDS[normalized_band])
    if len(ids) == 0:
        raise ValueError("Panopticon requires at least one band")
    return torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0).repeat(batch_size, 1)


def make_panopticon_input(x: torch.Tensor, bands: Sequence[str]) -> dict[str, torch.Tensor]:
    chn_ids = get_panopticon_channel_ids(
        bands=bands,
        batch_size=x.shape[0],
        device=x.device,
    )
    if chn_ids.shape[1] < x.shape[1]:
        pad_count = x.shape[1] - chn_ids.shape[1]
        pad_ids = chn_ids[:, -1:].repeat(1, pad_count)
        chn_ids = torch.cat([chn_ids, pad_ids], dim=1)
    elif chn_ids.shape[1] > x.shape[1]:
        raise ValueError(f"Panopticon received {x.shape[1]} image channels but {chn_ids.shape[1]} channel ids")
    return {"imgs": x, "chn_ids": chn_ids}


class PanopticonEncoder(nn.Module):
    def __init__(
        self,
        out_indices: Sequence[int] = (3, 5, 7, 11),
        out_channels: Sequence[int] = (768, 768, 768, 768),
        for_cls: bool = False,
        patch_size: int = 14,
        **kwargs: object,
    ) -> None:
        super().__init__()
        self.model = torch.hub.load("Panopticon-FM/panopticon", "panopticon_vitb14")
        self.out_indices = tuple(out_indices)
        self.output_channels = tuple(out_channels)
        self.out_channels = tuple(out_channels)
        self.for_cls = for_cls
        self.patch_size = patch_size
        self.neck = MultiLevelNeck(
            in_channels=list(out_channels),
            out_channels=out_channels[0],
            scales=[4, 2, 1, 0.5],
        )

    def forward(
        self,
        x: torch.Tensor | dict[str, torch.Tensor],
        bands: Optional[Sequence[str]] = None,
    ) -> torch.Tensor | tuple[torch.Tensor, ...]:
        x_dict = self._prepare_input(x=x, bands=bands)
        if self.for_cls:
            return self.model(x_dict)
        imgs = x_dict["imgs"]
        padded_imgs, padded_height, padded_width = self._pad_to_patch_size(imgs=imgs)
        x_dict = dict(x_dict)
        x_dict["imgs"] = padded_imgs
        blocks = self.model.get_intermediate_layers(
            x_dict,
            n=list(self.out_indices),
            return_class_token=True,
        )
        features: list[torch.Tensor] = []
        height_tokens = padded_height // self.patch_size
        width_tokens = padded_width // self.patch_size
        for block in blocks:
            patch_tokens = block[0]
            batch_size, _, channels = patch_tokens.shape
            feature = patch_tokens.reshape(batch_size, height_tokens, width_tokens, channels)
            feature = feature.permute(0, 3, 1, 2).contiguous()
            features.append(feature)
        return self.neck(tuple(features))

    def _prepare_input(
        self,
        x: torch.Tensor | dict[str, torch.Tensor],
        bands: Optional[Sequence[str]],
    ) -> dict[str, torch.Tensor]:
        if isinstance(x, dict):
            return x
        if bands is None:
            raise ValueError("Panopticon requires band names when input is not a dict")
        return make_panopticon_input(x=x, bands=bands)

    def _pad_to_patch_size(self, imgs: torch.Tensor) -> tuple[torch.Tensor, int, int]:
        height = imgs.shape[-2]
        width = imgs.shape[-1]
        padded_height = ((height + self.patch_size - 1) // self.patch_size) * self.patch_size
        padded_width = ((width + self.patch_size - 1) // self.patch_size) * self.patch_size
        if padded_height == height and padded_width == width:
            return imgs, height, width
        return F.pad(imgs, (0, padded_width - width, 0, padded_height - height)), padded_height, padded_width


panopticon_encoders = {
    "panopticon": {
        "encoder": PanopticonEncoder,
        "pretrained_settings": None,
        "params": {
            "out_indices": (3, 5, 7, 11),
            "out_channels": (768, 768, 768, 768),
            "patch_size": 14,
        },
    },
}

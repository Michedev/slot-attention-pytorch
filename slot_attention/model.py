from dataclasses import dataclass, field
from typing import List, Union, Literal, Dict, Iterable, Sequence, Optional, overload

import hydra.utils
from omegaconf import OmegaConf, DictConfig
from torch import nn
import torch

from slot_attention.paths import CONFIG
from slot_attention.slot_attention_module import SlotAttentionModule
from slot_attention.encoder_decoder import Encoder, Decoder


@dataclass(eq=False)
class SlotAttentionAE(nn.Module):
    width: int
    height: int
    encoder: Union[Encoder, torch.nn.Module]
    decoder: Union[Decoder, torch.nn.Module]
    slot_attention_module: SlotAttentionModule
    input_channels: int = 3
    w_broadcast: Optional[int] = None
    h_broadcast: Optional[int] = None
    name: str = 'slot-attention'
    lossf = nn.MSELoss()

    def __post_init__(self):
        super().__init__()
        if self.w_broadcast is None:
            self.w_broadcast = self.width
        if self.h_broadcast is None:
            self.h_broadcast = self.height
        self.slot_attention = SlotAttentionModule(self.num_slots, self.encoder_params['channels'][-1], self.latent_size,
                                                  self.attention_iters, self.eps, self.mlp_size)
    def spatial_broadcast(self, slot):
        slot = slot.unsqueeze(-1).unsqueeze(-1)
        return slot.repeat(1, 1, self.w_broadcast, self.h_broadcast)

    def forward(self, x) -> dict:
        with torch.no_grad():
            x = x * 2.0 - 1.0
        encoded = self.encoder(x)
        encoded = encoded.permute(0, 2, 1)
        z = self.slot_attention(encoded)
        bs = z.size(0)
        slots = z.flatten(0, 1)
        slots = self.spatial_broadcast(slots)
        img_slots, masks = self.decoder(slots)
        img_slots = img_slots.view(bs, self.num_slots, 3, self.width, self.height)
        masks = masks.view(bs, self.num_slots, 1, self.width, self.height)
        masks = masks.softmax(dim=1)
        recon_slots_mask = img_slots * masks
        recon_img = recon_slots_mask.sum(dim=1)
        loss = self.lossf(x, recon_img)
        with torch.no_grad():
            recon_slots_output = (img_slots + 1.0) / 2.
        return dict(loss=loss, z=z, mask=masks, slot=recon_slots_output)

    @overload
    @classmethod
    def from_config(cls, dataset_width: int = 64, dataset_height: int = 64, max_num_objects: int = 5) -> 'SlotAttentionAE':
        model_config_path = CONFIG / 'slot_attention.yaml'
        model_config = OmegaConf.load(model_config_path)
        model_config = model_config.merge_with_dotlist([f"dataset.width={dataset_width}",
                                                        f"dataset.height={dataset_height}",
                                                        f"dataset.max_num_objects={max_num_objects}"])
        return hydra.utils.instantiate(model_config.model)

    @overload
    @classmethod
    def from_config(cls, dataset_name: Literal[None, 'clevr_6', 'clevr_10'] = None):
        model_config_path = CONFIG / 'slot_attention.yaml'
        model_config = OmegaConf.load(model_config_path)
        if dataset_name is not None and dataset_name.startswith('clevr'):
            clevr_path = CONFIG / 'slot_attention-clevr.yaml'
            model_config = model_config.mergewith(OmegaConf.load(clevr_path))
            width = 128
            height = 128
            max_num_objects = 7 if dataset_name == 'clevr_6' else 11
            model_config = model_config.merge_with_dotlist([f"dataset.width={width}",
                                                            f"dataset.height={height}",
                                                            f"dataset.max_num_objects={max_num_objects}"])


    @classmethod
    def from_custom_config(cls, config: DictConfig):
        model = hydra.utils.instantiate(config)
        assert isinstance(model, SlotAttentionAE)
        return model
from typing import Union, Literal, Optional

import hydra.utils
import torch
from omegaconf import OmegaConf, DictConfig
from torch import nn

from slot_attention.encoder_decoder import Encoder, Decoder
from slot_attention.paths import CONFIG
from slot_attention.slot_attention_module import SlotAttentionModule


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

    def __init__(self, width: int, height: int, encoder: Union[nn.Module, Encoder], decoder: Union[nn.Module, Decoder],
                 slot_attention_module: Union[nn.Module, SlotAttentionModule], input_channels: int = 3,
                 w_broadcast: Optional[int] = None, h_broadcast: Optional[int] = None):
        super().__init__()
        self.width = width
        self.height = height
        self.encoder = encoder
        self.decoder = decoder
        self.slot_attention_module = slot_attention_module
        self.input_channels = input_channels
        self.w_broadcast = w_broadcast
        self.h_broadcast = h_broadcast
        if self.w_broadcast is None:
            self.w_broadcast = self.width
        if self.h_broadcast is None:
            self.h_broadcast = self.height

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

    @classmethod
    def from_config(cls, dataset_name: Literal[None, 'clevr_6', 'clevr_10'] = None, dataset_width: int = 64, dataset_height: int = 64, max_num_objects: int = 5):
        model_config_path = CONFIG / 'slot_attention.yaml'
        model_config = OmegaConf.load(model_config_path)
        if dataset_name is not None and dataset_name.startswith('clevr'):
            clevr_path = CONFIG / 'slot_attention-clevr.yaml'
            model_config = model_config.mergewith(OmegaConf.load(clevr_path))
            dataset_width = 128
            dataset_height = 128
            max_num_objects = 7 if dataset_name == 'clevr_6' else 11
            model_config.merge_with_dotlist([f"dataset.width={dataset_width}",
                                             f"dataset.height={dataset_height}",
                                             f"dataset.max_num_objects={max_num_objects}"])
            return hydra.utils.instantiate(model_config.model)
        elif dataset_name is None:
            model_config.merge_with_dotlist([f"dataset.width={dataset_width}",
                                             f"dataset.height={dataset_height}",
                                             f"dataset.max_num_objects={max_num_objects}"])
        else:
            raise ValueError(f'{dataset_name} argument not start with "clevr" or is None')
        return hydra.utils.instantiate(model_config.model)

    @classmethod
    def from_custom_config(cls, config: DictConfig):
        model = hydra.utils.instantiate(config.model)
        assert isinstance(model, SlotAttentionAE)
        return model
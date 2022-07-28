import omegaconf
import torch.nn as nn
from omegaconf import OmegaConf, DictConfig

from slot_attention_pytorch.model import SlotAttentionAE
from slot_attention_pytorch.positional_embedding import PositionalEmbedding


def test_init_slot_attention():
    model = SlotAttentionAE.from_config()
    assert isinstance(model, SlotAttentionAE)
    assert model.width == 64
    assert model.height == 64
    assert model.slot_attention_module.latent_size == 64
    expected_encoder = nn.Sequential(
        nn.Conv2d(3, 32, kernel_size=5, padding=2),
        nn.ReLU(),
        nn.Conv2d(32, 32, kernel_size=5, padding=2),
        nn.ReLU(),
        nn.Conv2d(32, 32, kernel_size=5, padding=2),
        nn.ReLU(),
        nn.Conv2d(32, 32, kernel_size=5, padding=2),
        nn.ReLU(),
        PositionalEmbedding(64, 64, 32),
        nn.Flatten(2, 3),
        nn.GroupNorm(1, 32, affine=True, eps=0.001),
        nn.Conv2d(32, 32, 1),
        nn.ReLU(),
        nn.Conv2d(32, 32, 1)
    )
    iter_named_param_exp_encoder = iter(expected_encoder.named_parameters())
    for name, p in model.encoder.named_parameters():
        exp_name, exp_p = next(iter_named_param_exp_encoder)
        assert name == exp_name
        assert exp_p.shape == p.shape


def test_init_slot_attention_different_dataset_specs():
    model: SlotAttentionAE = SlotAttentionAE.from_config(dataset_width=32, dataset_height=32, max_num_objects=10)
    assert model.width == 32
    assert model.height == 32
    assert model.slot_attention_module.num_slots == 10
    expected_encoder = nn.Sequential(
        nn.Conv2d(3, 32, kernel_size=5, padding=2),
        nn.ReLU(),
        nn.Conv2d(32, 32, kernel_size=5, padding=2),
        nn.ReLU(),
        nn.Conv2d(32, 32, kernel_size=5, padding=2),
        nn.ReLU(),
        nn.Conv2d(32, 32, kernel_size=5, padding=2),
        nn.ReLU(),
        PositionalEmbedding(64, 64, 32),
        nn.Flatten(2, 3),
        nn.GroupNorm(1, 32, affine=True, eps=0.001),
        nn.Conv2d(32, 32, 1),
        nn.ReLU(),
        nn.Conv2d(32, 32, 1)
    )
    iter_named_param_exp_encoder = iter(expected_encoder.named_parameters())
    for name, p in model.encoder.named_parameters():
        exp_name, exp_p = next(iter_named_param_exp_encoder)
        assert name == exp_name
        assert exp_p.shape == p.shape


def test_init_slot_attention_custom_specs():
    custom_config: str = """
    dataset:
      width: 32
      height: 32
      max_num_objects: 3
    model:
      _target_: slot_attention_pytorch.model.SlotAttentionAE
      encoder:
        _target_: torch.nn.Sequential
        _args_:
          - _target_: torch.nn.Conv2d
            in_channels: ${model.input_channels}
            out_channels: 16
            kernel_size: 5
            padding: 2
          - _target_: torch.nn.ReLU
          - _target_: torch.nn.Conv2d
            in_channels: 16
            out_channels: 16
            kernel_size: 5
            padding: 2
          - _target_: torch.nn.ReLU
          - _target_: torch.nn.Conv2d
            in_channels: 16
            out_channels: 16
            kernel_size: 5
            padding: 2
          - _target_: torch.nn.ReLU
          - _target_: slot_attention_pytorch.positional_embedding.PositionalEmbedding
            width: ${dataset.width}
            height: ${dataset.height}
            channels: 16
          - _target_: torch.nn.Flatten
            start_dim: 2
            end_dim: 3
          - _target_: torch.nn.GroupNorm
            num_groups: 1
            num_channels: 16
            affine: true
            eps: 0.001
          - _target_: torch.nn.Conv2d
            in_channels: 16
            out_channels: 16
            kernel_size: 1
          - _target_: torch.nn.ReLU
          - _target_: torch.nn.Conv2d
            in_channels: 16
            out_channels: 16
            kernel_size: 1
      decoder:
        _target_: torch.nn.Sequential
        _args_:
          - _target_: slot_attention_pytorch.positional_embedding.PositionalEmbedding
            width: ${dataset.width}
            height: ${dataset.height}
            channels: 16
          - _target_: torch.nn.Conv2d
            in_channels: 16
            out_channels: 16
            kernel_size: 5
            padding: 2
          - _target_: torch.nn.ReLU
          - _target_: torch.nn.Conv2d
            in_channels: 16
            out_channels: 16
            kernel_size: 5
            padding: 2
          - _target_: torch.nn.ReLU
          - _target_: torch.nn.Conv2d
            in_channels: 16
            out_channels: 16
            kernel_size: 5
            padding: 2
          - _target_: torch.nn.ReLU
          - _target_: torch.nn.Conv2d
            in_channels: 16
            out_channels: 4
            kernel_size: 3
            padding: 1
      slot_attention_module:
        _target_: slot_attention_pytorch.slot_attention_module.SlotAttentionModule
        num_slots: ${dataset.max_num_objects}
        channels_enc: 16
        latent_size: 16
        attention_iters: 5
        eps: 1e-8
        mlp_size: 128
      w_broadcast: 10
      h_broadcast: 10
      width: 16
      height: 16
      input_channels: 3
    """
    custom_config: DictConfig = OmegaConf.create(custom_config)
    model = SlotAttentionAE.from_custom_config(custom_config)
    assert model.w_broadcast == 10
    assert model.h_broadcast == 10
    assert model.width == 16
    assert model.height == 16
    assert model.slot_attention_module.num_slots == 3
    expected_encoder = nn.Sequential(
        nn.Conv2d(3, 16, kernel_size=5, padding=2),
        nn.ReLU(),
        nn.Conv2d(16, 16, kernel_size=5, padding=2),
        nn.ReLU(),
        nn.Conv2d(16, 16, kernel_size=5, padding=2),
        nn.ReLU(),
        PositionalEmbedding(32, 32, 16),
        nn.Flatten(2, 3),
        nn.GroupNorm(1, 16, affine=True, eps=0.001),
        nn.Conv2d(16, 16, 1),
        nn.ReLU(),
        nn.Conv2d(16, 16, 1)
    )
    iter_named_param_exp_encoder = iter(expected_encoder.named_parameters())
    for name, p in model.encoder.named_parameters():
        exp_name, exp_p = next(iter_named_param_exp_encoder)
        assert name == exp_name
        assert exp_p.shape == p.shape

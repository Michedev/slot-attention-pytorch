import torch.nn as nn

from slot_attention.model import SlotAttentionAE
from slot_attention.positional_embedding import PositionalEmbedding


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
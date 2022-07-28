import torch

from slot_attention.model import SlotAttentionAE


def test_forward():
    model = SlotAttentionAE.from_config()

    dummy_batch = torch.randn(32, 3, model.width, model.height)

    prediction = model(dummy_batch)

    assert isinstance(prediction, dict)

    assert sorted(prediction.keys()) == sorted(['z', 'loss', 'mask', 'slot'])


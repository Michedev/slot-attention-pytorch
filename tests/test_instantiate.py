from slot_attention.model import SlotAttentionAE


def test_init_slot_attention():
    model = SlotAttentionAE.from_config()
    assert isinstance(model, SlotAttentionAE)
    assert model.width == 64
    assert model.height == 64
    assert model.slot_attention_module.latent_space
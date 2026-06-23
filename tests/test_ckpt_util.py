import torch

from utils.ckpt_util import _strip_compiled_prefix


def test_strip_compiled_prefix_handles_ddp_and_compile_wrappers():
    state = {
        "module._orig_mod.class_embed.weight": torch.ones(1),
        "module._orig_mod.model.blocks.0.norm1.weight": torch.ones(1),
        "_orig_mod.final_layer.linear.weight": torch.ones(1),
        "plain.weight": torch.ones(1),
    }

    stripped = _strip_compiled_prefix(state)

    assert "class_embed.weight" in stripped
    assert "model.blocks.0.norm1.weight" in stripped
    assert "final_layer.linear.weight" in stripped
    assert "plain.weight" in stripped

import torch

from drift_loss import drift_loss


def test_drift_loss_shapes():
    b, cg, cp, cn, d = 2, 4, 5, 3, 8
    gen = torch.randn(b, cg, d)
    pos = torch.randn(b, cp, d)
    neg = torch.randn(b, cn, d)
    loss, info = drift_loss(gen=gen, fixed_pos=pos, fixed_neg=neg)
    assert loss.shape == (b,)
    assert "scale" in info

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


def test_measure_weight_mode_matches_unit_weights():
    torch.manual_seed(0)
    gen = torch.randn(2, 4, 8)
    pos = torch.randn(2, 5, 8)
    neg = torch.randn(2, 3, 8)

    loss_post, _ = drift_loss(gen=gen, fixed_pos=pos, fixed_neg=neg, weight_mode="post")
    loss_measure, _ = drift_loss(gen=gen, fixed_pos=pos, fixed_neg=neg, weight_mode="measure")

    assert torch.allclose(loss_post, loss_measure)


def test_measure_weight_mode_matches_duplicated_positive_mass():
    torch.manual_seed(0)
    gen = torch.randn(1, 3, 8)
    pos = torch.randn(1, 1, 8)
    neg = torch.randn(1, 2, 8)

    duplicated_loss, _ = drift_loss(
        gen=gen,
        fixed_pos=pos.repeat(1, 2, 1),
        fixed_neg=neg,
        weight_mode="measure",
    )
    weighted_loss, _ = drift_loss(
        gen=gen,
        fixed_pos=pos,
        fixed_neg=neg,
        weight_pos=torch.full((1, 1), 2.0),
        weight_mode="measure",
    )
    post_loss, _ = drift_loss(
        gen=gen,
        fixed_pos=pos,
        fixed_neg=neg,
        weight_pos=torch.full((1, 1), 2.0),
        weight_mode="post",
    )

    assert torch.allclose(duplicated_loss, weighted_loss, atol=2e-3, rtol=1e-4)
    assert (duplicated_loss - weighted_loss).abs() < (duplicated_loss - post_loss).abs()

import torch

from drift_loss import _batched_log_prob, _make_mog, drift_loss, jsd_mog_loss, likelihood_mog_loss


def test_drift_loss_shapes():
    b, cg, cp, cn, d = 2, 4, 5, 3, 8
    gen = torch.randn(b, cg, d)
    pos = torch.randn(b, cp, d)
    neg = torch.randn(b, cn, d)
    loss, info = drift_loss(gen=gen, fixed_pos=pos, fixed_neg=neg)
    assert loss.shape == (b,)
    assert "scale" in info


def test_jsd_mog_loss_uses_sample_axis_as_mixture_axis():
    locs = torch.randn(2, 7, 3, 4)
    p = _make_mog(locs, torch.tensor(1.0))

    assert p.batch_shape == (2, 7)
    assert p.event_shape == (4,)
    assert _batched_log_prob(p, locs).shape == (2, 7, 3)


def test_jsd_mog_loss_backprops_to_tensor_sigma():
    torch.manual_seed(0)
    log_sigma = torch.nn.Parameter(torch.zeros(4))
    gen = torch.randn(2, 3, 4, requires_grad=True)
    real = torch.randn(2, 5, 4)

    loss = jsd_mog_loss(gen, real, sigma=log_sigma.exp())
    loss.backward()

    assert torch.isfinite(loss)
    assert log_sigma.grad is not None
    assert log_sigma.grad.shape == (4,)
    assert torch.isfinite(log_sigma.grad).all()


def test_likelihood_mog_loss_backprops_to_tensor_sigma():
    torch.manual_seed(0)
    log_sigma = torch.nn.Parameter(torch.zeros(4))
    gen = torch.randn(2, 3, 4, requires_grad=True)
    real = torch.randn(2, 5, 4)

    loss = likelihood_mog_loss(gen, real, sigma=log_sigma.exp())
    loss.backward()

    assert torch.isfinite(loss)
    assert log_sigma.grad is not None
    assert log_sigma.grad.shape == (4,)
    assert torch.isfinite(log_sigma.grad).all()

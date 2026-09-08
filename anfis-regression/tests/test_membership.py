import numpy as np
import pytest
import torch

from anfis.membership import MF_FAMILIES, build_mf


@pytest.mark.parametrize("family", MF_FAMILIES)
def test_shape_and_range(family):
    """Every family maps [N, d] -> [N, d, M] with memberships in [0, 1]."""
    mf = build_mf(family, n_inputs=3, n_mfs=4)
    mf.initialize(
        centers=torch.linspace(-2, 2, 4).repeat(3, 1),
        widths=torch.full((3, 4), 0.8),
    )
    x = torch.randn(64, 3)
    mu = mf(x)
    assert mu.shape == (64, 3, 4)
    assert torch.all(mu >= -1e-6), f"{family} produced negative memberships"
    assert torch.all(mu <= 1.0 + 1e-5), f"{family} exceeded 1"


@pytest.mark.parametrize("family", MF_FAMILIES)
def test_gradients_reach_every_premise_parameter(family):
    """A backward pass must touch all premise parameters, not just centres."""
    mf = build_mf(family, n_inputs=2, n_mfs=3)
    mf.initialize(torch.linspace(-1, 1, 3).repeat(2, 1), torch.full((2, 3), 0.6))
    x = torch.linspace(-2, 2, 50).repeat(2, 1).T.contiguous()
    mf(x).sum().backward()
    for name, p in mf.named_parameters():
        assert p.grad is not None, f"{family}.{name} received no gradient"
        assert torch.isfinite(p.grad).all(), f"{family}.{name} gradient is not finite"


@pytest.mark.parametrize("family", MF_FAMILIES)
def test_peaks_near_initialised_centre(family):
    """Membership should be maximal near the centre it was initialised at."""
    mf = build_mf(family, n_inputs=1, n_mfs=1)
    mf.initialize(torch.tensor([[0.5]]), torch.tensor([[0.4]]))
    grid = torch.linspace(-3, 4, 2001).unsqueeze(1)
    mu = mf(grid)[:, 0, 0]
    peak = grid[torch.argmax(mu), 0].item()
    assert abs(peak - 0.5) < 0.5, f"{family} peaks at {peak}, expected near 0.5"
    assert mu.max() > 0.5


@pytest.mark.parametrize("family", MF_FAMILIES)
def test_param_count_matches_declaration(family):
    mf = build_mf(family, n_inputs=4, n_mfs=2)
    counted = sum(p.numel() for p in mf.parameters())
    assert counted == mf.n_premise_params == 4 * 2 * mf.n_params_per_mf


def test_gaussian_matches_closed_form():
    mf = build_mf("gaussian", 1, 1)
    mf.initialize(torch.tensor([[0.0]]), torch.tensor([[1.0]]))
    x = torch.tensor([[0.0], [1.0], [2.0]])
    got = mf(x)[:, 0, 0].detach().numpy()
    want = np.exp(-0.5 * np.array([0.0, 1.0, 2.0]) ** 2)
    np.testing.assert_allclose(got, want, rtol=1e-3, atol=1e-3)


def test_smooth_variants_have_no_kinks():
    """The smooth triangular surrogate has a continuous second difference."""
    hard = build_mf("triangular", 1, 1, smooth=False)
    soft = build_mf("triangular", 1, 1, smooth=True, tau=0.1)
    for m in (hard, soft):
        m.initialize(torch.tensor([[0.0]]), torch.tensor([[1.0]]))
    grid = torch.linspace(-3, 3, 4001).unsqueeze(1)
    def curvature(m):
        return np.abs(np.diff(m(grid)[:, 0, 0].detach().numpy(), n=2)).max()

    assert curvature(soft) < curvature(hard)


def test_unknown_family_raises():
    with pytest.raises(KeyError):
        build_mf("nonexistent", 2, 2)

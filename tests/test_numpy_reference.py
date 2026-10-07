"""Cross-check the PyTorch GRPO implementation against an independent NumPy reference."""

import numpy as np
import pytest
import torch

from bitnet_rl import numpy_ref as npr
from bitnet_rl.grpo import group_advantages, grpo_loss, kl_k3


def _random_group(seed, g=5, t=7):
    rng = np.random.default_rng(seed)
    logp = rng.normal(-1.5, 0.5, (g, t))
    old = logp + rng.normal(0, 0.25, (g, t))  # ratios on both sides of the clip range
    ref = logp + rng.normal(0, 0.3, (g, t))
    adv = rng.normal(size=g)
    mask = (rng.random((g, t)) > 0.2).astype(np.float64)
    mask[:, 0] = 1.0
    return logp, old, ref, adv, mask


def test_advantages_match_torch():
    r = np.random.default_rng(0).normal(size=(3, 8))
    np.testing.assert_allclose(
        npr.group_advantages(r), group_advantages(torch.tensor(r)).numpy(), rtol=1e-9, atol=1e-9
    )


def test_kl_matches_torch():
    rng = np.random.default_rng(1)
    a, b = rng.normal(size=50), rng.normal(size=50)
    np.testing.assert_allclose(npr.kl_k3(a, b), kl_k3(torch.tensor(a), torch.tensor(b)).numpy(), atol=1e-12)


@pytest.mark.parametrize("seed", range(5))
@pytest.mark.parametrize("clip_eps,beta", [(0.2, 0.001), (0.1, 0.05), (10.0, 0.0)])
def test_loss_value_matches_torch(seed, clip_eps, beta):
    logp, old, ref, adv, mask = _random_group(seed)
    expected = npr.grpo_loss(logp, old, ref, adv, mask, clip_eps, beta)
    t = lambda x: torch.tensor(x)  # noqa: E731
    got, _ = grpo_loss(t(logp), t(old), t(ref), t(adv), t(mask), clip_eps, beta)
    assert float(got) == pytest.approx(expected, rel=1e-9, abs=1e-12)


@pytest.mark.parametrize("seed", range(5))
def test_analytic_gradient_matches_torch_autograd(seed):
    logp, old, ref, adv, mask = _random_group(seed)
    lp = torch.tensor(logp, requires_grad=True)
    loss, _ = grpo_loss(lp, torch.tensor(old), torch.tensor(ref), torch.tensor(adv),
                        torch.tensor(mask), 0.2, 0.01)
    loss.backward()
    analytic = npr.grpo_loss_grad(logp, old, ref, adv, mask, 0.2, 0.01)
    np.testing.assert_allclose(analytic, lp.grad.numpy(), rtol=1e-8, atol=1e-12)


def test_analytic_gradient_matches_finite_differences():
    logp, old, ref, adv, mask = _random_group(3, g=3, t=4)
    analytic = npr.grpo_loss_grad(logp, old, ref, adv, mask, 0.2, 0.01)
    numeric = np.zeros_like(logp)
    h = 1e-6
    for idx in np.ndindex(*logp.shape):
        up, dn = logp.copy(), logp.copy()
        up[idx] += h
        dn[idx] -= h
        numeric[idx] = (
            npr.grpo_loss(up, old, ref, adv, mask, 0.2, 0.01)
            - npr.grpo_loss(dn, old, ref, adv, mask, 0.2, 0.01)
        ) / (2 * h)
    np.testing.assert_allclose(analytic, numeric, rtol=1e-5, atol=1e-8)


def test_group_of_one_is_rejected():
    with pytest.raises(ValueError):
        npr.group_advantages(np.array([1.0]))


def test_tabular_grpo_lowers_perplexity_of_rewarded_output_in_three_iterations():
    improved, finals = 0, []
    for seed in range(50):
        h = npr.tabular_grpo(iters=3, seed=seed)
        assert h["perplexity"][0] == pytest.approx(6.0)  # uniform over 6 actions
        improved += h["perplexity"][-1] < h["perplexity"][0]
        finals.append(h["perplexity"][-1])
    assert improved == 50
    assert np.mean(finals) < 3.0


def test_tabular_grpo_makes_no_update_when_all_rewards_are_equal():
    # an all-zero-reward group has zero advantage everywhere, so the policy must not move
    found = False
    for seed in range(100):
        h = npr.tabular_grpo(group_size=2, iters=1, seed=seed, beta=0.0)
        if h["mean_reward"][0] == 0.0:
            found = True
            assert h["perplexity"][-1] == pytest.approx(h["perplexity"][0])
    assert found

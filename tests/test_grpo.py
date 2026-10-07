import math

import pytest
import torch

from bitnet_rl.bitlinear import BitLinear
from bitnet_rl.grpo import GRPOConfig, group_advantages, grpo_loss, kl_k3, sample_group, token_logprobs
from bitnet_rl.rewards import accuracy_reward, extract_answer, format_reward, rule_reward
from bitnet_rl.toy import CharTokenizer, TinyCausalLM


def test_advantages_are_group_normalised():
    r = torch.tensor([[1.0, 2.0, 3.0, 4.0], [0.0, 0.0, 1.0, 1.0]])
    a = group_advantages(r, eps=0.0)
    assert torch.allclose(a.mean(-1), torch.zeros(2), atol=1e-6)
    assert torch.allclose(a.std(-1), torch.ones(2), atol=1e-6)


def test_identical_rewards_give_zero_advantage():
    assert torch.all(group_advantages(torch.full((8,), 2.0)) == 0)


def test_group_of_one_is_rejected():
    with pytest.raises(ValueError):
        group_advantages(torch.tensor([1.0]))


def test_kl_k3_is_zero_at_equality_and_nonnegative():
    x = torch.randn(100)
    assert torch.allclose(kl_k3(x, x), torch.zeros(100), atol=1e-7)
    assert (kl_k3(x, torch.randn(100)) >= -1e-6).all()


def _loss_inputs(g=4, t=5):
    logp = torch.randn(g, t, requires_grad=True)
    return logp, logp.detach().clone(), logp.detach().clone(), torch.ones(g, t)


def test_positive_advantage_increases_logprob():
    logp, old, ref, mask = _loss_inputs()
    adv = torch.tensor([1.0, 1.0, 1.0, 1.0])
    loss, _ = grpo_loss(logp, old, ref, adv, mask, clip_eps=0.2, beta=0.0)
    loss.backward()
    assert (logp.grad < 0).all()  # descending the loss raises logp


def test_clipping_blocks_gradient_beyond_trust_region():
    logp = torch.full((1, 3), 1.0, requires_grad=True)
    old = torch.zeros(1, 3)
    loss, st = grpo_loss(logp, old, old, torch.tensor([1.0]), torch.ones(1, 3), 0.2, 0.0)
    loss.backward()
    assert torch.all(logp.grad == 0)
    assert st["clip_frac"] == 1.0


def test_negative_advantage_is_clipped_below():
    logp = torch.full((1, 3), -1.0, requires_grad=True)
    loss, _ = grpo_loss(logp, torch.zeros(1, 3), torch.zeros(1, 3), torch.tensor([-1.0]),
                        torch.ones(1, 3), 0.2, 0.0)
    loss.backward()
    assert torch.all(logp.grad == 0)


def test_masked_tokens_do_not_contribute():
    logp, old, ref, _ = _loss_inputs()
    mask = torch.ones(4, 5)
    mask[:, 3:] = 0
    adv = torch.tensor([1.0, -1.0, 0.5, -0.5])
    grpo_loss(logp, old, ref, adv, mask, 0.2, 0.1)[0].backward()
    assert torch.all(logp.grad[:, 3:] == 0)


def test_kl_penalty_pulls_towards_reference():
    ref = torch.zeros(2, 4)
    logp = torch.full((2, 4), 0.5, requires_grad=True)
    loss, _ = grpo_loss(logp, logp.detach(), ref, torch.zeros(2), torch.ones(2, 4), 0.2, 1.0)
    loss.backward()
    assert (logp.grad > 0).all()  # descending lowers logp back towards ref


def test_loss_is_length_normalised_per_sequence():
    # one long and one short sequence with equal per-token value contribute equally
    logp = torch.zeros(2, 4)
    mask = torch.tensor([[1.0, 1, 1, 1], [1.0, 0, 0, 0]])
    loss, _ = grpo_loss(logp, logp, logp, torch.tensor([1.0, 1.0]), mask, 0.2, 0.0)
    assert math.isclose(float(loss), -1.0, abs_tol=1e-6)


def test_format_and_accuracy_rewards():
    good = "<think>2+2 is 4</think><answer>4</answer>"
    assert format_reward(good) == 1.0
    assert format_reward("<answer>4</answer>") == 0.0
    assert format_reward("<think>x</think><answer>4") == 0.0
    assert extract_answer(good) == "4"
    assert extract_answer("so \\boxed{7}") == "7"
    assert accuracy_reward(good, "4") == 1.0
    assert accuracy_reward("<think>a</think><answer>4.0</answer>", "4") == 1.0
    assert accuracy_reward(good, "5") == 0.0
    assert accuracy_reward("no answer here", "4") == 0.0
    r = rule_reward(good, "4")
    assert (r.reward, r.accuracy, r.format) == (2.0, 1.0, 1.0)
    assert rule_reward("<think>a</think><answer>9</answer>", "4").reward == 1.0


def test_bitlinear_uses_ternary_weights_and_trains_via_ste():
    layer = BitLinear(16, 8, bias=False)
    assert set(layer.ternary_weights().unique().tolist()) <= {-1.0, 0.0, 1.0}
    layer(torch.randn(4, 16)).sum().backward()
    assert layer.weight.grad is not None and layer.weight.grad.abs().sum() > 0


@pytest.mark.parametrize("bitlinear", [False, True])
def test_sampling_and_logprobs_are_consistent(bitlinear):
    torch.manual_seed(0)
    tok = CharTokenizer()
    model = TinyCausalLM(tok.vocab_size, bitlinear=bitlinear)
    prompt = tok.encode("1+2=")
    comp, mask = sample_group(model, prompt, 6, 10, tok.eos_id, tok.pad_id)
    assert comp.shape == mask.shape and comp.shape[0] == 6
    # after EOS everything is masked out
    for row, m in zip(comp, mask):
        n = int(m.sum())
        assert (m[:n] == 1).all() and (m[n:] == 0).all()
        if n < comp.shape[1]:
            assert row[n - 1] == tok.eos_id
    lp = token_logprobs(model, prompt, comp)
    assert lp.shape == comp.shape and (lp <= 0).all()


def test_one_gradient_step_improves_rewarded_sequence():
    torch.manual_seed(0)
    tok = CharTokenizer()
    model = TinyCausalLM(tok.vocab_size)
    prompt = tok.encode("1+2=")
    target = torch.tensor([tok.encode("3") + [tok.eos_id]] * 2)
    other = torch.tensor([tok.encode("7") + [tok.eos_id]] * 2)
    comp = torch.cat([target[:1], other[:1], target[1:], other[1:]])
    mask = torch.ones_like(comp, dtype=torch.float)
    adv = group_advantages(torch.tensor([1.0, 0.0, 1.0, 0.0]))
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    with torch.no_grad():
        old = token_logprobs(model, prompt, comp)
    before = old[0].sum().item()
    loss, _ = grpo_loss(token_logprobs(model, prompt, comp), old, old, adv, mask, 0.2, 0.0)
    loss.backward()
    opt.step()
    after = token_logprobs(model, prompt, comp)[0].sum().item()
    assert after > before


def test_config_defaults_match_r1_choices():
    cfg = GRPOConfig()
    assert cfg.group_size == 16 and cfg.beta == 0.001 and cfg.temperature == 1.0

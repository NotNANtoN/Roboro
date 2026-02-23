import torch

from roboro.nn.blocks import CategoricalSupport


def test_categorical_support_initialization() -> None:
    """Verify bins are constructed correctly."""
    support = CategoricalSupport(v_min=-10.0, v_max=10.0, num_atoms=3)
    expected_bins = torch.tensor([-10.0, 0.0, 10.0])
    assert torch.allclose(
        support.support, expected_bins
    ), f"Expected {expected_bins}, got {support.support}"
    assert support.delta_z == 10.0


def test_expected_value_forward() -> None:
    """Verify that forward computes the correct expected scalar value from logits."""
    support = CategoricalSupport(v_min=-10.0, v_max=10.0, num_atoms=3)

    # Very high logit for the middle bin (0.0) means prob approx 1.0 for 0.0.
    logits = torch.tensor([[0.0, 100.0, 0.0]])
    expected_val = support(logits)
    assert torch.allclose(expected_val, torch.tensor([[0.0]]), atol=1e-4)

    # Uniform logits = uniform probability = mean of support (-10+0+10)/3 = 0.0
    logits_uniform = torch.tensor([[1.0, 1.0, 1.0]])
    expected_uniform = support(logits_uniform)
    assert torch.allclose(expected_uniform, torch.tensor([[0.0]]), atol=1e-4)

    # Deterministic prob for 10.0
    logits_high = torch.tensor([[-100.0, -100.0, 100.0]])
    expected_high = support(logits_high)
    assert torch.allclose(expected_high, torch.tensor([[10.0]]), atol=1e-4)


def test_to_categorical_scalar_projection() -> None:
    """Verify projection of a single scalar TD target into bins (simplified C51)."""
    support = CategoricalSupport(v_min=-10.0, v_max=10.0, num_atoms=3)

    # Target exactly on a bin
    target_zero = torch.tensor([[0.0]])
    dist_zero = support.to_categorical(target_zero)
    expected_zero = torch.tensor([[0.0, 1.0, 0.0]])
    assert torch.allclose(dist_zero, expected_zero)

    # Target halfway between 0.0 and 10.0
    target_five = torch.tensor([[5.0]])
    dist_five = support.to_categorical(target_five)
    expected_five = torch.tensor([[0.0, 0.5, 0.5]])
    assert torch.allclose(dist_five, expected_five)

    # Target out of bounds (should clamp to 10.0)
    target_oob = torch.tensor([[20.0]])
    dist_oob = support.to_categorical(target_oob)
    expected_oob = torch.tensor([[0.0, 0.0, 1.0]])
    assert torch.allclose(dist_oob, expected_oob)


def test_c51_full_distribution_projection() -> None:
    """Verify the full L2 projection algorithm for categorical DQN."""
    support = CategoricalSupport(v_min=-10.0, v_max=10.0, num_atoms=3)

    # Bins are [-10, 0, 10]
    next_dist = torch.tensor([[0.2, 0.5, 0.3]])
    rewards = torch.tensor([1.0])
    dones = torch.tensor([False])
    gamma = 0.5

    # Manual Calculation:
    # Tz = 1.0 + 0.5 * [-10, 0, 10] = [-4, 1, 6]
    # Projected support locations:
    # -4 is between -10 and 0. (Dist 6 to -10, Dist 4 to 0). Weights: 0.4 to -10, 0.6 to 0.
    #   -> Mass 0.2 gives: 0.08 to -10, 0.12 to 0.
    # 1 is between 0 and 10. (Dist 1 to 0, Dist 9 to 10). Weights: 0.9 to 0, 0.1 to 10.
    #   -> Mass 0.5 gives: 0.45 to 0, 0.05 to 10.
    # 6 is between 0 and 10. (Dist 6 to 0, Dist 4 to 10). Weights: 0.4 to 0, 0.6 to 10.
    #   -> Mass 0.3 gives: 0.12 to 0, 0.18 to 10.
    #
    # Total for -10: 0.08
    # Total for 0: 0.12 + 0.45 + 0.12 = 0.69
    # Total for 10: 0.05 + 0.18 = 0.23

    expected_dist = torch.tensor([[0.08, 0.69, 0.23]])

    projected_dist = support.c51_project(next_dist, rewards, dones, gamma)

    # Verify the mass is conserved (sums to 1)
    assert torch.allclose(projected_dist.sum(dim=-1), torch.ones(1))

    # Verify the exact distribution probabilities
    assert torch.allclose(
        projected_dist, expected_dist, atol=1e-5
    ), f"Expected {expected_dist}, got {projected_dist}"


def test_c51_project_with_done() -> None:
    """Verify that when done=True, the next distribution is ignored and mass goes only to reward."""
    support = CategoricalSupport(v_min=-10.0, v_max=10.0, num_atoms=3)

    next_dist = torch.tensor([[0.2, 0.5, 0.3]])
    rewards = torch.tensor([5.0])
    dones = torch.tensor([True])
    gamma = 0.5

    # Since done=True, Tz = R + 0 = 5.0
    # The entire distribution is collapsed to the scalar value 5.0.
    # 5.0 is exactly between 0 and 10, so it should split 50/50.

    expected_dist = torch.tensor([[0.0, 0.5, 0.5]])

    projected_dist = support.c51_project(next_dist, rewards, dones, gamma)

    assert torch.allclose(projected_dist, expected_dist, atol=1e-5)

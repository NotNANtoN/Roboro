"""Tests for core data types."""

import torch

from roboro.core.types import Batch


class TestBatch:
    def test_batch_creation(self):
        batch = Batch(
            obs=torch.randn(8, 16),
            actions=torch.randn(8, 4),
            rewards=torch.randn(8),
            next_obs=torch.randn(8, 16),
            dones=torch.zeros(8, dtype=torch.bool),
        )
        assert batch.batch_size == 8

    def test_batch_to_device(self):
        batch = Batch(
            obs=torch.randn(4, 8),
            actions=torch.randn(4, 2),
            rewards=torch.randn(4),
            next_obs=torch.randn(4, 8),
            dones=torch.zeros(4, dtype=torch.bool),
        )
        batch.to("cpu")  # should not crash
        assert batch.obs.device == torch.device("cpu")

    def test_batch_optional_fields_default_none(self):
        batch = Batch(
            obs=torch.randn(4, 8),
            actions=torch.randn(4, 2),
            rewards=torch.randn(4),
            next_obs=torch.randn(4, 8),
            dones=torch.zeros(4, dtype=torch.bool),
        )
        assert batch.weights is None
        assert batch.log_probs is None
        assert batch.indices is None
        assert batch.returns is None
        assert batch.extras == {}

    def test_batch_with_optional_fields(self):
        batch = Batch(
            obs=torch.randn(4, 8),
            actions=torch.randn(4, 2),
            rewards=torch.randn(4),
            next_obs=torch.randn(4, 8),
            dones=torch.zeros(4, dtype=torch.bool),
            weights=torch.ones(4),
            indices=torch.arange(4),
        )
        assert batch.weights is not None
        assert batch.indices is not None

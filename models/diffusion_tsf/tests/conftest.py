import pytest
import torch
import sys
import os

# Add the project root directory to sys.path
# conftest.py is in models/diffusion_tsf/tests/
# We want to go up 3 levels: tests -> diffusion_tsf -> models -> root
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
sys.path.append(root_dir)

@pytest.fixture
def device():
    return "cpu"

@pytest.fixture
def batch_size():
    return 4

@pytest.fixture
def seq_len():
    return 64

@pytest.fixture
def forecast_len():
    return 16

@pytest.fixture
def height():
    return 32

@pytest.fixture
def sample_batch(batch_size, seq_len):
    """Returns a sample batch of univariate time series."""
    # (batch, seq_len)
    return torch.randn(batch_size, seq_len)

@pytest.fixture
def sample_batch_multivariate(batch_size, seq_len):
    """Returns a sample batch of multivariate time series."""
    # (batch, num_vars=3, seq_len)
    return torch.randn(batch_size, 3, seq_len)

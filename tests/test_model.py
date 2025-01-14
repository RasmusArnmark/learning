import pytest
import torch
from src.model import MyAwesomeModel


def test_model_forward():
    """Test that the model produces outputs of the correct shape."""
    model = MyAwesomeModel()

    dummy_input = torch.randn(8, 1, 28, 28)  # Batch of 8, single-channel 28x28 images
    output = model(dummy_input)

    assert output.shape == (8, 10), "Output should have shape (batch_size, 10)"
    # Check if probabilities sum to 1
    assert torch.allclose(torch.exp(output).sum(dim=1), torch.tensor(1.0).expand(8), atol=1e-6), \
        "Output should sum to 1 after exponentiation"


def test_model_parameters():
    """Test that the model has the expected number of parameters."""
    model = MyAwesomeModel()
    total_params = sum(p.numel() for p in model.parameters())
    a = 1
    print(f"Total model parameters: {total_params}")
    # Adjust the expected value to the correct one (249,162).
    assert total_params == 249_162, "Unexpected number of model parameters"

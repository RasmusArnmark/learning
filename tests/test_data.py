import pytest
import torch
from src.data import normalize, preprocess_data, corrupt_mnist


@pytest.fixture
def dummy_data(tmp_path):
    """Generate dummy data for testing."""
    raw_dir = tmp_path / "raw"
    processed_dir = tmp_path / "processed"
    raw_dir.mkdir()
    processed_dir.mkdir()

    for i in range(6):
        torch.save(torch.randn(100, 28, 28), raw_dir / f"train_images_{i}.pt")
        torch.save(torch.randint(0, 10, (100,)), raw_dir / f"train_target_{i}.pt")

    torch.save(torch.randn(100, 28, 28), raw_dir / "test_images.pt")
    torch.save(torch.randint(0, 10, (100,)), raw_dir / "test_target.pt")

    return str(raw_dir), str(processed_dir)


def test_normalize():
    """Test that normalize scales data correctly."""
    data = torch.randn(100, 28, 28)
    normalized_data = normalize(data)

    assert torch.isclose(normalized_data.mean(), torch.tensor(0.0), atol=1e-5), "Mean should be approximately 0"
    assert torch.isclose(normalized_data.std(), torch.tensor(1.0), atol=1e-5), "Std should be approximately 1"


def test_preprocess_data(dummy_data):
    """Test that preprocess_data correctly processes and saves data."""
    raw_dir, processed_dir = dummy_data

    preprocess_data(raw_dir, processed_dir)

    train_images = torch.load(f"{processed_dir}/train_images.pt")
    train_target = torch.load(f"{processed_dir}/train_target.pt")
    test_images = torch.load(f"{processed_dir}/test_images.pt")
    test_target = torch.load(f"{processed_dir}/test_target.pt")

    assert train_images.shape[1:] == (1, 28, 28), "Train images should have shape (N, 1, 28, 28)"
    assert train_target.ndim == 1, "Train targets should be a 1D tensor"
    assert test_images.shape[1:] == (1, 28, 28), "Test images should have shape (N, 1, 28, 28)"
    assert test_target.ndim == 1, "Test targets should be a 1D tensor"


def test_corrupt_mnist(dummy_data):
    """Test that corrupt_mnist returns valid datasets."""
    raw_dir, processed_dir = dummy_data
    preprocess_data(raw_dir, processed_dir)

    train_set, test_set = corrupt_mnist()

    train_images, train_target = train_set[0]
    test_images, test_target = test_set[0]

    assert train_images.shape == (1, 28, 28), "Train images should have shape (1, 28, 28)"
    assert isinstance(train_target.item(), int), "Train targets should be integers"
    assert test_images.shape == (1, 28, 28), "Test images should have shape (1, 28, 28)"
    assert isinstance(test_target.item(), int), "Test targets should be integers"

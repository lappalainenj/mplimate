import matplotlib.pyplot as plt
import numpy as np
import pytest

from mplimate.imshow import Imshow


@pytest.fixture
def sample_images():
    """Create a sample array of images for testing."""
    # Create a 2x3x4x4 array (2 samples, 3 frames, 4x4 pixels)
    images = np.zeros((2, 3, 4, 4))
    # Add some patterns to distinguish frames
    for s in range(2):  # samples
        for f in range(3):  # frames
            images[s, f] = np.full((4, 4), f)
    return images


def test_imshow_initialization(sample_images):
    """Test basic initialization of Imshow class."""
    anim = Imshow(images=sample_images)

    assert anim.n_samples == 2
    assert anim.frames == 3
    assert anim.sleep == 0.01
    assert anim.update is True
    plt.close()


def test_imshow_custom_parameters(sample_images):
    """Test Imshow initialization with custom parameters."""
    fig, ax = plt.subplots()
    anim = Imshow(
        images=sample_images, fig=fig, ax=ax, update=False, sleep=0.02, cmap="viridis"
    )

    assert anim.fig is fig
    assert anim.ax is ax
    assert anim.sleep == 0.02
    assert anim.update is False
    assert anim.kwargs.get("cmap") == "viridis"
    plt.close()


def test_imshow_animation_steps(sample_images):
    """Test animation initialization and frame updates."""
    anim = Imshow(images=sample_images)

    # Test initialization
    anim.batch_sample = 0
    anim.init(frame=0)
    assert np.array_equal(anim.img.get_array(), sample_images[0, 0])

    # Test frame update
    anim.animate(frame=1)
    assert np.array_equal(anim.img.get_array(), sample_images[0, 1])

    plt.close()


def test_imshow_invalid_input():
    """Test that invalid inputs raise appropriate errors."""
    with pytest.raises(ValueError):
        # Invalid shape: missing frames dimension
        invalid_images = np.zeros((2, 4, 4))
        Imshow(images=invalid_images)

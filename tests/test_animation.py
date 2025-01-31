import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import pytest

from mplimate.animation import Animation, AnimationCollector


class SimpleAnimation(Animation):
    """Simple animation class for testing."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.frames = 5
        self.n_samples = 2

    def init(self, frame=0):
        if self.fig is None:
            self.init_figure(None, None, [4, 4])
        self.ax.clear()

    def animate(self, frame):
        self.ax.plot([0, 1], [0, frame])


@pytest.fixture
def simple_animation():
    """Fixture providing a basic animation instance."""
    anim = SimpleAnimation(sleep=None)  # Disable sleep for testing
    yield anim
    plt.close(anim.fig)


@pytest.fixture
def temp_dir():
    """Fixture providing a temporary directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


def test_animation_initialization():
    """Test basic animation initialization."""
    anim = SimpleAnimation()
    assert anim.frames == 5
    assert anim.n_samples == 2
    assert anim.update is True
    assert anim.batch_sample == 0


def test_get_indices(simple_animation):
    """Test _get_indices method."""
    # Test "all" option
    assert simple_animation._get_indices("frames", "all") == list(range(5))

    # Test specific indices
    assert simple_animation._get_indices("frames", [1, 3]) == [1, 3]

    # Test invalid input
    with pytest.raises(ValueError):
        simple_animation._get_indices("frames", 42)


def test_run_animation(simple_animation):
    """Test running animation."""
    simple_animation.run(frames=[0, 1], samples=[0], repeat=1)
    assert simple_animation.batch_sample == 0


def test_animation_export(simple_animation, temp_dir):
    """Test exporting animation to video."""
    output_file = "test_animation"

    # Test WebM export
    simple_animation.export(
        output_file,
        frames=[0, 1],
        samples=[0],
        dpi=72,
        framerate=30,
        dest_path=temp_dir,
        type="webm",
    )
    assert (temp_dir / f"{output_file}.webm").exists()

    # Test MP4 export
    simple_animation.export(
        output_file,
        frames=[0, 1],
        samples=[0],
        dpi=72,
        framerate=30,
        dest_path=temp_dir,
        type="mp4",
        delete_if_exists=True,
    )
    assert (temp_dir / f"{output_file}.mp4").exists()


def test_animation_collector():
    """Test AnimationCollector functionality."""

    class TestCollector(AnimationCollector):
        def __init__(self):
            super().__init__(sleep=None)
            self.animations = [SimpleAnimation(sleep=None), SimpleAnimation(sleep=None)]

    collector = TestCollector()

    # Test batch_sample propagation
    collector.batch_sample = 1
    assert all(anim.batch_sample == 1 for anim in collector.animations)

    # Test initialization
    collector.init()
    assert all(not anim.update for anim in collector.animations)

    # Clean up
    for anim in collector.animations:
        plt.close(anim.fig)
    plt.close(collector.fig)


def test_plot_single_frame(simple_animation):
    """Test plotting a single frame."""
    simple_animation.plot(sample=0, frame=1)
    assert simple_animation.batch_sample == 0  # Should restore original sample


def test_backend():
    """Test backend verification."""
    import matplotlib

    original_backend = matplotlib.get_backend()
    matplotlib.use("Agg")

    anim = SimpleAnimation()
    assert anim.config.is_notebook() is False

    matplotlib.use(original_backend)


def test_animation_interruption(simple_animation, capsys):
    """Test handling of KeyboardInterrupt during animation."""

    def mock_animate(*args):
        raise KeyboardInterrupt

    simple_animation.animate = mock_animate
    simple_animation.run(frames=[0], samples=[0])

    captured = capsys.readouterr()
    assert "Animation interrupted" in captured.out


def test_convert_errors(temp_dir):
    """Test error handling in convert function."""
    # Test invalid video type
    with pytest.raises(ValueError):
        simple_animation = SimpleAnimation()
        simple_animation.export(
            "test", frames=[0], samples=[0], dest_path=temp_dir, type="invalid"
        )

    # Test file exists error
    test_file = temp_dir / "test.mp4"
    test_file.touch()
    with pytest.raises(FileExistsError):
        simple_animation = SimpleAnimation()
        simple_animation.export(
            "test",
            frames=[0],
            samples=[0],
            dest_path=temp_dir,
            delete_if_exists=False,
            type="mp4",
        )

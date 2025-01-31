import logging
import os
import subprocess
import tempfile
from contextlib import contextmanager
from pathlib import Path
from time import sleep
from typing import Any, Callable, Generator, Iterable, List, Literal, Optional, Union

import ffmpeg
import matplotlib
import matplotlib.pyplot as plt
from IPython import display

__all__ = ["Animation", "AnimationCollector", "convert"]


class AnimationConfig:
    """Configuration management for animations.

    Handles environment variables and default settings for animation behavior.
    """

    def __init__(self):
        self.testing = os.getenv("TESTING", "False").lower() == "true"
        self.colab = bool(os.getenv("COLAB_RELEASE_TAG"))
        self.animation_dir = Path(os.getenv("ANIMATION_DIR", "animations"))
        self.mplbackend = os.getenv("MPLBACKEND")

        if self.mplbackend is not None:
            matplotlib.use(self.mplbackend)

        if not self.testing:
            matplotlib.interactive(True)

    def is_notebook(self) -> bool:
        """Check if running in a notebook environment."""
        return matplotlib.get_backend().lower() == "nbagg" or self.colab


@contextmanager
def temporary_animation_dir(
    base_path: Optional[Union[str, Path]] = None,
) -> Generator[Path, None, None]:
    """Create and manage a temporary directory for animation frames.

    Args:
        base_path: Optional base path for the temporary directory. If provided,
                  the temporary directory will be created under this path.

    Yields:
        Path object pointing to the temporary directory.
    """
    if base_path is not None:
        base_path = Path(base_path)
        base_path.mkdir(parents=True, exist_ok=True)
        temp_dir = tempfile.TemporaryDirectory(dir=str(base_path))
    else:
        temp_dir = tempfile.TemporaryDirectory()
    try:
        yield Path(temp_dir.name)
    finally:
        temp_dir.cleanup()


def check_ffmpeg() -> bool:
    """Verify ffmpeg installation.

    Returns:
        True if ffmpeg is available, False otherwise.

    Example:
        >>> if not check_ffmpeg():
        ...     raise RuntimeError("ffmpeg is required for video export")
    """
    try:
        subprocess.run(
            ["ffmpeg", "-version"], check=True, capture_output=True, text=True
        )
        return True
    except (subprocess.SubprocessError, FileNotFoundError):
        return False


class Animation:
    """Base class for creating animations.

    This class provides the foundation for creating animations using matplotlib.
    Subclasses must implement `init` and `animate` methods.

    Args:
        path: Path to save the animation. Defaults to config.animation_dir.
        fig: Existing matplotlib Figure instance.
        suffix: Suffix format string for the output path.
        sleep: Delay between frames in seconds.

    Examples:
        >>> class MyAnimation(Animation):
        ...     def init(self, frame=0):
        ...         self.init_figure(None, None, [4, 4])
        ...     def animate(self, frame):
        ...         self.ax.plot([0, frame], [0, 1])
        >>> anim = MyAnimation()
        >>> anim.run(frames=range(10))
    """

    def __init__(
        self,
        path: Optional[Union[str, Path]] = None,
        fig: Optional[matplotlib.figure.Figure] = None,
        suffix: str = "{}",
        sleep: Optional[float] = 0.1,
    ):
        self.config = AnimationConfig()
        self.fig = fig
        self.update = True
        self.batch_sample = 0
        self.frames = 0
        self.n_samples = 0
        self.path = (
            self.config.animation_dir if path is None else Path(path)
        ) / suffix.format(self.__class__.__name__)
        self.sleep = sleep
        self._temp_dir = None
        self._path = None

    def init_figure(
        self, fig: Optional[plt.Figure], ax: Optional[plt.Axes], figsize: List[int]
    ) -> None:
        """Initialize the figure.

        Args:
            fig: Existing Figure instance or None.
            ax: Existing Axis instance or None.
            figsize: Size of the figure.
        """
        if fig is None or ax is None:
            self.fig, self.ax = plt.subplots(figsize=figsize)
        else:
            self.fig = fig
            self.ax = ax

    def init(self, frame: int = 0) -> None:
        """Initialize the animation.

        Args:
            frame: Initial frame number.

        Raises:
            NotImplementedError: If not implemented by subclass.
        """
        raise NotImplementedError("Subclasses should implement this method.")

    def animate(self, frame: int) -> None:
        """Animate a single frame.

        Args:
            frame: Frame number to animate.

        Raises:
            NotImplementedError: If not implemented by subclass.
        """
        raise NotImplementedError("Subclasses should implement this method.")

    def update_figure(self, clear_output: bool = True) -> None:
        """Update the figure canvas.

        Args:
            clear_output: Whether to clear the previous output.
        """
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        if matplotlib.get_backend().lower() != "nbagg" or self.config.colab:
            display.display(self.fig)
            if clear_output:
                display.clear_output(wait=True)

    def animate_save(self, frame: int, dpi: int = 100) -> None:
        """Update the figure to the given frame and save it.

        Args:
            frame: Frame number to animate and save.
            dpi: Dots per inch for the saved image.
        """
        self.animate(frame)
        identifier = f"{self.batch_sample:04}_{frame:04}"
        self.fig.savefig(
            self._path / f"{identifier}.png",
            dpi=dpi,
            bbox_inches="tight",
            facecolor=self.fig.get_facecolor(),
            edgecolor="none",
        )

    def _get_indices(self, key: str, input: Union[str, Iterable]) -> list[int]:
        """Get sorted list of indices based on input.

        Args:
            key: Attribute name to get total number of indices.
            input: Input specifying which indices to return.

        Returns:
            Sorted list of indices.

        Raises:
            ValueError: If input is invalid.
        """
        total = getattr(self, key)
        _indices = set(range(total))
        if isinstance(input, str) and input == "all":
            indices = _indices
        elif isinstance(input, Iterable):
            indices = _indices.intersection(set(input))
        else:
            raise ValueError(f"Invalid input for {key}: {input}")
        return sorted(indices)

    def _run_animation_loop(
        self,
        frames_list: list[int],
        samples_list: list[int],
        frame_callback: Callable[[int], None],
    ) -> None:
        """Execute animation loop over samples and frames.

        Args:
            frames_list: List of frame indices to animate.
            samples_list: List of sample indices to animate.
            frame_callback: Function to call for each frame.
        """

        if self.config.testing:
            frames_list = frames_list[:1]
            samples_list = samples_list[:1]

        for sample in samples_list:
            self.batch_sample = sample
            for frame in frames_list:
                frame_callback(frame)
                if self.update:
                    self.update_figure()
                if self.sleep is not None:
                    sleep(self.sleep)

    def run(
        self,
        frames: Union[str, Iterable] = "all",
        samples: Union[str, Iterable] = "all",
        repeat: int = 1,
    ) -> None:
        """Play animation within a Jupyter notebook.

        Args:
            frames: Frames to animate.
            samples: Samples to animate.
            repeat: Number of times to repeat the animation.
        """
        frames_list, samples_list = self._initialize_animation(frames, samples)

        if self.config.testing:
            frames_list = frames_list[:1]
            samples_list = samples_list[:1]
            repeat = 1

        try:
            for _ in range(repeat):
                self._run_animation_loop(frames_list, samples_list, self.animate)
        except KeyboardInterrupt:
            print("Animation interrupted. Displaying last frame.")
            self.update_figure(clear_output=False)
            return

    def _initialize_animation(
        self, frames: Union[str, Iterable], samples: Union[str, Iterable]
    ) -> tuple[list[int], list[int]]:
        """Initialize the animation state.

        Args:
            frames: Frames to animate.
            samples: Samples to animate.

        Returns:
            Tuple of frames list and samples list.
        """
        self.update = True
        self.init()
        frames_list = self._get_indices("frames", frames)
        samples_list = self._get_indices("n_samples", samples)
        return frames_list, samples_list

    def plot(self, sample: int, frame: int) -> None:
        """Plot a single frame for a specific sample.

        Args:
            sample: Sample number to plot.
            frame: Frame number to plot.
        """
        previous_sample = self.batch_sample
        self.update = True
        self.init()
        self.batch_sample = sample
        self.animate(frame)
        self.batch_sample = previous_sample

    def _create_temp_dir(self, path: Optional[Union[str, Path]] = None) -> None:
        """Create a temporary directory as destination for the images.

        Args:
            path: Path to create the temporary directory.
        """
        self._temp_dir = tempfile.TemporaryDirectory()
        self._path = Path(self._temp_dir.name)

    def export(
        self,
        fname: str,
        frames: Union[str, Iterable] = "all",
        dpi: int = 100,
        framerate: int = 30,
        samples: Union[str, Iterable] = "all",
        delete_if_exists: bool = False,
        source_path: Optional[Union[str, Path]] = None,
        dest_path: Optional[Union[str, Path]] = None,
        type: Optional[Literal["mp4", "webm"]] = None,
    ) -> None:
        """Export animation to video file.

        Args:
            fname: Output filename. Extension (.mp4/.webm) determines output type
                unless type is specified.
            frames: Frame indices to export, or "all".
            dpi: Resolution of output frames.
            framerate: Frames per second in output video.
            samples: Sample indices to export, or "all".
            delete_if_exists: Whether to overwrite existing files.
            source_path: Path for temporary frame files.
            dest_path: Output directory for video file.
            type: Video format ("mp4" or "webm"). If None, inferred from fname
                extension or defaults to "webm".

        Raises:
            RuntimeError: If ffmpeg is not available.
            FileExistsError: If output file exists and delete_if_exists is False.

        Example:
            >>> anim = MyAnimation()
            >>> anim.export("my_animation", frames=range(10), type="mp4")
        """
        if not check_ffmpeg():
            raise RuntimeError(
                "ffmpeg is required for video export. "
                "Please install it using your system's package manager."
            )

        # Determine video type from filename if not explicitly specified
        fname_path = Path(fname)
        if type is None:
            if fname_path.suffix.lower() in [".mp4", ".webm"]:
                type = fname_path.suffix.lower()[1:]  # Remove the dot
            else:
                type = "webm"  # Default if no valid extension found

        # Strip any extension and add the correct one
        fname = fname_path.stem
        dest_path = Path(dest_path or self.path)
        output_file = dest_path / f"{fname}.{type}"
        if output_file.exists() and not delete_if_exists:
            raise FileExistsError(f"File {output_file} already exists.")

        with temporary_animation_dir(source_path) as temp_path:
            self._path = temp_path
            frames_list, samples_list = self._initialize_animation(frames, samples)

            try:
                self._run_animation_loop(
                    frames_list,
                    samples_list,
                    lambda frame: self.animate_save(frame, dpi=dpi),
                )
            except Exception as e:
                logging.error("Error during animation: %s", e)
                raise

            self.convert(
                fname,
                delete_if_exists=delete_if_exists,
                framerate=framerate,
                source_path=source_path,
                dest_path=dest_path,
                type=type,
            )

    def convert(
        self,
        fname: str,
        delete_if_exists: bool = False,
        framerate: int = 30,
        source_path: Optional[Union[str, Path]] = None,
        dest_path: Optional[Union[str, Path]] = None,
        type: Literal["mp4", "webm"] = "mp4",
    ) -> None:
        """Convert PNG files in the animations directory to video.

        Args:
            fname: Output filename.
            delete_if_exists: Whether to delete existing output file.
            framerate: Frame rate of the output video.
            source_path: Source path for input PNG files.
            dest_path: Destination path for the output video.
            type: Output video type.
        """
        dest_path = Path(dest_path or self.path)
        dest_path.mkdir(parents=True, exist_ok=True)
        convert(
            source_path or self._path,
            dest_path / f"{fname}.{type}",
            framerate,
            delete_if_exists,
            type=type,
        )

    def __del__(self):
        """Ensure cleanup of temporary directory."""
        if self._temp_dir is not None:
            self._temp_dir.cleanup()


def convert(
    directory: Union[str, Path],
    dest: Union[str, Path],
    framerate: int,
    delete_if_exists: bool,
    type: Literal["mp4", "webm"] = "mp4",
) -> None:
    """Convert PNG files in directory to MP4 or WebM.

    Args:
        directory: Source directory containing PNG files.
        dest: Destination path for the output video.
        framerate: Frame rate of the output video.
        delete_if_exists: Whether to delete existing output file.
        type: Output video type.

    Raises:
        ValueError: If unsupported video type is specified.
        FileExistsError: If output file exists and delete_if_exists is False.
    """
    video = Path(dest)

    if type == "mp4":
        kwargs = dict(
            vcodec="libx264",
            vprofile="high",
            vlevel="4.0",
            vf="pad=ceil(iw/2)*2:ceil(ih/2)*2",  # to make sizes even
            pix_fmt="yuv420p",
            crf=18,
        )
    elif type == "webm":
        kwargs = dict(
            vcodec="libvpx-vp9",
            vf="pad=ceil(iw/2)*2:ceil(ih/2)*2",
            pix_fmt="yuva420p",
            crf=18,
            threads=4,
        )
    else:
        raise ValueError(f"Unsupported video type: {type}")

    if video.exists():
        if delete_if_exists:
            video.unlink()
        else:
            raise FileExistsError(f"File {video} already exists.")

    try:
        (
            ffmpeg.input(
                f"{directory}/*_*.png", pattern_type="glob", framerate=framerate
            )
            .output(str(video), **kwargs)
            .run(
                overwrite_output=True,
                quiet=True,
                capture_stdout=True,
                capture_stderr=True,
            )
        )
    except FileNotFoundError as e:
        if "ffmpeg" in str(e):
            logging.warning("Check ffmpeg installation: %s", e)
            return
        else:
            raise
    except ffmpeg.Error as e:
        logging.error("ffmpeg error: %s", e.stderr.decode("utf8"))
        raise e

    logging.info("Created %s", video)


class AnimationCollector(Animation):
    """Collects multiple animations and updates them simultaneously.

    This class allows coordinating multiple animations to run in sync.
    Subclasses must populate the animations list.

    Example:
        >>> class MyCollector(AnimationCollector):
        ...     def __init__(self):
        ...         super().__init__()
        ...         self.animations = [MyAnimation(), MyAnimation()]
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.animations: list[Animation] = []

    def init(self, frame: int = 0) -> None:
        """Initialize all collected animations.

        Args:
            frame: Initial frame number.
        """
        for animation in self.animations:
            animation.init(frame)
            animation.update = False

    def animate(self, frame: int) -> None:
        """Animate all collected animations for a single frame.

        Args:
            frame: Frame number to animate.
        """
        for animation in self.animations:
            animation.animate(frame)
        if self.update:
            self.update_figure()

    def __setattr__(self, key: str, val: Any) -> None:
        """Set attributes for all Animation objects at once.

        Args:
            key: Attribute name to set.
            val: Value to set for the attribute.
        """
        if key == "batch_sample" and hasattr(self, "animations"):
            for animation in self.animations:
                setattr(animation, key, val)
        super().__setattr__(key, val)

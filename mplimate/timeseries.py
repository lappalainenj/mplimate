"""Time series animation."""

from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

from .animation import Animation


class TimeSeries(Animation):
    """Animates time series data using line plots.

    Args:
        data: Array of time series to animate (n_samples, n_frames, n_series).
        x: Optional x-axis values. If None, uses range(n_frames).
        fig: Existing Figure instance or None.
        ax: Existing Axis instance or None.
        update: Whether to update the canvas after an animation step.
        figsize: Size of the figure.
        sleep: Time to sleep between frames.
        **kwargs: Additional arguments passed to plt.plot.

    Example:
        >>> # Create sample data: 2 samples, 100 frames, 3 time series each
        >>> t = np.linspace(0, 10, 100)
        >>> data = np.zeros((2, 100, 3))
        >>> for i in range(2):  # For each sample
        ...     for j in range(3):  # For each time series
        ...         phase = 2 * np.pi * i / 2 + j * np.pi / 3
        ...         data[i, :, j] = np.sin(t + phase)
        >>> anim = TimeSeries(data, x=t, sleep=0.05)
        >>> anim.run()
    """

    def __init__(
        self,
        data: np.ndarray,
        x: Optional[np.ndarray] = None,
        fig: Optional[plt.Figure] = None,
        ax: Optional[plt.Axes] = None,
        update: bool = True,
        figsize: Tuple[int, int] = (8, 4),
        sleep: float = 0.01,
        **kwargs,
    ) -> None:
        super().__init__(None, fig)

        if len(data.shape) != 3:
            raise ValueError(
                f"Expected 3D array (n_samples, n_frames, n_series), got "
                f"shape {data.shape}"
            )

        self.init_figure(fig, ax, figsize)
        self.data = data
        self.n_samples, self.frames, self.n_series = data.shape
        self.x = x if x is not None else np.arange(self.frames)
        self.kwargs = kwargs
        self.update = update
        self.sleep = sleep
        self.lines = []

    def init(self, frame: int = 0) -> None:
        """Initialize the animation with empty lines.

        Args:
            frame: Initial frame number (unused).
        """
        self.ax.set_xlim(self.x[0], self.x[-1])
        self.ax.set_ylim(np.min(self.data) * 1.1, np.max(self.data) * 1.1)

        # Create a line for each time series
        self.lines = []
        for _ in range(self.n_series):
            (line,) = self.ax.plot([], [], **self.kwargs)
            self.lines.append(line)

    def animate(self, frame: int) -> None:
        """Animate a single frame by updating the line data.

        Args:
            frame: Frame number to animate.
        """
        # Update each line with data up to the current frame
        for i, line in enumerate(self.lines):
            line.set_data(
                self.x[: frame + 1], self.data[self.batch_sample, : frame + 1, i]
            )

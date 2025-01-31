# mplimate

Python library for creating animated matplotlib visualizations. Made for interactive use
in notebooks and for export with ffmpeg.

## Installation

```bash
pip install git+https://github.com/lappalainenj/mplimate.git
```

## Usage

Play an imshow animation:

```python
import numpy as np
import matplotlib
from mplimate import Imshow

# Arrays passed to the animation objects expect to be in shape (samples, frames, ...)
anim = Imshow(np.random.rand(1, 10, 64, 64))

# Run the animation
anim.run()

# Or export the animation
anim.export("animation.mp4", dpi=300, framerate=10, delete_if_exists=True)
```

More examples in the [examples](examples) folder.

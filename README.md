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
anim.run(frames="all", samples="all", repeat=1)
```

Export an imshow animation:

```python
anim.export(directory="animations", dest="animations/imshow.mp4", framerate=24)
```

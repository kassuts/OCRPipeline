import json
from collections import OrderedDict
from pathlib import Path

import numpy as np
from PIL import Image


def read_json(filename):
    filename = Path(filename)
    with filename.open('rt') as handle:
        return json.load(handle, object_hook=OrderedDict)


def load_image(config):
    return np.asarray(Image.open(config["image_directory"]))

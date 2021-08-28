import json
import os
from collections import OrderedDict
from pathlib import Path

import numpy as np
from PIL import Image


def read_json(filename):
    filename = Path(filename)
    with filename.open('rt') as handle:
        return json.load(handle, object_hook=OrderedDict)


def load_images_in_directory(config):
    directory = config.get('image_directory')
    image_dict = {}
    for filename in os.listdir(directory):
        filename_path = os.path.join(directory, filename)
        image_dict[filename_path] = np.asarray(Image.open(filename_path))
    return image_dict

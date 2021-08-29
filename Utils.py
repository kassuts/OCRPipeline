import json
import os
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

# from main import PROJECT_DIRECTORY
PROJECT_DIRECTORY = os.path.abspath(os.getcwd())


def read_json_file(filename):
    filename = Path(filename)
    with filename.open('rt') as handle:
        return json.load(handle, object_hook=OrderedDict)


def load_images_in_directory(config):
    directory = config.get('image_directory')
    if directory is None:
        raise Exception("Image directory not given")

    directory = os.path.join(PROJECT_DIRECTORY, directory)
    image_dict = {}
    for filename in os.listdir(directory):
        filename_path = os.path.join(directory, filename)
        image_dict[filename_path] = np.asarray(Image.open(filename_path))
    return image_dict


def remove_duplicates(df):
    if isinstance(df, pd.DataFrame):
        return df.drop_duplicates()

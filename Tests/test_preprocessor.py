import os
import unittest
from unittest.mock import MagicMock

from PreProcessor import PreProcessing
from Utils import read_json_file, load_images_in_directory

PROJECT_DIRECTORY = os.path.abspath(os.getcwd())


class PrePreprocessorTests(unittest.TestCase):
    def setUp(self):
        self.config = read_json_file(os.path.join(PROJECT_DIRECTORY, "configs/test_config.json"))
        self.image = MagicMock()

    def test_handle_unknown_preprocessor(self):
        with self.assertRaises(RuntimeError):
            PreProcessing(image=self.image, config=self.config).run()

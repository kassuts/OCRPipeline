from collections import OrderedDict

import pandas as pd

from OCRExtractor import ExtractText
from PreProcessor import PreProcessing
from Utils import read_json, load_images_in_directory
from store_extracted_text import store_text

PATH_TO_CONFIG = "/Users/eitankassuto/PycharmProjects/OCRPipeline/configs/default_config.json"


def main():
    config = read_json(PATH_TO_CONFIG)
    images = load_images_in_directory(config)
    storage_dataframe = pd.read_csv(config.get("storage_file_path")) if "storage_file_path" in config else OrderedDict()
    for image_path, image in images.items():
        img_clean = PreProcessing(image=image, config=config).run()
        img_to_text = ExtractText(image=img_clean).extractinformation()
        print(img_to_text)
        output_df, storage_dict = store_text(img_to_text, image_path, storage_dataframe)
        output_df.to_csv("/Users/eitankassuto/PycharmProjects/OCRPipeline/extracted_text.csv")


if __name__ == '__main__':
    main()

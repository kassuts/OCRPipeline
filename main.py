import os

import pandas as pd


from OCRExtractor import ExtractText
from PreProcessor import PreProcessing
from Utils import read_json, load_images_in_directory, remove_duplicates
from StoreExtractedText import PostProcessAndStore

PROJECT_DIRECTORY = os.path.abspath(os.getcwd())
PATH_TO_CONFIG = os.path.join(PROJECT_DIRECTORY, "configs/default_config.json")
OUTPUT_PATH = os.path.join(PROJECT_DIRECTORY, "extracted_text.csv")


def main():
    output_df = pd.DataFrame([])
    config = read_json(PATH_TO_CONFIG)
    images = load_images_in_directory(config)
    for image_path, image in images.items():
        img_clean = PreProcessing(image=image, config=config).run()
        extracted_text = ExtractText(image=img_clean).extractinformation()
        output_df, storage_dict = PostProcessAndStore(extracted_text=extracted_text,
                                                      config=config,
                                                      image_path=image_path).store_text()

    output_df = remove_duplicates(output_df)
    output_df.to_csv(OUTPUT_PATH)


if __name__ == '__main__':
    main()

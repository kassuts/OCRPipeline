import os
import pandas as pd


from OCRExtractor import ExtractText
from PreProcessor import PreProcessing
from Utils import read_json_file, load_images_in_directory, remove_duplicates
from StoreExtractedText import PostProcessAndStore

PROJECT_DIRECTORY = os.path.abspath(os.getcwd())
PATH_TO_CONFIG = os.path.join(PROJECT_DIRECTORY, "configs/default_config.json")
OUTPUT_PATH = os.path.join(PROJECT_DIRECTORY, "extracted_text.csv")


def main():
    output_df = pd.DataFrame([])
    config = read_json_file(PATH_TO_CONFIG)
    images = load_images_in_directory(config)
    for image_path, image in images.items():
        # Stage 1: Preprocess image
        img_clean = PreProcessing(image=image, config=config).run()
        # Stage 2: Extract Text from image
        extracted_text = ExtractText(image=img_clean).extract_information()
        # Stage 3: PostProcess and create structured file [dict] for storing extracted text
        output_df, storage_dict = PostProcessAndStore(extracted_text=extracted_text,
                                                      config=config,
                                                      image_path=image_path).store_text()
    # Remove duplicates and save to csv
    output_df = remove_duplicates(output_df)
    output_df.to_csv(OUTPUT_PATH)
    print("Output saved at {}".format(OUTPUT_PATH))


if __name__ == '__main__':
    main()

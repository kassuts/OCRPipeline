from OCRExtractor import ExtractText
from PreProcessor import PreProcessing
from Utils import read_json, load_image

PATH_TO_CONFIG = "/Users/eitankassuto/PycharmProjects/OCRPipeline/configs/default_config.json"


def main():
    config = read_json(PATH_TO_CONFIG)
    img = load_image(config)
    img_clean = PreProcessing(image=img, config=config).run()
    img_to_text = ExtractText(image=img_clean).extractinformation()
    print(img_to_text)


if __name__ == '__main__':
    main()

import numpy as np
from PIL import Image

from OCRExtractor import OCRExtractor
from PreProcessor import PreProcessing

if __name__ == '__main__':
    # path_to_image = "/Users/eitankassuto/PycharmProjects/OCRPipeline/pytesseract_test_image.png"
    path_to_image = "/Users/eitankassuto/PycharmProjects/OCRPipeline/test.png"
    img = Image.open(path_to_image)
    img_clean = PreProcessing().run(np.asarray(img))
    img_to_text = OCRExtractor().extractinformation(img_clean)
    print(img_to_text)

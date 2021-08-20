import cv2
import pytesseract
from PIL import Image

if __name__ == '__main__':
    # pytesseract.pytesseract.tesseract_cmd = r'C:\Users\USER\AppData\Local\Tesseract-OCR\tesseract.exe'
    path_to_image = "/Users/eitankassuto/PycharmProjects/OCRPipeline/pytesseract_test_image.png"
    print("print")
    print(pytesseract.image_to_string(Image.open(path_to_image)))

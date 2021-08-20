from pytesseract import pytesseract


class OCRExtractor:
    def __init__(self):
        pass
    @staticmethod
    def extractinformation(img):
        extractedInformation = pytesseract.image_to_string(img)
        return extractedInformation

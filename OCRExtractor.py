from pytesseract import pytesseract


class ExtractText:
    def __init__(self, image):
        self.img = image

    def extractinformation(self):
        return pytesseract.image_to_string(self.img)

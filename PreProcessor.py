import cv2
import numpy as np
from PIL import Image
from scipy.ndimage import interpolation as inter


class PreProcessing:
    def __init__(self):
        pass
    @staticmethod
    def imageResize(img):
        """
        Scaling of image 300 DPI

        :return:
        """
        img_resized = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)  # Inter Cubic
        return img_resized

    @staticmethod
    def bgrtogrey(img):
        """
        BGR to GRAY

        :return:
        """
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return img

    @staticmethod
    def checkDayOrNight(img, thrshld):
        """
        Increase Brightness

        :param thrshld:
        :return:
        """
        is_light = np.mean(img) > thrshld
        return 0 if is_light else 1  # 0 --> light and 1 -->dark

    @staticmethod
    def increaseBrightness(img):
        alpha = 1
        beta = 40
        img = cv2.addWeighted(img, alpha, np.zeros(img.shape, img.dtype), 0, beta)
        return img

    @staticmethod
    def handle_brightness(img, threshold):
        """
        if dark increase brightness
        :param img:
        :param threshold:
        :return:
        """

        if PreProcessing.checkDayOrNight(img, threshold) == 1:
            return PreProcessing.increaseBrightness(img)  # increase brightness

    @staticmethod
    def threshold(img):
        # Various thresholding method
        # img = cv2.threshold(cv2.bilateralFilter(img, 5, 75, 75), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        # img = cv2.threshold(cv2.medianBlur(img, 3), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        # img = cv2.adaptiveThreshold(cv2.GaussianBlur(img, (5, 5), 0), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        #                             cv2.THRESH_BINARY, 31, 2)
        img = cv2.adaptiveThreshold(cv2.bilateralFilter(img, 9, 75, 75), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY, 31, 2)
        # img = cv2.adaptiveThreshold(cv2.medianBlur(img, 3), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31,
        #                             2)
        return img

    @staticmethod
    def noise_removal(img):
        """
        Noise reduction

        :param img:
        :return:
        """
        kernel = np.ones((1, 1), np.uint8)
        img = cv2.dilate(img, kernel, iterations=1)
        img = cv2.erode(img, kernel, iterations=1)
        return img

    @staticmethod
    def determine_score(arr, angle):
        data = inter.rotate(arr, angle, reshape=False, order=0)
        histogram = np.sum(data, axis=1)
        score = np.sum((histogram[1:] - histogram[:-1]) ** 2)
        return histogram, score

    @staticmethod
    def correct_skew(image, delta=1, limit=5):
        """

        :param delta:
        :param limit:
        :return:
        """

        # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

        scores = []
        angles = np.arange(-limit, limit + delta, delta)
        for angle in angles:
            histogram, score = PreProcessing.determine_score(thresh, angle)
            scores.append(score)

        best_angle = angles[scores.index(max(scores))]

        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, best_angle, 1.0)
        rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        print("The Rotation angle is: ", best_angle)
        return rotated

    @staticmethod
    def run(img):
        preprocessor = PreProcessing()
        # remove noise
        img = preprocessor.noise_removal(img)
        # resize
        img = preprocessor.imageResize(img)
        # convert to grayscale
        img = preprocessor.bgrtogrey(img)
        # threshold
        img = preprocessor.threshold(img)
        # correct skew
        img = preprocessor.correct_skew(img)

        return img


if __name__ == '__main__':
    path_to_image = "/Users/eitankassuto/PycharmProjects/OCRPipeline/test.png"
    img = Image.open(path_to_image)
    print(img.size)
    preprocessor = PreProcessing()
    img_resize = preprocessor.imageResize(np.asarray(img))
    img_resize_pil = Image.fromarray(img_resize)
    print(img_resize_pil.size)

import cv2
import numpy as np
from PIL import Image
from scipy.ndimage import interpolation as inter
import matplotlib.pyplot as plt

class PreProcessing:
    def __init__(self, image, config):
        self.img = image
        self.original_width, self.original_height = self.img.shape[:2]
        self.config = config
        print("original Width x Height of image is {width} x {height}".format(width=self.original_width,
                                                                              height=self.original_height))
    @staticmethod
    def image_resize(image, params=None):
        """
        Scaling up of image

        :return:
        """
        params = {} if params is None else params

        img_resized = cv2.resize(image, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)  # Inter Cubic
        return img_resized

    @staticmethod
    def bgr_to_grey(image, params=None):
        """
        BGR to GRAY

        :return:
        """
        params = {} if params is None else params
        img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return img

    @staticmethod
    def threshold(image, params=None):
        """
        Apply threshold to image to limit to 255
        :param image:
        :param params:
        :return:
        """
        params = {} if params is None else params
        img = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        return img

    @staticmethod
    def noise_removal(image, params=None):
        """
        Remove noise from image
        :param img:
        :return:
        """
        params = {} if params is None else params
        img_clean = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 15)
        return img_clean

    @staticmethod
    def determine_score(arr, angle):
        """
        determine score for skew correction
        :param arr:
        :param angle:
        :return:
        """
        data = inter.rotate(arr, angle, reshape=False, order=0)
        histogram = np.sum(data, axis=1)
        score = np.sum((histogram[1:] - histogram[:-1]) ** 2)
        return histogram, score

    @staticmethod
    def correct_skew(image, params=None):
        """
        Correct skew of image
        :param delta:
        :param limit:
        :return:
        """
        params = {} if params is None else params
        limit = params.get('limit')
        delta = params.get('delta')

        scores = []
        angles = np.arange(-limit, limit + delta, delta)
        for angle in angles:
            histogram, score = PreProcessing.determine_score(image, angle)
            scores.append(score)

        best_angle = angles[scores.index(max(scores))]

        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, best_angle, 1.0)
        rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        print("The Rotation angle is: ", best_angle)
        return rotated

    def run(self):
        """
        Run function for preprocessor
        :return:
        """
        img_clean = self.img
        if self.config['preprocessors'] is None:
            return self.img
        for preprocessor in self.config['preprocessors']:
            preprocessor_name = preprocessor['name']
            preprocessor_params = preprocessor['params'] if len(preprocessor['params']) > 0 else None
            img_clean = getattr(PreProcessing, preprocessor_name)(img_clean, preprocessor_params)
            # plt.imshow(img_clean)
            # plt.savefig('preprocess_{}.png'.format(preprocessor_name))

        return img_clean


if __name__ == '__main__':
    path_to_image = "/Users/eitankassuto/PycharmProjects/OCRPipeline/test.png"
    img = Image.open(path_to_image)
    print(img.size)
    preprocessor = PreProcessing()
    img_resize = preprocessor.image_resize(np.asarray(img))
    img_resize_pil = Image.fromarray(img_resize)
    print(img_resize_pil.size)

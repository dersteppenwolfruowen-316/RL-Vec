"""图像预处理模块。

提供输入图像的预处理操作，包括缩放、降噪、对比度增强和二值化。
"""
from typing import Tuple, Optional, Dict, Any
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import cv2


class ImagePreprocessor:
    def __init__(
        self,
        target_size: Tuple[int, int] = (512, 512),
        normalize: bool = True,
        remove_noise: bool = True,
        enhance_contrast: bool = True,
        binarize: bool = False,
        threshold: int = 127,
    ):
        self.target_size = target_size
        self.normalize = normalize
        self.remove_noise = remove_noise
        self.enhance_contrast = enhance_contrast
        self.binarize = binarize
        self.threshold = threshold

    def preprocess(self, image: Image.Image) -> Image.Image:
        if not isinstance(image, Image.Image):
            if isinstance(image, np.ndarray):
                image = Image.fromarray(image)
            else:
                raise ValueError("image must be PIL.Image or numpy.ndarray")

        image = self._resize(image)

        if self.remove_noise:
            image = self._remove_noise(image)

        if self.enhance_contrast:
            image = self._enhance_contrast(image)

        if self.binarize:
            image = self._binarize(image)

        if self.normalize:
            image = self._normalize(image)

        return image

    def _resize(self, image: Image.Image) -> Image.Image:
        if self.target_size:
            image = image.resize(self.target_size, Image.LANCZOS)
        return image

    def _remove_noise(self, image: Image.Image) -> Image.Image:
        img_array = np.array(image)

        if len(img_array.shape) == 3:
            img_gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        else:
            img_gray = img_array

        denoised = cv2.fastNlMeansDenoising(img_gray, None, 10, 7, 21)

        if len(img_array.shape) == 3:
            denoised = cv2.cvtColor(denoised, cv2.COLOR_GRAY2RGB)

        return Image.fromarray(denoised)

    def _enhance_contrast(self, image: Image.Image) -> Image.Image:
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(1.5)

        enhancer = ImageEnhance.Sharpness(image)
        image = enhancer.enhance(1.2)

        return image

    def _binarize(self, image: Image.Image) -> Image.Image:
        img_array = np.array(image)

        if len(img_array.shape) == 3:
            if img_array.shape[2] == 3:
                img_gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            else:
                img_gray = img_array[:, :, 0]
        else:
            img_gray = img_array

        _, binary = cv2.threshold(img_gray, self.threshold, 255, cv2.THRESH_BINARY_INV)

        return Image.fromarray(binary)

    def _normalize(self, image: Image.Image) -> Image.Image:
        return image

    def preprocess_array(self, image: np.ndarray) -> np.ndarray:
        pil_image = Image.fromarray(image)
        processed = self.preprocess(pil_image)
        return np.array(processed)


class TextRemovalPreprocessor:
    def __init__(
        self,
        min_text_height: int = 10,
        max_text_height: int = 100,
        text_threshold: float = 0.8,
    ):
        self.min_text_height = min_text_height
        self.max_text_height = max_text_height
        self.text_threshold = text_threshold

    def remove_text(self, image: np.ndarray) -> np.ndarray:
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image

        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        mask = np.zeros_like(gray)

        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)

            aspect_ratio = w / max(h, 1)
            extent = cv2.contourArea(contour) / (w * h) if w * h > 0 else 0

            if (self.min_text_height <= h <= self.max_text_height and
                0.1 <= aspect_ratio <= 10 and
                extent > self.text_threshold):

                cv2.drawContours(mask, [contour], -1, 255, -1)

        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        result = cv2.inpaint(image, mask, 3, cv2.INPAINT_TELEA)

        return result

    def detect_text_regions(self, image: np.ndarray) -> list:
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image

        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        text_regions = []

        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)

            aspect_ratio = w / max(h, 1)
            extent = cv2.contourArea(contour) / (w * h) if w * h > 0 else 0

            if (self.min_text_height <= h <= self.max_text_height and
                0.1 <= aspect_ratio <= 10 and
                extent < self.text_threshold):  # text has low extent (strokes), not solid fills

                text_regions.append({
                    "x": x,
                    "y": y,
                    "width": w,
                    "height": h,
                    "aspect_ratio": aspect_ratio,
                    "extent": extent,
                })

        return text_regions


class DrawingEnhancer:
    def __init__(
        self,
        line_width: int = 2,
        dilation_size: int = 1,
        morphology_iterations: int = 1,
    ):
        self.line_width = line_width
        self.dilation_size = dilation_size
        self.morphology_iterations = morphology_iterations

    def enhance(self, image: np.ndarray) -> np.ndarray:
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image

        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (self.line_width, self.line_width)
        )
        dilated = cv2.dilate(binary, kernel, iterations=self.morphology_iterations)

        result = cv2.bitwise_not(dilated)

        if len(image.shape) == 3:
            result = cv2.cvtColor(result, cv2.COLOR_GRAY2RGB)

        return result

    def thin_lines(self, image: np.ndarray) -> np.ndarray:
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image

        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        skeleton = np.zeros_like(binary)
        element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))

        done = False
        while not done:
            eroded = cv2.erode(binary, element)
            temp = cv2.dilate(eroded, element)
            temp = cv2.subtract(binary, temp)
            skeleton = cv2.bitwise_or(skeleton, temp)
            binary = eroded.copy()

            done = cv2.countNonZero(binary) == 0

        result = cv2.bitwise_not(skeleton)

        if len(image.shape) == 3:
            result = cv2.cvtColor(result, cv2.COLOR_GRAY2RGB)

        return result


def create_preprocessing_pipeline(
    steps: list,
    config: Optional[Dict[str, Any]] = None
):
    config = config or {}

    preprocessors = []

    for step in steps:
        if step == "resize":
            preprocessors.append(
                ImagePreprocessor(target_size=config.get("target_size", (512, 512))))
        elif step == "denoise":
            preprocessors.append(
                ImagePreprocessor(remove_noise=True))
        elif step == "contrast":
            preprocessors.append(
                ImagePreprocessor(enhance_contrast=True))
        elif step == "binarize":
            preprocessors.append(
                ImagePreprocessor(binarize=True, threshold=config.get("threshold", 127)))
        elif step == "text_removal":
            preprocessors.append(TextRemovalPreprocessor())
        elif step == "enhance":
            preprocessors.append(DrawingEnhancer())

    def pipeline(image):
        for preprocessor in preprocessors:
            if isinstance(preprocessor, ImagePreprocessor):
                image = preprocessor.preprocess(image)
            elif isinstance(preprocessor, TextRemovalPreprocessor):
                if isinstance(image, Image.Image):
                    image = np.array(image)
                image = preprocessor.remove_text(image)
            elif isinstance(preprocessor, DrawingEnhancer):
                if isinstance(image, Image.Image):
                    image = np.array(image)
                image = preprocessor.enhance(image)
        return image

    return pipeline

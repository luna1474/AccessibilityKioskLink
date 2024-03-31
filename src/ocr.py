import cv2
import pytesseract
import os
import numpy as np
from PIL import Image, ImageFilter
import tempfile
import re
IMAGE_SIZE = 1800
BINARY_THREHOLD = 180


class ocr():
    def __init__(self):
        # OCR 엔진에 한글 언어를 설정합니다.
        pytesseract.pytesseract.tesseract_cmd = '/usr/bin/tesseract'
        self.custom_config = r'--oem 1 --psm 6 -l kor --tessdata-dir /root/projects/tessdata_fast'

        # 캡처된 이미지 디렉토리를 지정합니다.
        self.captured_images_directory = "./projects/"

    def process_image_for_ocr(self, file_path):
        # TODO : Implement using opencv
        temp_filename = self.set_image_dpi(file_path)
        im_new = self.remove_noise_and_smooth(temp_filename)
        return im_new

    def set_image_dpi(self, file_path):
        im = Image.open(file_path)
        length_x, width_y = im.size
        factor = max(1, int(IMAGE_SIZE / length_x))
        size = factor * length_x, factor * width_y
        # size = (1800, 1800)
        im_resized = im.resize(size, Image.BILINEAR)
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
        temp_filename = temp_file.name
        im_resized.save(temp_filename, dpi=(300, 300))
        return temp_filename

    def image_smoothening(self, img):
        ret1, th1 = cv2.threshold(img, BINARY_THREHOLD, 255, cv2.THRESH_BINARY)
        ret2, th2 = cv2.threshold(th1, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        blur = cv2.GaussianBlur(th2, (1, 1), 0)
        ret3, th3 = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return th3

    def remove_noise_and_smooth(self, file_name):
        img = cv2.imread(file_name, 0)
        filtered = cv2.adaptiveThreshold(img.astype(np.uint8), 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 41,
                                        3)
        kernel = np.ones((1, 1), np.uint8)
        opening = cv2.morphologyEx(filtered, cv2.MORPH_OPEN, kernel)
        closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)
        img = self.image_smoothening(img)
        or_image = cv2.bitwise_or(img, closing)
        return or_image

    # 이미지 전처리 함수
    def preprocess_image2(self, image):
        # Get local maximum:
        kernelSize = 5
        maxKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernelSize, kernelSize))
        localMax = cv2.morphologyEx(image, cv2.MORPH_CLOSE, maxKernel, None, None, 1, cv2.BORDER_REFLECT101)

        # Perform gain division
        gainDivision = np.where(localMax == 0, 0, (image/localMax))

        # Clip the values to [0,255]
        gainDivision = np.clip((255 * gainDivision), 0, 255)

        # Convert the mat type from float to uint8:
        gainDivision = gainDivision.astype("uint8")
        return gainDivision
        # Convert RGB to grayscale:
        grayscaleImage = cv2.cvtColor(gainDivision, cv2.COLOR_BGR2GRAY)
        
        # Get binary image via Otsu:
        _, binaryImage = cv2.threshold(grayscaleImage, 100, 200, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Set kernel (structuring element) size:
        kernelSize = 3
        # Set morph operation iterations:
        opIterations = 1

        # Get the structuring element:
        morphKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernelSize, kernelSize))

        # Perform closing:
        binaryImage = cv2.morphologyEx( binaryImage, cv2.MORPH_CLOSE, morphKernel, None, None, opIterations, cv2.BORDER_REFLECT101 )
        
        # Flood fill (white + black):
        cv2.floodFill(binaryImage, mask=None, seedPoint=(int(0), int(0)), newVal=(255))

        # Invert image so target blobs are colored in white:
        binaryImage = 255 - binaryImage

        # Find the blobs on the binary image:
        contours, hierarchy = cv2.findContours(binaryImage, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # Process the contours:
        for i, c in enumerate(contours):

            # Get contour hierarchy:
            currentHierarchy = hierarchy[0][i][3]

            # Look only for children contours (the holes):
            if currentHierarchy != -1:

                # Get the contour bounding rectangle:
                boundRect = cv2.boundingRect(c)

                # Get the dimensions of the bounding rect:
                rectX = boundRect[0]
                rectY = boundRect[1]
                rectWidth = boundRect[2]
                rectHeight = boundRect[3]

                # Get the center of the contour the will act as
                # seed point to the Flood-Filling:
                fx = rectX + 0.5 * rectWidth
                fy = rectY + 0.5 * rectHeight

                # Fill the hole:
                cv2.floodFill(binaryImage, mask=None, seedPoint=(int(fx), int(fy)), newVal=(0))
        return binaryImage

    def filter_text(self, text):
        # 숫자와 특수 기호를 제외한 문자만 남깁니다.
        filtered_text = re.sub(r'[^가-힣\s]', '', text)
        return filtered_text

    def process_ocr(self, image, confidence: int = 60, proximity_threshold_x: int = 100, proximity_threshold_y: int = 30):
        # 이미지 파일을 읽어옵니다.
        preprocessed_image = self.preprocess_image2(image)
        cv2.imwrite("t2.jpg", preprocessed_image)
        preprocessed_image = self.process_image_for_ocr("t2.jpg")
        cv2.imwrite("t.jpg", preprocessed_image)

        # 이미지에서 텍스트를 추출합니다.
        data = pytesseract.image_to_data(preprocessed_image, config=self.custom_config, output_type=pytesseract.Output.DICT)

        # 각 단어의 위치 정보를 추출합니다.
        previous_x, previous_y = -1, -1  # 이전 단어의 좌표를 저장하기 위한 변수
        merged_word = ''  # 병합된 단어를 저장하기 위한 변수
        ocr_results = []
        for i, word in enumerate(data['text']):
            # confidence가 일정 수준 이상인 단어만 고려합니다.
            if int(data['conf'][i]) > confidence:
                x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]

                # 이전 단어와의 좌표 차이를 계산합니다.
                if previous_x != -1 and previous_y != -1:
                    x_diff = abs(x - previous_x)
                    y_diff = abs(y - previous_y)

                    # 좌표 차이가 15 이하인 경우, 단어를 병합합니다.
                    if x_diff <= proximity_threshold_x and y_diff <= proximity_threshold_y:
                        merged_word += ' ' + word
                        continue

                # 병합된 단어를 저장하고, 다음 단어를 처리합니다.
                if merged_word:
                    ocr_results.append([merged_word.strip(), previous_x, previous_y])
                    merged_word = ''

                # 이전 단어의 좌표를 업데이트합니다.
                previous_x, previous_y = x, y

                # 현재 단어를 추가합니다.
                ocr_results.append([word, x, y])

        return ocr_results
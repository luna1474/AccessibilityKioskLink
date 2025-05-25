import cv2
import pytesseract
import os # For example usage if uncommented
import numpy as np
# from PIL import Image # No longer needed as direct input, OpenCV handles image reading if needed.
import re

# --- Module Constants ---
IMAGE_RESIZE_TARGET_WIDTH = 1800
"""OCR 전처리 시 이미지 크기 조정을 위한 목표 너비입니다. 이 값보다 작으면 확대하지 않습니다."""

DEFAULT_BINARY_THRESHOLD = 180
"""이미지 스무딩 과정에서 사용되는 기본 이진화 임계값입니다."""

DEFAULT_OCR_CONFIDENCE_THRESHOLD = 60
"""OCR 결과에서 유효하다고 판단하는 최소 신뢰도 값입니다."""

DEFAULT_PROXIMITY_THRESHOLD_X = 100
"""OCR 결과 병합 시 x축 근접성 임계값입니다. (단어의 시작점 기준)"""

DEFAULT_PROXIMITY_THRESHOLD_Y = 30
"""OCR 결과 병합 시 y축 근접성 임계값입니다. (단어의 시작점 기준)"""


class ocr:
    """
    OCR (Optical Character Recognition) 처리를 위한 클래스입니다.
    이미지 전처리, Tesseract OCR 엔진을 사용한 텍스트 추출, 결과 후처리 기능을 제공합니다.
    모든 이미지 처리는 인메모리(NumPy 배열) 방식으로 수행됩니다.
    """

    def __init__(self):
        """
        ocr 클래스 초기화 메서드입니다.
        Tesseract 실행 경로와 OCR에 사용될 기본 설정을 정의합니다.
        """
        # Tesseract OCR 엔진 실행 파일의 경로를 지정합니다.
        # 시스템 환경에 따라 이 경로는 달라질 수 있습니다 (예: '/usr/local/bin/tesseract').
        pytesseract.pytesseract.tesseract_cmd = '/usr/bin/tesseract'
        
        # Tesseract OCR 엔진의 사용자 정의 설정입니다:
        # --oem 1: OCR 엔진 모드 설정 (1 = LSTM만 사용, 더 정확할 수 있으나 기존 엔진보다 느릴 수 있음).
        # --psm 6: 페이지 분할 모드 설정 (6 = 이미지를 단일 균일 텍스트 블록으로 간주).
        # -l kor: OCR 언어를 한국어('kor')로 설정합니다.
        # --tessdata-dir ... : 사용자 정의 학습 데이터가 있는 디렉토리 경로를 지정합니다.
        #                        기본 tessdata 경로 외에 추가적인 학습 데이터를 사용하고자 할 때 필요합니다.
        self.custom_config = r'--oem 1 --psm 6 -l kor --tessdata-dir /root/AccessibilityKioskLink/src/tessdata_fast'

    def _resize_image(self, image: np.ndarray, target_width: int = IMAGE_RESIZE_TARGET_WIDTH) -> np.ndarray | None:
        """
        OpenCV 이미지를 지정된 목표 너비로 비율을 유지하며 크기를 조정합니다.
        너비가 target_width보다 작으면 원본 크기를 유지합니다 (확대하지 않음).

        Args:
            image (np.ndarray): 크기를 조정할 OpenCV 이미지 (NumPy 배열).
            target_width (int): 목표 너비.

        Returns:
            np.ndarray | None: 크기가 조정된 OpenCV 이미지.
                               입력 이미지가 유효하지 않거나 오류 발생 시 None을 반환합니다.
        """
        if image is None:
            print("OCR Error (_resize_image): 입력 이미지가 None입니다.")
            return None
        
        try:
            height, width = image.shape[:2]
            if width == 0:
                print(f"OCR Error (_resize_image): 이미지 너비가 0입니다.")
                return None

            if width <= target_width: # 너비가 목표보다 작거나 같으면 원본 유지
                return image.copy() # 원본 수정을 방지하기 위해 복사본 반환

            factor = target_width / width # 축소 비율
            new_width = target_width
            new_height = int(height * factor)
            
            # 축소 시 INTER_AREA가 일반적으로 권장됩니다.
            resized_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
            return resized_image
        except Exception as e:
            print(f"OCR Error (_resize_image): 이미지 리사이징 중 오류 발생: {e}")
            return None

    def _apply_image_smoothening(self, image: np.ndarray, binary_threshold_value: int = DEFAULT_BINARY_THRESHOLD) -> np.ndarray | None:
        """
        OpenCV 그레이스케일 이미지에 스무딩 처리를 적용하여 노이즈를 줄이고 대비를 향상시킵니다.

        Args:
            image (np.ndarray): 스무딩 처리를 적용할 OpenCV 그레이스케일 이미지.
            binary_threshold_value (int): 초기 이진화에 사용될 임계값.

        Returns:
            np.ndarray | None: 스무딩 처리된 OpenCV 이미지. 오류 발생 시 None을 반환합니다.
        """
        if image is None:
            print("OCR Error (_apply_image_smoothening): 입력 이미지가 None입니다.")
            return None
        if len(image.shape) != 2: # 그레이스케일 이미지만 처리
            print("OCR Error (_apply_image_smoothening): 입력 이미지는 그레이스케일이어야 합니다.")
            return None
            
        try:
            # 1. 고정 임계값을 사용한 이진화
            _, th1 = cv2.threshold(image, binary_threshold_value, 255, cv2.THRESH_BINARY)
            # 2. Otsu의 이진화 적용 (th1 결과에 적용)
            _, th2 = cv2.threshold(th1, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            # 3. 가우시안 블러 적용 (노이즈 감소)
            blur = cv2.GaussianBlur(th2, (1, 1), 0) # (1,1)은 매우 약한 블러링
            # 4. 다시 Otsu의 이진화 적용 (블러링된 이미지에 적용하여 최종 결과 개선)
            _, th3 = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            return th3
        except Exception as e:
            print(f"OCR Error (_apply_image_smoothening): 이미지 스무딩 중 오류 발생: {e}")
            return None

    def _remove_noise_and_apply_smoothing(self, image: np.ndarray) -> np.ndarray | None:
        """
        그레이스케일 OpenCV 이미지에 적응형 임계값 처리, 모폴로지 연산 및 추가 스무딩을 적용합니다.

        Args:
            image (np.ndarray): 처리할 OpenCV 그레이스케일 이미지.

        Returns:
            np.ndarray | None: 처리된 OpenCV 이미지. 오류 발생 시 None을 반환합니다.
        """
        if image is None:
            print("OCR Error (_remove_noise_and_apply_smoothing): 입력 이미지가 None입니다.")
            return None
        if len(image.shape) != 2: # 그레이스케일 이미지만 처리
            print("OCR Error (_remove_noise_and_apply_smoothing): 입력 이미지는 그레이스케일이어야 합니다.")
            return None
            
        try:
            # 입력 이미지가 uint8 타입인지 확인 및 변환 (필요시)
            image_uint8 = image.astype(np.uint8) if image.dtype != np.uint8 else image

            # 적응형 임계값(Adaptive Thresholding) 적용
            filtered_img = cv2.adaptiveThreshold(image_uint8, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 41, 3)
            
            # 모폴로지(Morphology) 연산
            kernel = np.ones((1, 1), np.uint8) # 1x1 커널은 최소한의 연산
            opening_img = cv2.morphologyEx(filtered_img, cv2.MORPH_OPEN, kernel)
            closing_img = cv2.morphologyEx(opening_img, cv2.MORPH_CLOSE, kernel)
            
            # 원본 이미지(image_uint8)에 대해 스무딩을 적용
            smoothed_original_img = self._apply_image_smoothening(image_uint8)
            if smoothed_original_img is None:
                print("OCR Error (_remove_noise_and_apply_smoothing): 내부 스무딩 호출 실패.")
                return None

            # 스무딩된 원본과 모폴로지 처리된 이미지를 비트 OR 연산으로 결합
            or_image = cv2.bitwise_or(smoothed_original_img, closing_img)
            return or_image
        except Exception as e:
            print(f"OCR Error (_remove_noise_and_apply_smoothing): 노이즈 제거 및 스무딩 중 오류: {e}")
            return None

    def _alternative_image_preprocessing(self, image: np.ndarray) -> np.ndarray | None:
        """
        [이전 preprocess_image2 함수]
        이미지에 대해 국부적 명암 보정, Otsu 이진화, 모폴로지 연산 등을 포함한 대체 전처리 방법을 적용합니다.
        이 함수는 입력 이미지를 그레이스케일로 변환하여 처리하며,
        최종적으로 텍스트는 흰색(255), 배경은 검은색(0)인 이진 이미지를 반환합니다.

        Args:
            image (np.ndarray): 입력 OpenCV 이미지 (컬러 또는 그레이스케일).

        Returns:
            np.ndarray | None: 전처리된 이진 OpenCV 이미지 (텍스트는 흰색, 배경은 검은색). 오류 발생 시 None.
        """
        if image is None:
            print("OCR Error (_alternative_image_preprocessing): 입력 이미지가 None입니다.")
            return None

        try:
            # 단계 1: 그레이스케일 변환
            if len(image.shape) == 3 and image.shape[2] == 3:
                 gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            elif len(image.shape) == 2:
                 gray_image = image.copy()
            else:
                print(f"OCR Error (_alternative_image_preprocessing): 지원하지 않는 이미지 형태입니다. Shape: {image.shape}")
                return None

            # 단계 2: 국부적 명암 보정 (Local Contrast Enhancement using Gain Division)
            kernel_size_local_max = 5
            max_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size_local_max, kernel_size_local_max))
            local_max_approx = cv2.morphologyEx(gray_image, cv2.MORPH_CLOSE, max_kernel, iterations=1)
            local_max_approx_safe = np.where(local_max_approx == 0, 1, local_max_approx).astype(float)
            gain_division_float = gray_image.astype(float) / local_max_approx_safe
            if np.max(gain_division_float) > 0:
                gain_division_normalized = np.clip((255 * (gain_division_float / np.max(gain_division_float))), 0, 255)
            else:
                gain_division_normalized = np.zeros_like(gain_division_float)
            processed_gain_img = gain_division_normalized.astype("uint8")
            
            # 단계 3: Otsu 이진화 (텍스트 검은색, 배경 흰색)
            _, binary_img_otsu_inv = cv2.threshold(processed_gain_img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            
            # 단계 4: 이미지 반전 (텍스트 흰색, 배경 검은색)
            binary_img_text_white = cv2.bitwise_not(binary_img_otsu_inv)

            # 단계 5: 모폴로지 닫힘 연산
            kernel_size_morph_closing = 3
            morph_kernel_closing = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size_morph_closing, kernel_size_morph_closing))
            closed_img = cv2.morphologyEx(binary_img_text_white, cv2.MORPH_CLOSE, morph_kernel_closing, iterations=1)
            
            # 단계 6: 홀 채우기 (Flood Fill based)
            img_for_filling = closed_img.copy()
            h_fill, w_fill = img_for_filling.shape[:2]
            fill_mask = np.zeros((h_fill + 2, w_fill + 2), np.uint8)
            if img_for_filling[0,0] == 0: 
                cv2.floodFill(img_for_filling, fill_mask, (0,0), 128)
            img_for_filling[img_for_filling == 0] = 255
            img_for_filling[img_for_filling == 128] = 0
            
            return img_for_filling
        except Exception as e:
            print(f"OCR Error (_alternative_image_preprocessing): 대체 전처리 중 오류 발생: {e}")
            return None

    def _perform_ocr_extraction(self, image: np.ndarray) -> dict:
        """
        전처리된 OpenCV 이미지에 대해 Pytesseract OCR을 수행하여 텍스트 데이터를 추출합니다.

        Args:
            image (np.ndarray): OCR을 수행할 전처리된 OpenCV 이미지.

        Returns:
            dict: Pytesseract의 image_to_data 결과 (딕셔너리 형태). 오류 시 빈 딕셔너리.
        """
        if image is None:
            print("OCR Error (_perform_ocr_extraction): 입력 이미지가 None입니다.")
            return {}
        try:
            ocr_raw_data = pytesseract.image_to_data(image, config=self.custom_config, output_type=pytesseract.Output.DICT)
            return ocr_raw_data
        except pytesseract.TesseractError as e:
            print(f"OCR Error (_perform_ocr_extraction): Tesseract 실행 오류: {e}")
            return {}
        except Exception as e:
            print(f"OCR Error (_perform_ocr_extraction): Pytesseract 호출 중 예외 발생: {e}")
            return {}

    def _parse_and_merge_text_results(self,
                                      ocr_data: dict,
                                      confidence_thresh: int = DEFAULT_OCR_CONFIDENCE_THRESHOLD,
                                      prox_thresh_x: int = DEFAULT_PROXIMITY_THRESHOLD_X,
                                      prox_thresh_y: int = DEFAULT_PROXIMITY_THRESHOLD_Y) -> list:
        """
        Pytesseract OCR 결과(data 딕셔너리)를 파싱하고, 설정된 신뢰도 및
        근접성 임계값을 기준으로 단어들을 병합하여 최종 텍스트 목록을 생성합니다.

        Args:
            ocr_data (dict): Pytesseract의 image_to_data 결과.
            confidence_thresh (int): 유효한 단어로 간주할 최소 신뢰도 점수.
            prox_thresh_x (int): 단어 병합을 위한 X축 최대 거리 (단어 시작점 기준).
            prox_thresh_y (int): 단어 병합을 위한 Y축 최대 거리 (단어 시작점 기준).

        Returns:
            list: 병합된 OCR 결과. 각 요소는 [텍스트, x좌표, y좌표] 리스트 형태입니다.
                  x, y 좌표는 병합된 텍스트 블록의 첫 단어의 시작점을 나타냅니다.
        """
        parsed_results = []
        if not ocr_data or 'text' not in ocr_data or not ocr_data['text']:
            return parsed_results

        num_items = len(ocr_data['text'])
        current_merged_text = ""
        first_word_x_in_block, first_word_y_in_block = -1, -1 
        last_processed_word_x, last_processed_word_y = -1, -1

        try:
            for i in range(num_items):
                # 필수 키 존재 및 인덱스 유효성 검사
                if not all(key in ocr_data and i < len(ocr_data[key]) for key in ['conf', 'left', 'top', 'text']):
                    continue

                word_text = ocr_data['text'][i].strip()
                try:
                    confidence_score = int(float(ocr_data['conf'][i]))
                except ValueError:
                    continue 
                
                if not word_text or confidence_score < confidence_thresh:
                    if current_merged_text:
                        parsed_results.append([current_merged_text, first_word_x_in_block, first_word_y_in_block])
                        current_merged_text = ""
                        first_word_x_in_block, first_word_y_in_block = -1, -1
                    last_processed_word_x, last_processed_word_y = -1,-1
                    continue

                current_word_x, current_word_y = ocr_data['left'][i], ocr_data['top'][i]

                if not current_merged_text: # 새 블록 시작
                    current_merged_text = word_text
                    first_word_x_in_block, first_word_y_in_block = current_word_x, current_word_y
                else: # 기존 블록에 병합 시도
                    y_difference = abs(current_word_y - last_processed_word_y)
                    # X 근접성은 이전 단어의 시작점(last_processed_word_x)과 현재 단어 시작점(current_word_x) 간의 거리로 판단
                    x_difference = abs(current_word_x - last_processed_word_x) 
                    
                    if y_difference <= prox_thresh_y and x_difference <= prox_thresh_x :
                        current_merged_text += " " + word_text
                    else: 
                        parsed_results.append([current_merged_text, first_word_x_in_block, first_word_y_in_block])
                        current_merged_text = word_text
                        first_word_x_in_block, first_word_y_in_block = current_word_x, current_word_y
                
                last_processed_word_x, last_processed_word_y = current_word_x, current_word_y
            
            if current_merged_text: # 루프 후 남은 병합 텍스트 추가
                parsed_results.append([current_merged_text, first_word_x_in_block, first_word_y_in_block])

        except (KeyError, IndexError) as e:
            print(f"OCR Error (_parse_and_merge_text_results): OCR 데이터 구조 접근 중 오류: {e}")
        except Exception as e:
            print(f"OCR Error (_parse_and_merge_text_results): 결과 파싱 중 예기치 않은 오류: {e}")
        
        return parsed_results

    def _orchestrate_image_preprocessing(self, raw_cv_image: np.ndarray) -> np.ndarray | None:
        """
        OCR을 위한 전체 이미지 전처리 과정을 조율합니다.
        입력된 원본 OpenCV 이미지를 받아 여러 단계의 전처리 과정을 거친 후
        OCR에 적합한 형태로 변환된 이미지를 반환합니다.

        Args:
            raw_cv_image (np.ndarray): 원본 OpenCV 이미지 (BGR 컬러 또는 그레이스케일).

        Returns:
            np.ndarray | None: 최종 전처리된 OpenCV 이미지 (그레이스케일, 이진화됨).
                               오류 발생 시 None을 반환합니다.
        """
        if raw_cv_image is None:
            print("OCR Error (_orchestrate_image_preprocessing): 입력 이미지가 None입니다.")
            return None

        # 단계 1: 대체 전처리 (_alternative_image_preprocessing)
        # 이 함수는 명암 보정, Otsu 이진화 (텍스트 흰색, 배경 검은색으로 결과 반환) 등을 수행합니다.
        # 이 단계에서 이미지가 그레이스케일로 변환됩니다 (내부적으로).
        preprocessed_stage1 = self._alternative_image_preprocessing(raw_cv_image)
        if preprocessed_stage1 is None:
            print("OCR Error (_orchestrate_image_preprocessing): 1단계 전처리 실패.")
            return None
        
        # 단계 2: 이미지 리사이징
        # preprocessed_stage1은 그레이스케일 이미지입니다.
        resized_image = self._resize_image(preprocessed_stage1, target_width=IMAGE_RESIZE_TARGET_WIDTH)
        if resized_image is None:
            print("OCR Error (_orchestrate_image_preprocessing): 이미지 리사이징 실패.")
            return None

        # 단계 3: 추가 노이즈 제거 및 스무딩
        # resized_image는 그레이스케일입니다.
        final_processed_image = self._remove_noise_and_apply_smoothing(resized_image)
        if final_processed_image is None:
            print("OCR Error (_orchestrate_image_preprocessing): 최종 노이즈 제거 및 스무딩 단계 실패.")
            return None
            
        return final_processed_image

    def process_ocr(self,
                    cv_image: np.ndarray,
                    confidence: int = DEFAULT_OCR_CONFIDENCE_THRESHOLD,
                    proximity_x: int = DEFAULT_PROXIMITY_THRESHOLD_X,
                    proximity_y: int = DEFAULT_PROXIMITY_THRESHOLD_Y) -> list:
        """
        주어진 OpenCV 이미지에 대해 전체 OCR 프로세스를 수행합니다.
        이미지 전처리, Tesseract OCR 실행, 결과 파싱 및 단어 병합을 포함합니다.

        Args:
            cv_image (np.ndarray): OCR을 수행할 원본 OpenCV 이미지 (BGR 또는 그레이스케일).
            confidence (int): OCR 결과에서 사용할 최소 신뢰도.
            proximity_x (int): 단어 병합 시 X축 근접성 임계값.
            proximity_y (int): 단어 병합 시 Y축 근접성 임계값.

        Returns:
            list: 최종 OCR 결과. 각 요소는 [텍스트, x좌표, y좌표] 형태의 리스트입니다.
                  오류 발생 또는 텍스트 미검출 시 빈 리스트를 반환합니다.
        """
        if cv_image is None:
            print("OCR Error (process_ocr): 입력 이미지가 None입니다.")
            return []

        # 1. 이미지 전처리 오케스트레이션
        preprocessed_cv_image = self._orchestrate_image_preprocessing(cv_image)
        if preprocessed_cv_image is None:
            print("OCR Error (process_ocr): 이미지 전처리 파이프라인 실패.")
            return []
        
        # 2. Tesseract OCR 실행
        ocr_raw_data = self._perform_ocr_extraction(preprocessed_cv_image)
        if not ocr_raw_data:
            print("OCR Warning (process_ocr): OCR 엔진이 텍스트 데이터를 추출하지 못했습니다.")
            return []

        # 3. OCR 결과 파싱 및 병합
        ocr_final_results = self._parse_and_merge_text_results(
            ocr_raw_data,
            confidence_thresh=confidence,
            prox_thresh_x=proximity_x,
            prox_thresh_y=proximity_y
        )
        
        return ocr_final_results

    def filter_text_korean_only(self, text: str) -> str:
        """
        입력된 텍스트에서 한글 문자 및 공백을 제외한 모든 문자(숫자, 영어, 특수 기호 등)를 제거합니다.

        Args:
            text (str): 필터링할 원본 텍스트.

        Returns:
            str: 한글 및 공백만 남은 필터링된 텍스트. 
                 입력이 문자열이 아니거나 오류 발생 시 빈 문자열을 반환합니다.
        """
        if not isinstance(text, str):
            print("OCR Warning (filter_text_korean_only): 입력값이 문자열이 아닙니다. 빈 문자열을 반환합니다.")
            return ""
        try:
            # 정규 표현식: [^가-힣\s]
            #   [^...] : 괄호 안의 문자를 제외한 모든 문자와 매치됩니다.
            #   가-힣 : 유니코드 한글 음절 범위 (U+AC00 ~ U+D7A3).
            #   \s    : 공백 문자 (스페이스, 탭, 줄바꿈 등).
            # 즉, '한글 또는 공백이 아닌 모든 문자'를 찾아 ''(빈 문자열)로 대체(제거)합니다.
            filtered_text = re.sub(r'[^가-힣\s]', '', text)
            return filtered_text
        except Exception as e:
            print(f"OCR Error (filter_text_korean_only): 텍스트 필터링 중 오류 발생: {e}")
            return ""

# --- Example Usage (for testing purposes) ---
if __name__ == '__main__':
    ocr_instance = ocr()
    sample_text_for_filtering = "Hello OCR! 안녕하세요 123. 필터링 테스트 중입니다."
    filtered_sample_text = ocr_instance.filter_text_korean_only(sample_text_for_filtering)
    print(f"원본 텍스트: \"{sample_text_for_filtering}\"")
    print(f"필터링된 텍스트 (한글/공백만): \"{filtered_sample_text}\"")
    
    # 이미지 처리 테스트 예제 (실제 이미지 파일 필요, 경로 수정 필요)
    # test_image_file_path = "path_to_your_test_image.png" 
    # if os.path.exists(test_image_file_path):
    #     input_cv_image = cv2.imread(test_image_file_path)
    #     if input_cv_image is not None:
    #         print(f"\n'{test_image_file_path}' 이미지 로드 성공. OCR 처리 시작...")
    #         detected_ocr_results = ocr_instance.process_ocr(input_cv_image)
    #         if detected_ocr_results:
    #             print("\n--- 감지된 텍스트 목록 ---")
    #             for text_item_detail in detected_ocr_results:
    #                 # 결과에 필터링 적용 예시
    #                 # clean_text = ocr_instance.filter_text_korean_only(text_item_detail[0])
    #                 # print(f"  텍스트: \"{clean_text}\", 위치: (x={text_item_detail[1]}, y={text_item_detail[2]})")
    #                 print(f"  텍스트: \"{text_item_detail[0]}\", 위치: (x={text_item_detail[1]}, y={text_item_detail[2]})")
    #         else:
    #             print("이미지에서 텍스트를 감지하지 못했습니다.")
    #     else:
    #         print(f"오류: '{test_image_file_path}' 이미지를 읽을 수 없습니다.")
    # else:
    #     print(f"테스트 오류: '{test_image_file_path}' 파일을 찾을 수 없습니다. 이미지 OCR 테스트를 실행하려면 유효한 경로를 설정하세요.")
    
    print("\nocr.py 모듈 로드 완료. filter_text_korean_only 테스트 실행됨. 이미지 OCR 테스트는 주석 처리됨.")

# Ensure old functions are removed if they were part of the original file structure
# For example, if 'process_image_for_ocr', 'set_image_dpi', 'image_smoothening', 
# 'remove_noise_and_smooth' were top-level functions or methods of this class
# and are now fully replaced by the private helpers and new orchestration,
# they should not be present in this final version of the file.
# The overwrite operation should handle this.
```

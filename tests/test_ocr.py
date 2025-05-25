import unittest
import sys
import os

# 테스트 대상 모듈이 있는 src 디렉토리를 Python 경로에 추가합니다.
# 이렇게 하면 'from ocr import ocr'와 같이 src 폴더 내의 모듈을 직접 임포트할 수 있습니다.
# __file__은 현재 이 test_ocr.py 파일의 경로를 나타냅니다.
# os.path.dirname(__file__)은 'tests' 디렉토리 경로입니다.
# os.path.join(os.path.dirname(__file__), '../src')는 'tests/../src', 즉 'src' 디렉토리 경로가 됩니다.
# os.path.abspath는 이 상대 경로를 절대 경로로 변환합니다.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from ocr import ocr # 'ocr.py' 모듈에서 'ocr' 클래스를 임포트합니다.

class TestOCR(unittest.TestCase):
    """
    `ocr.py` 모듈의 `ocr` 클래스에 대한 단위 테스트를 포함합니다.
    주요 테스트 대상은 텍스트 필터링 로직과 같이 격리된 순수 함수 기능입니다.
    """

    def setUp(self):
        """
        각 테스트 메서드 실행 전에 호출됩니다.
        테스트에 필요한 `ocr` 클래스의 인스턴스를 생성합니다.
        """
        self.ocr_processor = ocr() # ocr 클래스의 인스턴스 생성

    def test_filter_text_korean_only(self):
        """
        `filter_text_korean_only` 메서드가 한글 및 공백만 정확히 필터링하는지 테스트합니다.
        다양한 입력 문자열 케이스를 포함합니다:
        - 한글, 영어, 숫자, 특수문자 혼합
        - 공백 포함 (선행, 후행, 중간)
        - 특수문자만 있는 경우
        - 영어만 있는 경우
        - 빈 문자열
        - None 입력 (메서드가 이를 어떻게 처리하는지에 따라 테스트 조정 가능, 현재는 빈 문자열 반환 가정)
        """
        self.assertEqual(self.ocr_processor.filter_text_korean_only("abc한글123"), "한글", "영어/숫자 혼합 실패")
        self.assertEqual(self.ocr_processor.filter_text_korean_only("  띄 어 쓰 기  "), " 띄 어 쓰 기 ", "공백 유지 실패")
        self.assertEqual(self.ocr_processor.filter_text_korean_only("!@#$"), "", "특수문자만 있는 경우 실패")
        self.assertEqual(self.ocr_processor.filter_text_korean_only("English Only"), "", "영어만 있는 경우 실패")
        self.assertEqual(self.ocr_processor.filter_text_korean_only(""), "", "빈 문자열 입력 실패")
        self.assertEqual(self.ocr_processor.filter_text_korean_only("한글만"), "한글만", "한글만 있는 경우 실패")
        
        # filter_text_korean_only 메서드가 None 입력 시 빈 문자열을 반환하도록 수정되었으므로,
        # 해당 케이스에 대한 테스트를 추가하거나 기존 print 경고를 확인합니다.
        # 현재 구현은 None 입력 시 빈 문자열을 반환하고 경고를 출력합니다.
        self.assertEqual(self.ocr_processor.filter_text_korean_only(None), "", "None 입력 처리 실패")

    # test_parse_and_merge_text_results_basic는 지침에 따라 이 단계에서는 생략합니다.
    # def test_parse_and_merge_text_results_basic(self):
    #     pass 

if __name__ == '__main__':
    """
    이 스크립트가 직접 실행될 때 테스트를 실행합니다.
    예: python tests/test_ocr.py
    """
    unittest.main()

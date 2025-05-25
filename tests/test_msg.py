import unittest
import sys
import os

# 테스트 대상 모듈이 있는 src 디렉토리를 Python 경로에 추가합니다.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from Msg import Msg  # 'Msg.py' 모듈에서 'Msg' 클래스를 임포트합니다.
import numpy as np   # 'photo' 필드 테스트를 위해 NumPy를 임포트합니다.

class TestMsg(unittest.TestCase):
    """
    `Msg.py` 모듈의 `Msg` 데이터클래스에 대한 단위 테스트를 포함합니다.
    객체 생성 및 필드 기본값 할당의 정확성을 검증합니다.
    """

    def test_msg_creation_with_all_fields(self):
        """
        모든 필드에 값을 지정하여 `Msg` 객체를 생성하는 경우를 테스트합니다.
        각 필드가 제공된 값으로 올바르게 초기화되었는지 확인합니다.
        """
        # 테스트용 NumPy 배열 생성 (사진 데이터 예시)
        sample_photo_array = np.array([[1, 2, 3], [4, 5, 6]])
        
        msg_instance = Msg(
            msg="test_message",
            photo=sample_photo_array,
            idx=10,
            x=100,
            y=200
        )
        
        self.assertEqual(msg_instance.msg, "test_message", "메시지 내용(msg) 필드 불일치")
        # NumPy 배열 비교는 np.array_equal 사용
        self.assertTrue(np.array_equal(msg_instance.photo, sample_photo_array), "사진 데이터(photo) 필드 불일치")
        self.assertEqual(msg_instance.idx, 10, "인덱스(idx) 필드 불일치")
        self.assertEqual(msg_instance.x, 100, "x 좌표(x) 필드 불일치")
        self.assertEqual(msg_instance.y, 200, "y 좌표(y) 필드 불일치")

    def test_msg_default_creation(self):
        """
        필드 값을 지정하지 않고 `Msg` 객체를 생성하는 경우 (기본값 사용)를 테스트합니다.
        모든 필드가 예상된 기본값 (대부분 None)으로 초기화되었는지 확인합니다.
        """
        msg_instance = Msg()
        
        self.assertIsNone(msg_instance.msg, "기본 메시지 내용(msg)이 None이 아님")
        self.assertIsNone(msg_instance.photo, "기본 사진 데이터(photo)가 None이 아님")
        self.assertIsNone(msg_instance.idx, "기본 인덱스(idx)가 None이 아님")
        self.assertIsNone(msg_instance.x, "기본 x 좌표(x)가 None이 아님")
        self.assertIsNone(msg_instance.y, "기본 y 좌표(y)가 None이 아님")

    def test_msg_partial_creation(self):
        """
        일부 필드에만 값을 지정하여 `Msg` 객체를 생성하는 경우를 테스트합니다.
        지정된 필드는 해당 값으로, 미지정 필드는 기본값으로 초기화되었는지 확인합니다.
        """
        msg_instance = Msg(msg="partial_test", idx=5)
        
        self.assertEqual(msg_instance.msg, "partial_test", "부분 지정 시 메시지 내용(msg) 필드 불일치")
        self.assertIsNone(msg_instance.photo, "부분 지정 시 사진 데이터(photo)가 None이 아님")
        self.assertEqual(msg_instance.idx, 5, "부분 지정 시 인덱스(idx) 필드 불일치")
        self.assertIsNone(msg_instance.x, "부분 지정 시 x 좌표(x)가 None이 아님")
        self.assertIsNone(msg_instance.y, "부분 지정 시 y 좌표(y)가 None이 아님")

if __name__ == '__main__':
    """
    이 스크립트가 직접 실행될 때 테스트를 실행합니다.
    예: python tests/test_msg.py
    """
    unittest.main()

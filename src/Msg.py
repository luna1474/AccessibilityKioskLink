"""
이 모듈은 애플리케이션 내의 다양한 구성 요소 간에 전달되는 메시지 구조를 정의합니다.
메시지는 텍스트, 사진, 인덱스, 좌표 등 다양한 유형의 데이터를 포함할 수 있습니다.
"""
from dataclasses import dataclass, field # field를 추가하여 photo의 기본값을 None으로 명시적으로 설정
from typing import Any # photo 필드의 타입을 명시하기 위해 추가

@dataclass
class Msg:
    """
    애플리케이션의 여러 부분 간에 통신하는 데 사용되는 메시지 객체입니다.
    다양한 유형의 데이터를 캡슐화하여 구성 요소 간의 정보 교환을 표준화합니다.
    """
    msg: str = None
    """메시지의 주 내용 또는 유형을 나타내는 문자열입니다. (예: "photo", "mouse", "command")"""
    
    photo: Any = field(default=None)
    """
    사진 데이터를 담는 필드입니다. OpenCV 이미지(NumPy 배열) 또는 다른 이미지 형식이 될 수 있습니다.
    기본값은 None입니다.
    """
    
    idx: int = None
    """항목의 인덱스 또는 식별자를 나타내는 정수입니다. (예: OCR 결과 목록에서의 인덱스)"""
    
    x: int = None
    """x 좌표 값을 나타내는 정수입니다. (예: 마우스 클릭 위치의 x 좌표)"""
    
    y: int = None
    """y 좌표 값을 나타내는 정수입니다. (예: 마우스 클릭 위치의 y 좌표)"""

    # 기존 코드에 command 필드가 없었지만, 만약 다른 곳에서 사용된다면 추가할 수 있습니다.
    # 예를 들어, 특정 명령어를 전달해야 하는 경우:
    # command: str = None
    # """특정 명령어나 지시사항을 담는 문자열입니다. (예: "start_capture", "shutdown_process")"""
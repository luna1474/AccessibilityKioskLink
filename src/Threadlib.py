"""
이 모듈은 애플리케이션의 여러 부분에서 사용될 수 있는
기본 스레드 클래스 `Threadlib`을 제공합니다.
`threading.Thread`를 상속받아 로깅, 안전한 종료 매커니즘,
메인 큐 및 메시지 큐와의 상호작용을 위한 기본 구조를 포함합니다.
"""
import threading
from loglib import loglib # loglib 모듈에서 로거를 가져옵니다.

class Threadlib(threading.Thread):
    """
    애플리케이션의 다양한 기능을 백그라운드에서 실행하기 위한 기본 스레드 클래스입니다.
    Python의 `threading.Thread`를 확장하여 로깅, 이벤트 기반 종료,
    그리고 다른 스레드 또는 프로세스와의 통신을 위한 큐(Queue) 관리를 지원합니다.
    """
    def __init__(self, name: str, main_queue, message_queue):
        """
        Threadlib 인스턴스를 초기화합니다.

        Args:
            name (str): 스레드의 이름입니다. 로깅 시 식별자로 사용됩니다.
            main_queue (queue.Queue): 주 애플리케이션 로직 또는 다른 중요 컴포넌트와 통신하기 위한 큐입니다.
                                      이 스레드에서 생성된 주요 결과나 이벤트를 전달하는 데 사용될 수 있습니다.
            message_queue (queue.Queue): 이 스레드로 메시지나 작업을 전달받기 위한 큐입니다.
                                         다른 컴포넌트로부터의 명령이나 데이터를 수신하는 데 사용됩니다.
        """
        super(Threadlib, self).__init__() # 부모 클래스(threading.Thread)의 초기화 메서드 호출
        self._logger = self.get_logger() # loglib을 통해 로거 인스턴스를 가져옵니다.
        self._stop_event = threading.Event() # 스레드 종료를 위한 이벤트 객체 생성
        self._main_queue = main_queue # 메인 큐를 인스턴스 변수로 할당
        self._message_queue = message_queue # 메시지 큐를 인스턴스 변수로 할당
        self._running = True # 스레드의 실행 상태를 나타내는 플래그, True로 초기화
        self._name = name # 스레드 이름을 인스턴스 변수로 할당
        self._logger.debug(f"{self._name}: 스레드가 생성되었습니다.") # 스레드 생성 로그 기록

    def get_logger(self):
        """
        loglib를 사용하여 로거 인스턴스를 반환합니다.
        이 메서드는 로거 설정을 중앙에서 관리하고 일관된 로깅 방식을 제공하기 위해 사용될 수 있습니다.

        Returns:
            logging.Logger: 설정된 로거 인스턴스.
        """
        return loglib.get_logger() # loglib의 get_logger 함수를 호출하여 로거를 가져옵니다.

    def run(self):
        """
        스레드가 시작될 때 실행되는 주 로직입니다.
        이 메서드는 하위 클래스에서 반드시 오버라이드되어야 하며,
        스레드가 수행할 실제 작업을 정의합니다.
        기본 구현은 스레드 시작 로그만 기록합니다.
        하위 클래스에서는 `while self._running:` 루프와 같은 형태로
        `self._stop_event.is_set()`을 주기적으로 확인하여 안전하게 종료할 수 있도록 구현해야 합니다.
        """
        self._logger.debug(f"{self._name}: 스레드가 시작되었습니다.")
        # 여기에 실제 스레드 작업 로직이 위치해야 합니다 (하위 클래스에서 오버라이드).
        # 예시:
        # while self._running:
        #     if self._stop_event.is_set():
        #         break
        #     # 작업 수행
        #     time.sleep(0.1)

    def stop(self):
        """
        스레드를 안전하게 종료하도록 요청합니다.
        `_running` 플래그를 `False`로 설정하고 `_stop_event`를 설정하여,
        `run` 메서드 내의 루프가 종료될 수 있도록 신호를 보냅니다.
        스레드가 실제로 종료되는 것을 기다리려면 `join()` 메서드를 추가로 호출해야 합니다.
        """
        self._running = False # 실행 플래그를 False로 설정
        self._stop_event.set() # 종료 이벤트를 설정하여 대기 중인 작업에 알림
        self._logger.debug(f"{self._name}: 스레드 중지 요청됨.")
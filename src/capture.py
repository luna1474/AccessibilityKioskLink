"""
이 모듈은 비디오 캡처 기능을 제공합니다.
OpenCV를 사용하여 카메라 장치로부터 프레임을 지속적으로 읽어오고,
최신 프레임을 제공하는 역할을 합니다. 별도의 스레드에서 프레임 업데이트를 수행합니다.
"""
import cv2
import time
# import os # 현재 os 모듈은 사용되지 않으므로 주석 처리 또는 삭제 가능
from threading import Thread

class capture:
    """
    카메라로부터 비디오 프레임을 캡처하는 클래스입니다.
    지정된 카메라 장치를 열고, 별도의 스레드에서 지속적으로 프레임을 업데이트하여
    `take_picture()` 메서드를 통해 최신 프레임을 제공합니다.
    """
    def __init__(self, camera_index: int = 0, desired_fps: int = 30):
        """
        capture 클래스의 인스턴스를 초기화합니다.

        카메라 장치를 열고, 프레임 크기, 버퍼 크기 등을 설정합니다.
        프레임 업데이트를 위한 별도의 데몬 스레드를 시작하며, 리소스 관리를 위한 플래그를 설정합니다.

        Args:
            camera_index (int): 사용할 카메라 장치의 인덱스입니다. 기본값은 0입니다.
            desired_fps (int): 원하는 초당 프레임 수(FPS)입니다.

        Raises:
            RuntimeError: 카메라 장치를 열 수 없는 경우 발생합니다.
        """
        self._running = True  # 스레드 실행 제어 플래그
        self.is_stopped = False # stop() 메서드 중복 호출 방지 플래그

        self.cap = cv2.VideoCapture(camera_index)
        self._logger = self._get_internal_logger() # 내부 로거 사용 (print 대신)
        
        if not self.cap.isOpened():
            self._logger.error(f"카메라를 열 수 없습니다 (인덱스: {camera_index}). 연결 상태 및 권한을 확인해주세요.")
            raise RuntimeError(f"카메라를 열 수 없습니다 (인덱스: {camera_index}). 카메라 연결 상태 및 권한을 확인해주세요.")

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)

        self.FPS = 1.0 / desired_fps
        self.status = False
        self.frame = None

        self.thread = Thread(target=self.update, args=())
        self.thread.daemon = True # 메인 스레드 종료 시 자동 종료
        self.thread.start()
        self._logger.info(f"캡처 스레드 시작됨 (카메라 인덱스: {camera_index}, 목표 FPS: {desired_fps})")

    def _get_internal_logger(self):
        """간단한 내부 로거를 반환하거나, 기존 로깅 시스템에 연결합니다."""
        # 이 예제에서는 print를 사용하지만, 실제로는 logging 모듈 사용을 권장합니다.
        # from loglib import loglib # 만약 loglib이 사용 가능하다면
        # return loglib.get_logger()
        class SimpleLogger:
            def info(self, msg): print(f"INFO: {msg}")
            def error(self, msg): print(f"ERROR: {msg}")
            def warn(self, msg): print(f"WARN: {msg}")
            def debug(self, msg): print(f"DEBUG: {msg}")
        return SimpleLogger()

    def update(self):
        """
        별도의 스레드에서 실행되어 카메라로부터 지속적으로 프레임을 읽어오는 메서드입니다.
        `self._running` 플래그가 False가 되면 루프를 종료합니다.
        읽어온 프레임은 `self.frame`에, 읽기 성공 여부는 `self.status`에 저장됩니다.
        """
        self._logger.debug("캡처 업데이트 루프 시작됨.")
        while self._running:
            try:
                if self.cap.isOpened():
                    # self.status는 읽기 성공 여부(True/False), self.frame은 읽힌 프레임(NumPy 배열)입니다.
                    (self.status, current_frame) = self.cap.read()
                    if self.status:
                        self.frame = current_frame
                    else:
                        self._logger.warn("캡처 오류: 카메라에서 프레임을 읽는 데 실패했습니다 (status=False).")
                        # 프레임 읽기 실패 시, 잠시 대기 후 루프 계속 (연결 복구 가능성)
                        time.sleep(0.5) # 짧은 대기
                else:
                    self._logger.error("캡처 오류: 업데이트 루프에서 카메라가 열려있지 않습니다. 루프를 중단합니다.")
                    self._running = False # 카메라가 닫혔으면 루프 중단
                    break 
            except cv2.error as e:
                self._logger.error(f"캡처 오류: 프레임 읽기 중 OpenCV 오류 발생: {e}")
                time.sleep(self.FPS) # 오류 발생 시에도 FPS 간격 유지 시도
            except Exception as e:
                self._logger.error(f"캡처 오류: 업데이트 루프에서 예기치 않은 오류 발생: {e}")
                self._running = False # 알 수 없는 오류 발생 시 안전하게 루프 중단
                break
            
            # 설정된 FPS에 맞춰 다음 프레임 읽기까지 대기
            # self._running 플래그가 False면 즉시 종료되도록 time.sleep 대신 다른 방법 고려 가능
            # 예를 들어, 이벤트 기반 대기 또는 짧은 sleep과 플래그 확인 반복
            if self._running:
                 time.sleep(self.FPS)
        self._logger.debug("캡처 업데이트 루프 종료됨.")

    def take_picture(self) -> np.ndarray | None:
        """
        가장 최근에 읽어온 비디오 프레임을 반환합니다.
        `update` 메서드에 의해 `self.frame`이 지속적으로 업데이트됩니다.

        Returns:
            np.ndarray | None: 현재 캡처된 프레임. 프레임이 없거나 오류 발생 시 None.
        """
        if not self._running and not self.status: # 이미 중지되었고 마지막 상태도 실패였다면 None 반환
             self._logger.warn("take_picture 호출: 캡처가 중지되었고 유효한 프레임이 없습니다.")
             return None
        return self.frame

    def stop(self):
        """
        비디오 캡처 스레드를 중지하고 사용된 자원을 안전하게 해제합니다.
        `_running` 플래그를 False로 설정하여 업데이트 루프를 종료시키고,
        스레드가 완료될 때까지 대기한 후 VideoCapture 객체를 해제합니다.
        `is_stopped` 플래그를 사용하여 중복 호출을 방지합니다.
        """
        if not self.is_stopped:
            self._logger.info("캡처 스레드 중지 및 카메라 리소스 해제 요청됨...")
            self._running = False # 업데이트 루프 중지 신호

            if hasattr(self, 'thread') and self.thread.is_alive():
                self._logger.debug("캡처 업데이트 스레드 종료 대기 중...")
                self.thread.join(timeout=2) # 스레드 종료 대기 (최대 2초)
                if self.thread.is_alive():
                    self._logger.warn("캡처 업데이트 스레드가 시간 내에 종료되지 않았습니다.")
            
            if hasattr(self, 'cap') and self.cap.isOpened():
                self.cap.release() # VideoCapture 객체 해제
                self._logger.info("카메라 리소스가 성공적으로 해제되었습니다.")
            
            # cv2.destroyAllWindows() # 일반적으로 GUI 창 사용 시 필요. 여기서는 직접 사용 안 함.
            self.is_stopped = True
            self._logger.info("캡처 기능이 완전히 중지되었습니다.")
        else:
            self._logger.info("캡처 기능은 이미 중지되었습니다.")

    def __enter__(self):
        """컨텍스트 관리자 진입 시 호출됩니다."""
        self._logger.debug("캡처 컨텍스트 관리자 진입 (__enter__).")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        컨텍스트 관리자 종료 시 호출됩니다.
        예외 발생 여부와 관계없이 `stop()` 메서드를 호출하여 리소스를 정리합니다.
        """
        self._logger.debug(f"캡처 컨텍스트 관리자 종료 중 (__exit__), 예외 유형: {exc_type}")
        self.stop()

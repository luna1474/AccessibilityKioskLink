import cv2
import time
import os

class capture():
    def __init__(self):
        # JPEG 이미지를 저장할 디렉토리를 지정합니다.
        self.output_directory = "./projects/"

        # USB 캡처 장치에서 비디오를 가져옵니다.
        self.cap = cv2.VideoCapture(0)  # 0은 기본 카메라 장치입니다. USB 캡처 장치의 경우 다른 인덱스를 사용해야 할 수 있습니다.

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

        # 캡처 장치가 열렸는지 확인합니다.
        if not self.cap.isOpened():
            print("Cannot open camera")
            exit()

        # 디렉토리가 존재하지 않으면 생성합니다.
        if not os.path.exists(self.output_directory):
            os.makedirs(self.output_directory)


    def take_picture(self):
        # 비디오에서 프레임을 읽습니다.
        ret, frame = self.cap.read()
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            exit()

        # 현재 시간을 기반으로 파일 이름을 생성합니다.
        timestamp = int(time.time())
        filename = f"{output_directory}{timestamp}.jpg"

        # JPEG로 프레임을 저장합니다.
        cv2.imwrite(filename, frame)

        return f"{timestamp}.jpg"

    def stop(self):
        # 모든 작업이 완료되면 캡처 장치를 해제합니다.
        self.cap.release()

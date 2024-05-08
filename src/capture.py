import cv2
import time
import os
from threading import Thread

class capture():
    def __init__(self):
        # USB 캡처 장치에서 비디오를 가져옵니다.
        self.cap = cv2.VideoCapture(0)  # 0은 기본 카메라 장치입니다. USB 캡처 장치의 경우 다른 인덱스를 사용해야 할 수 있습니다.

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)
        self.cap.set(cv2.CAP_PROP_POS_AVI_RATIO, 1)

        # 캡처 장치가 열렸는지 확인합니다.
        if not self.cap.isOpened():
            print("Cannot open camera")
            exit()

        # FPS = 1/X
        # X = desired FPS
        self.FPS = 1/30
        self.FPS_MS = int(self.FPS * 1000)

        # Start frame retrieval thread
        self.thread = Thread(target=self.update, args=())
        self.thread.daemon = True
        self.thread.start()

    def update(self):
        while True:
            if self.cap.isOpened():
                (self.status, self.frame) = self.cap.read()
            time.sleep(self.FPS)

    def take_picture(self):
        # 비디오에서 프레임을 읽습니다.
        return self.frame

    def stop(self):
        # 모든 작업이 완료되면 캡처 장치를 해제합니다.
        self.cap.release()


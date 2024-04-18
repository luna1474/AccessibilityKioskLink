import time
from ocr import ocr
from capture import capture
from socketlib import socketlib
from mouselib import mouselib
import cv2
import queue
import os
import sys
from Msg import Msg

import serial


if __name__ == '__main__':
    ser = serial.Serial('/dev/ttyS0', 9600)  # Change '/dev/ttyS0' to the appropriate UART port

    socket_queue = queue.Queue()
    main_queue = queue.Queue()
    server_model = socketlib("server", "0.0.0.0", 5050, main_queue, socket_queue)
    mouse_model = mouselib("server", "0.0.0.0", 5051, None, None)
    server_model.start()
    mouse_model.start()

    cap = capture()
    time.sleep(5)

    while True:
        filename = cap.take_picture()
        image = cv2.imread(filename)
        socket_queue.put(image)
        os.remove(filename)

        print("wait ACK")
        while(main_queue.empty()):
            time.sleep(1)
        print("received msg")
        msg = main_queue.get()
        if msg.msg != "mouse":
            print("fatal error")
            break

        data: str = f"click,{msg.x},{msg.y}"
        print(data)
        ser.write(data.encode()) # command to pico

    cap.stop()
    server_model.stop()
    server_model.join()
    print("end")

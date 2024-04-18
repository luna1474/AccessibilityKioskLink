from webserver import webserver
from socketlib import socketlib
from ocr import ocr
from Msg import Msg

import time
import queue
import cv2


main_queue = queue.Queue()
webserver_queue = queue.Queue()
socket_queue = queue.Queue()
mouse_queue = queue.Queue()

ocr_model = ocr()
webserver_model = webserver(main_queue, webserver_queue)
socket_model = socketlib("client", "172.30.101.158", 5050, main_queue, socket_queue)

webserver_model.start()
socket_model.start()

while(True):
    if not main_queue.empty():
        msg: Msg = main_queue.get()
        print("main_queue receviced: " + msg.msg)

        if msg.msg == "stop":
            webserver_model.stop()
            socket_model.stop()
            break
        elif msg.msg == "photo":
            cv2.imwrite("received.jpg", msg.photo)
            ocr_results = ocr_model.process_ocr(msg.photo)
            name_list = [result[0] for result in ocr_results]
            with webserver_queue.mutex:
                webserver_queue.queue.clear()
            webserver_queue.put(name_list)
            
        elif msg.msg == "mouse":
            try:
                obj = ocr_results[msg.idx]
                print(obj)
                msg = Msg()
                msg.msg = "mouse"
                msg.x = obj[1]
                msg.y = obj[2]
                socket_queue.put(msg)
            except:
                print("indexing error")

    else:
        time.sleep(0.2)

webserver_model.join()
socket_model.join()
print("end")

import socket
import pickle
from Msg import Msg
import time
import socketlib
# import pyautogui

class mouselib(socketlib.socketlib):
    def __init__(self, mode: str, ip_addr: str, port: int, main_queue, message_queue):
        super().__init__(mode, ip_addr, port, main_queue, message_queue)

    def run(self):
        if (self._mode == "server"):
            self.run_server()

            while (self._running):
                try:
                    msg: Msg = self.my_recv(1024)
                    self.click_mouse(msg)
                except Exception as e:
                    print(e)
                    self._logger.error("cannot receive data")
                    time.sleep(1)
                time.sleep(0.1)
                
        elif(self._mode == "client"):
            while (self._running):
                if (self._connection == None): self.run_client()

                if (not self._message_queue.empty()):
                    msg = self._message_queue.get()
                    self.my_send(msg)
                else:
                    time.sleep(0.5)

    def click_mouse(self, msg):
        x = msg.x
        y = msg.y
        # pyautogui.click(x, y)

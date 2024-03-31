import socket
import pickle
from Msg import Msg
import time
import socketlib


class mouselib(socketlib.socketlib):
    def __init__(self, mode: str, ip_addr: str, port: int, main_queue, message_queue):
        super().__init__(mode, ip_addr, port, main_queue, message_queue)

    def run(self):
        if (self._mode == "server"):
            import serial
            self.ser = serial.Serial('/dev/ttyS0', 9600)  # Change '/dev/ttyS0' to the appropriate UART port
            self.run_server()

            while (self._running):
                try:
                    data: str = self.my_recv(1024)
                    self.ser.write(data.encode()) # command to pico
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
                    data: str = f"click,{msg.x},{msg.y}"
                    self.my_send(data)
                else:
                    time.sleep(0.5)

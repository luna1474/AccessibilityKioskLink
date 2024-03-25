import socket
import pickle
from Msg import Msg
import time
import Threadlib

class socketlib(Threadlib.Threadlib):
    def __init__(self, mode: str, ip_addr: str, port: int, main_queue, message_queue):
        super().__init__("Socket : " + mode, main_queue, message_queue)
        self._connection = None
        self._mode = mode
        self._ip_addr = ip_addr
        self._port = port

    def run(self):
        super()
        if (self._mode == "server"):
            self.run_server()

            while (self._running):
                while (not self._message_queue.empty()):
                    img = self._message_queue.get()
                    self.photo_send(img)
                    print(f"server photo send")

                    msg: Msg = self.my_recv(4096)
                    print(f"server received {msg.msg}")
                    self._main_queue.put(msg)
                time.sleep(0.2)
                
        elif(self._mode == "client"):
            while (self._running):
                self.run_client()

                error_count = 0
                while (self._running and error_count < 20):
                    try:
                        image = self.photo_recv()
                        msg = Msg()
                        msg.msg = "photo"
                        msg.photo = image
                        self._main_queue.put(msg)
                        error_count = 0

                        while(self._message_queue.empty()):
                            time.sleep(0.2)
                        msg = self._message_queue.get()
                        self.my_send(msg)
                        print(f"client ACK send")

                    except Exception as e:
                        print(e)
                        self._logger.error("cannot receive data")
                        error_count += 1
                        time.sleep(1)

                self._connection.close()
        
    # =============================================================================
    # 서버 코드
    # =============================================================================
    def run_server(self):
        # 1. 초기화
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        
        # 2. bind
        server.bind((self._ip_addr, self._port))
        
        # 3. listen
        server.listen(1)
        
        # 4. accept
        
        self._logger.debug('ready to connect client')
        self._connection, self._ip_addr = server.accept()
        

    # =============================================================================
    # 클라이언트 코드
    # =============================================================================
    def run_client(self):
        # 1. 초기화
        self._connection = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        
        # 2. connect
        self._logger.debug('try connect to server')

        try_count = 0
        while (True):
            try:
                self._connection.connect((self._ip_addr, self._port))
                break
            except:
                self._logger.error(f"cannot connect to server. retry : {try_count}")
                try_count += 1
                
                if 60 < try_count:
                    time.sleep(120)
                elif 10 < try_count:
                    time.sleep(60)
                else:
                    time.sleep(5)
        self._logger.info(f'successfully connect to server - {self._ip_addr}:{self._port}')
        print(f'successfully connect to server - {self._ip_addr}:{self._port}')
        
        
    # =============================================================================
    # 공통 코드
    # =============================================================================

    def stop(self):
        self._connection.close()
        self._logger.debug('Close Connection')
        super()

    def my_recv(self, B_SIZE):
        data = b""
        # while self._running:
        packet = self._connection.recv(B_SIZE)
        # if not packet: break
        data += packet

        msg = pickle.loads(data)
        return msg

    def my_send(self, msg):
        data = pickle.dumps(msg) # 직렬화
        self._connection.sendall(data)
        return 0

    def photo_send(self, img):
        # 이미지 파일을 pickle로 직렬화
        img_data = pickle.dumps(img)
        
        # 데이터 크기 전송
        data_size = len(img_data)
        self._connection.sendall(data_size.to_bytes(4, byteorder='big'))  # 데이터 크기를 4바이트로 전송
        
        # 직렬화된 이미지 데이터 전송
        while img_data:
            sent_bytes = self._connection.send(img_data)
            img_data = img_data[sent_bytes:]

    def photo_recv(self):
        # 데이터 크기 수신
        data_size_bytes = self._connection.recv(4)  # 데이터 크기는 4바이트로 전송되었으므로 4바이트를 받음
        data_size = int.from_bytes(data_size_bytes, byteorder='big')  # 데이터 크기를 정수로 변환
        
        # 직렬화된 이미지 데이터 수신
        img_data = b''  # 이미지 데이터를 저장할 변수 초기화
        while len(img_data) < data_size:
            packet = self._connection.recv(min(data_size - len(img_data), 4096))  # 최대 4096바이트씩 데이터 수신
            if not packet:
                break
            img_data += packet
        
        # 직렬화된 이미지 데이터를 원본 이미지로 복원
        img = pickle.loads(img_data)
        return img
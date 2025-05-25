"""
이 모듈은 `Threadlib`를 확장하여 TCP/IP 소켓 통신 기능을 제공합니다.
서버 또는 클라이언트 모드로 작동할 수 있으며, JSON 형식으로 직렬화된 메시지 및 사진 데이터를 송수신합니다.
사진 데이터는 Base64로 인코딩되어 JSON 메시지에 포함됩니다.
"""
import socket
# import pickle # pickle은 더 이상 사용되지 않습니다.
from Msg import Msg
import time
import Threadlib
import json # JSON 직렬화/역직렬화를 위해 사용됩니다.
import base64 # 이미지 데이터의 Base64 인코딩/디코딩을 위해 사용됩니다.
import cv2 # OpenCV를 사용하여 이미지 데이터를 처리합니다.
import numpy as np # NumPy 배열(OpenCV 이미지)을 다루기 위해 사용됩니다.
from dataclasses import asdict # Msg 객체를 딕셔너리로 변환하기 위해 사용됩니다.

class socketlib(Threadlib.Threadlib):
    """
    TCP/IP 소켓 통신을 처리하는 스레드 클래스입니다.
    서버 또는 클라이언트 모드로 동작하며, JSON 메시지 및 Base64 인코딩된 이미지 데이터를 송수신합니다.
    `Threadlib`를 상속받아 자체 스레드에서 실행됩니다.
    """
    def __init__(self, mode: str, ip_addr: str, port: int, main_queue, message_queue):
        """
        socketlib 인스턴스를 초기화합니다.

        Args:
            mode (str): 소켓 모드 ("server" 또는 "client").
            ip_addr (str): 서버 또는 클라이언트의 IP 주소.
            port (int): 사용할 포트 번호.
            main_queue (queue.Queue): 메인 애플리케이션 로직으로 메시지를 보내기 위한 큐.
            message_queue (queue.Queue): 이 소켓 스레드로 메시지를 받기 위한 큐.
        """
        super().__init__(f"Socket : {mode} ({ip_addr}:{port})", main_queue, message_queue)
        self._connection: socket.socket | None = None # 활성 소켓 연결 객체
        self._mode: str = mode # 소켓 작동 모드 ("server" 또는 "client")
        self._ip_addr: str = ip_addr # IP 주소
        self._port: int = port # 포트 번호
        self._logger.info(f"소켓 라이브러리 초기화됨: 모드={mode}, IP={ip_addr}, 포트={port}")

    def run(self):
        """
        소켓 스레드의 주 실행 로직입니다.
        모드에 따라 서버 또는 클라이언트 루프를 실행합니다.
        """
        super()
        if (self._mode == "server"):
            try:
                self.run_server()
            except socket.error as e:
                self._logger.error(f"SocketLib Server: Failed to start server: {e}")
                self.stop() # Ensure thread cleanup if server can't start
                return
            except Exception as e: # Catch any other unexpected error during server setup
                self._logger.error(f"SocketLib Server: Unexpected error during server setup: {e}")
                self.stop()
                return

            while (self._running):
                if not self._connection: # Connection might have been lost or failed in run_server
                    self._logger.error("SocketLib Server: No active connection. Attempting to re-establish.")
                    try:
                        # This assumes run_server can be called again to wait for a new connection.
                        # Or, a more specific re-accept logic might be needed.
                        # For now, let's try to re-run the accept logic.
                        # This part needs careful thought based on desired server behavior on disconnect.
                        # A simple approach: re-run accept part of run_server or a dedicated method.
                        # For this change, I'll log and break if connection is lost, assuming run_server sets it.
                        # A robust server would re-listen/re-accept.
                        # Let's assume run_server fully establishes the connection or throws error.
                        # If connection is lost mid-loop, it's handled by send/recv errors.
                        # This check is more for if run_server completed but _connection is still None.
                        self._logger.info("SocketLib Server: Waiting for new connection...")
                        # Simplified: Re-accepting logic should be in run_server or a new method.
                        # For now, if _connection is None after run_server, it's an issue.
                        # The original code implies run_server() blocks until a connection or fails.
                        # So, if we are past run_server(), _connection should be set.
                        # This path (self._connection is None here) should ideally not be hit if run_server is robust.
                        # However, if a connection drops, the send/recv will fail.
                        self._logger.error("SocketLib Server: Lost connection. Exiting server run loop.")
                        break # Exit the while self._running loop
                    except socket.error as e:
                        self._logger.error(f"SocketLib Server: Error trying to re-establish connection: {e}")
                        break # Exit loop
                    except Exception as e:
                        self._logger.error(f"SocketLib Server: Unexpected error re-establishing connection: {e}")
                        break


                while (not self._message_queue.empty() and self._running): # self._running 조건 추가
                    img = self._message_queue.get() # 메시지 큐에서 이미지 가져오기
                    if not self._connection:
                        self._logger.error("SocketLib 서버: 사진 전송 시도 중 연결 없음.")
                        break # 내부 루프 중단, 외부 루프에서 연결 재시도
                    self.photo_send(img) # 사진 전송
                    self._logger.info("SocketLib 서버: 사진 전송 완료.") # print 문을 logger로 변경

                    # 사진 전송 후 클라이언트로부터 응답 또는 다음 메시지 수신
                    if not self._connection:
                        self._logger.error("SocketLib 서버: 응답 수신 시도 중 연결 없음.")
                        break
                    received_msg: Msg | None = self.my_recv(4096) # 메시지 수신 (버퍼 크기 4096)
                    if received_msg:
                        self._logger.info(f"SocketLib 서버: 메시지 수신됨 - {received_msg.msg}") # print 문을 logger로 변경
                        self._main_queue.put(received_msg) # 메인 큐로 전달
                    else:
                        self._logger.warn("SocketLib 서버: 수신된 메시지 없음 또는 오류 발생.")
                        # 연결 문제 발생 시 루프를 중단하여 재연결 시도
                        if not self._connection: break 
                time.sleep(0.2) # 메시지 큐 확인 주기
            
            if self._connection: # 서버 루프 종료 시 연결 닫기
                try:
                    self._connection.close()
                    self._logger.info("SocketLib 서버: 서버 루프 종료로 연결이 닫혔습니다.")
                except socket.error as e:
                    self._logger.error(f"SocketLib 서버: 연결 닫기 중 오류 발생: {e}")
                self._connection = None

        elif(self._mode == "client"):
            while (self._running): # 스레드가 실행 중인 동안 반복
                if not self._connection: # 연결이 없는 경우
                    try:
                        self.run_client() # 클라이언트 연결 시도
                        if not self._connection: # 연결 실패 시
                            self._logger.warn("SocketLib 클라이언트: 서버 연결 실패. 5초 후 재시도합니다.")
                            time.sleep(5) # 5초 대기 후 재시도
                            continue # 다음 루프 반복으로 넘어가 연결 재시도
                    except Exception as e:
                        self._logger.error(f"SocketLib 클라이언트: run_client 중 예외 발생: {e}. 5초 후 재시도합니다.")
                        time.sleep(5)
                        continue
                
                # 연결이 설정된 경우 내부 루프 실행
                error_count = 0
                while (self._running and self._connection and error_count < 20):
                    try:
                        # 사진 수신
                        image = self.photo_recv() 
                        if image is None:
                            self._logger.warn("SocketLib 클라이언트: photo_recv가 None을 반환했습니다. 연결 확인이 필요할 수 있습니다.")
                            error_count +=1
                            time.sleep(1) 
                            if error_count >= 5 : # 연속 오류 발생 시 연결 문제로 간주
                                self._logger.error("SocketLib 클라이언트: photo_recv에서 연속 오류 발생. 연결을 재설정합니다.")
                                if self._connection: self._connection.close()
                                self._connection = None 
                                break # 내부 루프 중단하여 외부 루프에서 연결 재시도
                            continue

                        # 사진을 Msg 객체에 담아 메인 큐로 전송
                        msg = Msg(msg="photo", photo=image)
                        self._main_queue.put(msg)
                        error_count = 0 # 성공 시 오류 카운트 초기화

                        # 메시지 큐에서 보낼 메시지 확인 및 전송
                        while(self._message_queue.empty() and self._running):
                            time.sleep(0.2) # 메시지 대기
                        if not self._running: break 
                        
                        msg_to_send = self._message_queue.get()
                        send_status = self.my_send(msg_to_send)
                        if send_status == 0 : # 성공적인 전송 가정 (my_send는 현재 오류 시에도 0을 반환)
                            self._logger.debug(f"SocketLib 클라이언트: ACK 메시지 전송됨 - {msg_to_send.msg}")
                        else: # my_send가 실패를 나타내는 값을 반환하도록 수정되었다고 가정
                             self._logger.error("SocketLib 클라이언트: my_send 실패. 오류 카운트 증가.")
                             error_count +=1
                             # 실패 시 즉시 재시도보다는 photo_recv에서 주로 연결 상태를 감지

                    except socket.error as e: # 소켓 관련 오류 (연결 끊김 등)
                        self._logger.error(f"SocketLib 클라이언트: 주 루프에서 소켓 오류 발생: {e}")
                        error_count += 1
                        time.sleep(1)
                    except Exception as e: # photo_recv, my_send 내부에서 처리되지 않은 기타 예외
                        self._logger.error(f"SocketLib 클라이언트: 주 루프에서 예기치 않은 오류 발생: {e}")
                        error_count += 1
                        time.sleep(1) # 잠시 대기
                    
                    if error_count >= 20: # 오류 횟수가 임계값을 초과하면 연결 문제로 간주
                        self._logger.error("SocketLib 클라이언트: 오류 임계값 도달. 연결을 재설정합니다.")
                        if self._connection:
                            try:
                                self._connection.close()
                            except socket.error as se_close:
                                self._logger.error(f"SocketLib 클라이언트: 소켓 닫기 중 오류: {se_close}")
                        self._connection = None 
                        break # 내부 루프 중단, 외부 루프에서 연결 재시도

                if self._connection: # 내부 루프가 정상 종료되었으나 연결이 남아있는 경우
                    try:
                        self._connection.close()
                        self._logger.info("SocketLib 클라이언트: 내부 루프 종료로 연결이 닫혔습니다.")
                    except socket.error as e:
                        self._logger.error(f"SocketLib 클라이언트: 연결 닫기 중 오류: {e}")
                self._connection = None # 다음 외부 루프 반복을 위해 연결 상태 초기화
        
    # =============================================================================
    # 서버 코드 (Server Code)
    # =============================================================================
    def run_server(self):
        """
        서버 모드로 소켓을 설정하고 클라이언트의 연결을 대기합니다.
        연결이 성공하면 `self._connection`에 클라이언트 소켓 객체를 저장합니다.

        Raises:
            socket.error: 소켓 바인딩 또는 수신 대기 중 오류 발생 시.
            Exception: 기타 예기치 않은 오류 발생 시.
        """
        server_socket = None
        try:
            server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM) # TCP 소켓 생성
            # SO_REUSEADDR 옵션 설정: 주소 재사용 허용 (서버 재시작 시 유용)
            server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self._logger.info(f"SocketLib 서버: IP {self._ip_addr}, 포트 {self._port}에 바인딩 시도 중...")
            server_socket.bind((self._ip_addr, self._port)) # 지정된 IP 주소와 포트에 소켓 바인딩
            server_socket.listen(1) # 연결 대기열 크기를 1로 설정 (한 번에 하나의 클라이언트만 처리)
            self._logger.info('SocketLib 서버: 클라이언트 연결 대기 중...')
            # 클라이언트 연결 수락 (연결될 때까지 대기)
            self._connection, client_address = server_socket.accept()
            self._logger.info(f'SocketLib 서버: {client_address} 에서 클라이언트 연결됨.')
            # 참고: self._ip_addr는 서버 자신의 IP이므로, client_address[0]으로 덮어쓰지 않습니다.
        except socket.error as e:
            self._logger.error(f"SocketLib 서버: 서버 설정 중 소켓 오류 발생: {e}")
            if server_socket: # 소켓이 생성되었다면 닫기 시도
                server_socket.close()
            self._connection = None # 연결 실패 시 None으로 설정
            raise # 예외를 다시 발생시켜 호출자(run 메서드)가 처리하도록 함
        except Exception as e:
            self._logger.error(f"SocketLib 서버: 서버 설정 중 예기치 않은 오류 발생: {e}")
            if server_socket:
                server_socket.close()
            self._connection = None
            raise
        # 주의: 현재 server_socket (리스닝 소켓)은 self에 저장되지 않아,
        # 연결 성공 후 스레드가 종료되면 리스닝 소켓이 명시적으로 닫히지 않을 수 있습니다.
        # 견고한 서버는 리스닝 소켓도 적절히 관리해야 합니다. (예: self.server_socket = server_socket)


    # =============================================================================
    # 클라이언트 코드 (Client Code)
    # =============================================================================
    def run_client(self):
        """
        클라이언트 모드로 서버에 연결을 시도합니다.
        연결 성공 시 `self._connection`에 서버 소켓 객체를 저장합니다.
        연결 실패 시 `self._connection`은 `None`으로 유지됩니다.
        이 메서드는 연결 시도 중 발생하는 `socket.error` 또는 기타 `Exception`을 로깅합니다.
        """
        try:
            # 새 TCP 소켓 생성
            self._connection = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self._logger.debug(f'SocketLib 클라이언트: 서버 {self._ip_addr}:{self._port}에 연결 시도 중...')
            # 지정된 IP 주소와 포트로 서버에 연결 시도
            self._connection.connect((self._ip_addr, self._port))
            self._logger.info(f'SocketLib 클라이언트: 서버에 성공적으로 연결됨 - {self._ip_addr}:{self._port}')
        except socket.error as e:
            self._logger.error(f"SocketLib 클라이언트: 서버 연결 실패 ({self._ip_addr}:{self._port}). 오류: {e}")
            if self._connection: # 부분적으로 연결되었을 수 있는 소켓 정리
                try: self._connection.close()
                except socket.error: pass # 이미 닫혔거나 문제 있는 소켓일 수 있음
            self._connection = None # 연결 실패를 명시
            # 이 함수는 연결 시도만 하고, 재시도 로직은 호출자(run 메서드)가 담당합니다.
        except Exception as e:
            self._logger.error(f"SocketLib 클라이언트: 연결 중 예기치 않은 오류 발생: {e}")
            if self._connection:
                try: self._connection.close()
                except socket.error: pass
            self._connection = None
        
        
    # =============================================================================
    # 공통 코드 (Common Code for Send/Receive)
    # =============================================================================

    def stop(self):
        """
        소켓 연결을 닫고 스레드를 중지합니다.
        `Threadlib`의 `stop` 메서드를 호출하여 스레드 종료 이벤트를 설정합니다.
        """
        if self._connection: # 활성 연결이 있는 경우
            try:
                self._connection.shutdown(socket.SHUT_RDWR) # 양방향 데이터 전송 중단
                self._connection.close() # 소켓 닫기
                self._logger.info('SocketLib: 연결이 정상적으로 닫혔습니다.')
            except socket.error as e:
                self._logger.error(f"SocketLib: 연결 닫기 중 오류 발생: {e}")
            finally:
                self._connection = None
        super().stop() # 부모 클래스의 stop 메서드 호출 (스레드 종료 이벤트 설정)

    def my_recv(self, buffer_size: int = 4096) -> Msg | None:
        """
        연결된 소켓으로부터 JSON 형식의 메시지를 수신하고 Msg 객체로 역직렬화합니다.
        사진 데이터가 포함된 경우 Base64 디코딩 및 OpenCV 이미지 변환을 수행합니다.

        Args:
            buffer_size (int): 한 번에 수신할 최대 바이트 수. JSON 메시지가 이보다 클 경우 문제가 발생할 수 있습니다.
                               더 견고한 구현을 위해서는 메시지 길이를 먼저 수신하는 프로토콜이 필요합니다.

        Returns:
            Msg | None: 수신 및 역직렬화된 Msg 객체. 오류 발생 또는 연결 종료 시 None을 반환합니다.
        """
        if not self._connection:
            self._logger.error("my_recv: 수신 시도 중 연결 없음.")
            return None
        try:
            data_bytes = self._connection.recv(buffer_size)
            if not data_bytes: # 상대방이 연결을 닫았을 경우 빈 바이트 수신
                self._logger.warn("my_recv: 상대방이 연결을 닫았습니다 (수신된 데이터 없음).")
                if self._connection: self._connection.close() # 여기서 연결을 정리
                self._connection = None
                return None

            json_str = data_bytes.decode('utf-8') # UTF-8로 디코딩
            data_dict = json.loads(json_str) # JSON 문자열을 딕셔너리로 변환

            # 사진 데이터 처리 (Base64 -> OpenCV 이미지)
            if data_dict.get('photo') and isinstance(data_dict['photo'], str):
                try:
                    img_bytes = base64.b64decode(data_dict['photo']) # Base64 디코딩
                    np_arr = np.frombuffer(img_bytes, np.uint8) # 바이트를 NumPy 배열로
                    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR) # NumPy 배열을 OpenCV 이미지로
                    if img is None:
                        self._logger.error("my_recv: Base64 문자열에서 이미지 디코딩 실패 (cv2.imdecode가 None 반환).")
                        data_dict['photo'] = None
                    else:
                        data_dict['photo'] = img
                except Exception as e:
                    self._logger.error(f"my_recv: Base64 이미지 디코딩 중 오류: {e}")
                    data_dict['photo'] = None
            
            # Msg dataclass의 필드와 일치하는 키만 필터링하여 Msg 객체 생성
            # 이는 수신된 JSON에 Msg에 없는 추가 필드가 있어도 오류 없이 처리하기 위함입니다.
            known_fields = Msg.__annotations__.keys() if hasattr(Msg, '__annotations__') else []
            filtered_data_dict = {k: v for k, v in data_dict.items() if k in known_fields}
            
            return Msg(**filtered_data_dict)

        except json.JSONDecodeError as e:
            self._logger.error(f"my_recv: JSON 디코딩 오류: {e}. 수신된 데이터: {data_bytes[:100]}") # 처음 100바이트만 로깅
            return None
        except UnicodeDecodeError as e:
            self._logger.error(f"my_recv: UTF-8 디코딩 오류: {e}. 수신된 데이터: {data_bytes[:100]}")
            return None
        except socket.timeout: # 소켓 타임아웃 설정 시 필요
            self._logger.warn("my_recv: 소켓 수신 타임아웃.")
            return None
        except socket.error as e: # 기타 소켓 오류 (예: 연결 리셋)
            self._logger.error(f"my_recv: 소켓 수신 오류: {e}")
            if self._connection: self._connection.close()
            self._connection = None
            return None
        except Exception as e: # 예상치 못한 기타 모든 오류
            self._logger.error(f"my_recv: 메시지 수신 중 예기치 않은 오류: {e}")
            return None


    def my_send(self, msg_obj: Msg) -> int:
        """
        Msg 객체를 JSON으로 직렬화하여 연결된 소켓으로 전송합니다.
        Msg 객체 내에 사진 데이터(NumPy 배열)가 있으면 Base64로 인코딩하여 JSON에 포함합니다.

        Args:
            msg_obj (Msg): 전송할 Msg 객체.

        Returns:
            int: 성공 시 0, 실패 시 -1 (또는 예외 발생). 현재 구현은 대부분 0을 반환하고 오류는 로깅합니다.
                 더 명확한 오류 처리를 위해 실패 시 -1 반환 또는 예외 발생을 고려할 수 있습니다.
        """
        if not self._connection:
            self._logger.error("my_send: 메시지 전송 시도 중 연결 없음.")
            return -1 # 연결 없음을 명시적으로 반환

        try:
            msg_dict = asdict(msg_obj) # Msg 객체를 딕셔너리로 변환

            # 사진 데이터 처리 (OpenCV 이미지 -> Base64 문자열)
            if msg_dict.get('photo') is not None and isinstance(msg_dict['photo'], np.ndarray):
                try:
                    # 이미지를 JPEG 형식으로 인코딩 후 Base64로 변환
                    success, buffer = cv2.imencode('.jpg', msg_dict['photo'])
                    if not success:
                        self._logger.error("my_send: 이미지 JPEG 인코딩 실패.")
                        msg_dict['photo'] = None # 사진 정보 제거 또는 오류 처리
                    else:
                        img_base64 = base64.b64encode(buffer).decode('utf-8') # Base64 인코딩 및 문자열 변환
                        msg_dict['photo'] = img_base64
                except Exception as e:
                    self._logger.error(f"my_send: 이미지 Base64 인코딩 중 오류: {e}")
                    msg_dict['photo'] = None
            
            json_str = json.dumps(msg_dict) # 딕셔너리를 JSON 문자열로 직렬화
            data_bytes = json_str.encode('utf-8') # UTF-8 바이트로 인코딩
            
            self._connection.sendall(data_bytes) # 모든 데이터 전송
            return 0 # 성공
        except socket.error as e: # 소켓 관련 오류 (예: 연결 끊김)
            self._logger.error(f"my_send: 소켓 오류 발생: {e}")
            if self._connection: self._connection.close()
            self._connection = None
            return -1
        except Exception as e: # 기타 예외 (JSON 직렬화 오류 등)
            self._logger.error(f"my_send: 메시지 전송 중 예기치 않은 오류: {e}")
            return -1


    def photo_send(self, img: np.ndarray):
        """
        사진(NumPy 배열)을 Base64 인코딩된 JSON 형식으로 전송합니다.
        먼저 데이터의 크기를 4바이트로 보내고, 그 다음 실제 데이터를 전송합니다.

        Args:
            img (np.ndarray): 전송할 OpenCV 이미지 (NumPy 배열).
        """
        if not self._connection:
            self._logger.error("photo_send: 사진 전송 시도 중 연결 없음.")
            return
        if not isinstance(img, np.ndarray):
            self._logger.error(f"photo_send: 입력값은 NumPy 배열이어야 합니다. 실제 타입: {type(img)}")
            return

        try:
            # 이미지를 JPEG 형식으로 인코딩 후 Base64로 변환
            success, buffer = cv2.imencode('.jpg', img)
            if not success:
                self._logger.error("photo_send: 이미지 JPEG 인코딩 실패.")
                return
            img_base64 = base64.b64encode(buffer).decode('utf-8')
        except Exception as e:
            self._logger.error(f"photo_send: 이미지 Base64 인코딩 중 오류: {e}")
            return

        # JSON 페이로드 생성
        payload = {'type': 'image', 'data': img_base64}
        try:
            json_bytes = json.dumps(payload).encode('utf-8')
        except Exception as e:
            self._logger.error(f"photo_send: JSON 직렬화 중 오류: {e}")
            return
        
        # 데이터 크기를 4바이트 빅엔디안으로 전송
        data_size = len(json_bytes)
        try:
            self._connection.sendall(data_size.to_bytes(4, byteorder='big'))
            # 실제 JSON 데이터 전송
            self._connection.sendall(json_bytes)
            self._logger.debug(f"photo_send: 사진 페이로드 전송 완료 (크기: {data_size} 바이트).")
        except socket.error as e:
            self._logger.error(f"photo_send: 소켓 오류 발생: {e}")
            if self._connection: self._connection.close()
            self._connection = None
        except Exception as e:
            self._logger.error(f"photo_send: 사진 전송 중 예기치 않은 오류: {e}")


    def photo_recv(self) -> np.ndarray | None:
        """
        사진 데이터를 수신하여 OpenCV 이미지(NumPy 배열)로 디코딩합니다.
        먼저 데이터 크기(4바이트)를 수신하고, 그 다음 실제 JSON 데이터를 수신합니다.
        JSON 데이터 내의 Base64 인코딩된 사진을 디코딩합니다.

        Returns:
            np.ndarray | None: 수신 및 디코딩된 OpenCV 이미지. 오류 발생 시 None.
        """
        if not self._connection:
            self._logger.error("photo_recv: 사진 수신 시도 중 연결 없음.")
            return None
        try:
            # 데이터 크기 수신 (4바이트)
            data_size_bytes = self._connection.recv(4)
            if not data_size_bytes or len(data_size_bytes) < 4: # 연결 종료 또는 불완전 수신
                self._logger.error("photo_recv: 데이터 크기 수신 실패 (연결 문제 가능성).")
                if self._connection: self._connection.close()
                self._connection = None
                return None
            data_size = int.from_bytes(data_size_bytes, byteorder='big') # 빅엔디안 바이트 순서로 정수 변환
            self._logger.debug(f"photo_recv: 수신할 사진 페이로드 크기: {data_size} 바이트.")
            
            # 실제 JSON 데이터 수신
            img_data_json_bytes = b''
            while len(img_data_json_bytes) < data_size:
                remaining_size = data_size - len(img_data_json_bytes)
                # 수신할 남은 크기와 일반적인 버퍼 크기(4096) 중 작은 값을 사용
                packet = self._connection.recv(min(remaining_size, 4096))
                if not packet: # 연결이 중간에 끊긴 경우
                    self._logger.error("photo_recv: 사진 데이터 수신 중 연결이 끊겼습니다.")
                    if self._connection: self._connection.close()
                    self._connection = None
                    return None
                img_data_json_bytes += packet
            
            if not img_data_json_bytes: # 데이터 크기를 받았으나 실제 데이터가 없는 경우
                self._logger.error("photo_recv: 데이터 크기 수신 후 실제 사진 데이터 없음.")
                return None

            # JSON 데이터 역직렬화 및 Base64 디코딩
            payload_str = img_data_json_bytes.decode('utf-8')
            payload = json.loads(payload_str)

            if payload.get('type') == 'image' and 'data' in payload:
                img_base64 = payload['data']
                try:
                    img_bytes = base64.b64decode(img_base64) # Base64 디코딩
                    np_arr = np.frombuffer(img_bytes, np.uint8) # 바이트를 NumPy 배열로
                    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR) # NumPy 배열을 OpenCV 이미지로
                    if img is None:
                        self._logger.error("photo_recv: Base64 데이터로부터 이미지 디코딩 실패 (cv2.imdecode가 None 반환).")
                        return None
                    return img
                except Exception as e:
                    self._logger.error(f"photo_recv: Base64 이미지 디코딩 중 오류: {e}")
                    return None
            else: # 예상치 못한 페이로드 구조
                self._logger.error(f"photo_recv: 유효하지 않은 페이로드 구조: {payload}")
                return None

        except json.JSONDecodeError as e:
            self._logger.error(f"photo_recv: JSON 디코딩 오류: {e}. 수신된 데이터: {img_data_json_bytes[:100] if 'img_data_json_bytes' in locals() else 'N/A'}")
            return None
        except UnicodeDecodeError as e:
            self._logger.error(f"photo_recv: UTF-8 디코딩 오류: {e}. 수신된 데이터: {img_data_json_bytes[:100] if 'img_data_json_bytes' in locals() else 'N/A'}")
            return None
        except socket.timeout:
            self._logger.warn("photo_recv: 소켓 수신 타임아웃.")
            return None
        except socket.error as e:
            self._logger.error(f"photo_recv: 소켓 오류 발생: {e}")
            if self._connection: self._connection.close()
            self._connection = None
            return None
        except Exception as e:
            self._logger.error(f"photo_recv: 사진 수신 중 예기치 않은 오류: {e}")
            return None
                    data_dict['photo'] = None # Or handle error appropriately
                else:
                    data_dict['photo'] = img
            except Exception as e:
                self._logger.error(f"Error decoding base64 image: {e}")
                data_dict['photo'] = None # Or handle error as per application needs

        # Filter out keys not in Msg dataclass before creating Msg object
        # This assumes Msg class fields are known or can be introspected.
        # For simplicity, assuming Msg fields are 'msg', 'photo', 'x', 'y', 'command'.
        # A better way would be to inspect Msg.__annotations__ or similar if Msg is a dataclass.
        known_fields = Msg.__annotations__.keys() if hasattr(Msg, '__annotations__') else ['msg', 'photo', 'x', 'y', 'command']
        filtered_data_dict = {k: v for k, v in data_dict.items() if k in known_fields}
        
        msg = Msg(**filtered_data_dict)
        return msg

    def my_send(self, msg_obj: Msg):
        msg_dict = asdict(msg_obj)

        if msg_dict.get('photo') is not None and isinstance(msg_dict['photo'], np.ndarray):
            try:
                _, buffer = cv2.imencode('.jpg', msg_dict['photo'])
                img_base64 = base64.b64encode(buffer).decode('utf-8')
                msg_dict['photo'] = img_base64
            except Exception as e:
                self._logger.error(f"Error encoding image to base64: {e}")
                msg_dict['photo'] = None # Or handle error as per application needs
        
        json_str = json.dumps(msg_dict)
        data_bytes = json_str.encode('utf-8')
        self._connection.sendall(data_bytes)
        return 0

    def photo_send(self, img: np.ndarray):
        if not isinstance(img, np.ndarray):
            self._logger.error(f"photo_send expected a numpy array, got {type(img)}")
            # Potentially raise an error or return early
            return

        try:
            _, buffer = cv2.imencode('.jpg', img)
            img_base64 = base64.b64encode(buffer).decode('utf-8')
        except Exception as e:
            self._logger.error(f"Error encoding image to base64 for photo_send: {e}")
            return # Or raise error

        payload = {'type': 'image', 'data': img_base64}
        json_bytes = json.dumps(payload).encode('utf-8')
        
        data_size = len(json_bytes)
        self._connection.sendall(data_size.to_bytes(4, byteorder='big'))
        self._connection.sendall(json_bytes)
        self._logger.debug(f"Sent photo_send payload size: {data_size}")


    def photo_recv(self):
        data_size_bytes = self._connection.recv(4)
        if not data_size_bytes:
            self._logger.error("Failed to receive data size for photo_recv.")
            return None
        data_size = int.from_bytes(data_size_bytes, byteorder='big')
        self._logger.debug(f"Received photo_recv payload size: {data_size}")
        
        img_data_json_bytes = b''
        while len(img_data_json_bytes) < data_size:
            remaining_size = data_size - len(img_data_json_bytes)
            packet = self._connection.recv(min(remaining_size, 4096))
            if not packet:
                self._logger.error("Connection broken while receiving photo data.")
                return None # Connection lost
            img_data_json_bytes += packet
        
        if not img_data_json_bytes:
            self._logger.error("No photo data received after size.")
            return None

        try:
            payload_str = img_data_json_bytes.decode('utf-8')
            payload = json.loads(payload_str)
        except json.JSONDecodeError as e:
            self._logger.error(f"JSON decode error in photo_recv: {e}")
            return None
        except UnicodeDecodeError as e:
            self._logger.error(f"Unicode decode error in photo_recv: {e}")
            return None

        if payload.get('type') == 'image' and 'data' in payload:
            img_base64 = payload['data']
            try:
                img_bytes = base64.b64decode(img_base64.encode('utf-8'))
                np_arr = np.frombuffer(img_bytes, np.uint8)
                img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                if img is None:
                    self._logger.error("cv2.imdecode returned None in photo_recv.")
                    return None
                return img
            except Exception as e:
                self._logger.error(f"Error decoding base64 image in photo_recv: {e}")
                return None
        else:
            self._logger.error(f"Invalid payload structure in photo_recv: {payload}")
            return None
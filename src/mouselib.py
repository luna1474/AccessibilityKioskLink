"""
이 모듈은 `socketlib.socketlib`를 확장하여 마우스 제어 관련 통신 기능을 제공합니다.
서버 모드에서는 시리얼 포트(`pyserial` 필요)를 통해 연결된 장치(예: 마이크로컨트롤러)로
수신된 명령을 전달합니다. 클라이언트 모드에서는 마우스 좌표와 같은 메시지를
서버로 전송합니다.
"""
import socket
# import pickle # pickle은 더 이상 사용되지 않음
from Msg import Msg
import time
import socketlib # 부모 클래스 socketlib 임포트

# 서버 모드에서만 pyserial이 필요하므로, 조건부 임포트 또는 run 메서드 내에서 임포트합니다.
# import serial # 주석 처리하고 run 메서드 내에서 임포트


class mouselib(socketlib.socketlib):
    """
    마우스 제어 메시지 통신을 위한 클래스입니다. `socketlib.socketlib`의 기능을 상속받습니다.
    서버 모드: 소켓을 통해 수신한 메시지를 시리얼 포트를 통해 외부 장치(예: Pico)로 전달합니다.
    클라이언트 모드: 생성된 마우스 관련 메시지(예: 클릭 좌표)를 소켓을 통해 서버로 전송합니다.
    """
    def __init__(self, mode: str, ip_addr: str, port: int, main_queue, message_queue):
        """
        mouselib 인스턴스를 초기화합니다.

        Args:
            mode (str): 소켓 모드 ("server" 또는 "client").
            ip_addr (str): 서버 IP 주소. 클라이언트 모드에서 연결할 대상 서버의 IP입니다.
                           서버 모드에서는 이 IP에 바인딩합니다.
            port (int): 사용할 포트 번호.
            main_queue (queue.Queue): 메인 애플리케이션 로직으로 메시지를 보내기 위한 큐 (현재 mouselib에서는 직접 사용 안함).
            message_queue (queue.Queue): 이 스레드로 메시지를 받기 위한 큐 (클라이언트 모드에서 전송할 메시지 수신).
        """
        # 부모 클래스 socketlib의 생성자 호출. 스레드 이름에 "Mouse" 추가하여 구분 용이.
        super().__init__(mode, ip_addr, port, main_queue, message_queue)
        # 스레드 이름에 "Mouse"와 모드, 주소를 명시하여 로깅 시 식별 용이성을 높입니다.
        self._name = f"MouseSocket : {mode} ({ip_addr}:{port})" 
        self.ser = None # 서버 모드에서 사용될 시리얼 포트 객체, 초기에는 None으로 설정합니다.
        self._logger.info(f"{self._name}: 마우스 제어 라이브러리가 초기화되었습니다.")

    def stop(self):
        """
        mouselib 스레드를 중지하고 관련된 모든 리소스(소켓 연결, 시리얼 포트 등)를 안전하게 정리합니다.
        부모 클래스(`socketlib`)의 `stop` 메서드를 호출하여 소켓 연결을 닫고,
        추가적으로 이 클래스에서 관리하는 시리얼 포트(`self.ser`)가 열려있다면 닫습니다.
        """
        self._logger.info(f"{self._name}: 중지 요청을 수신했습니다. 리소스 정리를 시작합니다...")
        
        # 부모 클래스(socketlib)의 stop을 먼저 호출합니다.
        # 이 메서드는 self._running을 False로 설정하고, _stop_event를 설정하며,
        # self._connection (소켓 연결)을 닫습니다.
        super().stop() 
        
        # 서버 모드에서 사용된 시리얼 포트가 열려있는 경우 닫습니다.
        # hasattr를 사용하여 self.ser 속성 존재 여부를 먼저 확인합니다 (객체 상태 안정성).
        if hasattr(self, 'ser') and self.ser and self.ser.is_open:
            try:
                self.ser.close()
                self._logger.info(f"{self._name}: 시리얼 포트({self.ser.name if hasattr(self.ser, 'name') else '알 수 없음'})가 성공적으로 닫혔습니다.")
            except Exception as e: # 시리얼 포트 닫기 중 발생할 수 있는 모든 예외 처리
                self._logger.error(f"{self._name}: 시리얼 포트({self.ser.name if hasattr(self.ser, 'name') else '알 수 없음'}) 닫기 중 오류 발생: {e}")
        
        self._logger.info(f"{self._name}: 모든 리소스 정리가 완료되었습니다.")


    def run(self):
        """
        mouselib 스레드의 주 실행 로직입니다.
        서버 모드: 시리얼 포트를 초기화하고, 소켓 서버를 실행하여 클라이언트로부터 메시지를 수신하고,
                  수신된 메시지를 시리얼 포트를 통해 연결된 장치로 전송합니다.
        클라이언트 모드: 소켓 클라이언트를 실행하고, 메시지 큐로부터 받은 마우스 관련 `Msg` 객체를
                       서버로 전송합니다.
        """
        if (self._mode == "server"):
            # 서버 모드 로직
            try:
                import serial # 서버 모드에서만 pyserial 임포트
                # 시리얼 포트 설정 (예: '/dev/ttyS0' 또는 'COM3' 등, 환경에 맞게 수정 필요)
                # timeout=1 설정으로 read/write 작업 시 최대 1초 대기
                self.ser = serial.Serial('/dev/ttyS0', 9600, timeout=1)
                self._logger.info(f"{self._name}: 시리얼 포트 /dev/ttyS0 @ 9600bps 로 연결됨.")
            except ImportError:
                self._logger.error(f"{self._name} 오류: pyserial 라이브러리를 찾을 수 없습니다. 'pip install pyserial'로 설치해주세요.")
                return # 스레드 실행 중단
            except serial.SerialException as e:
                self._logger.error(f"{self._name} 오류: 시리얼 포트 /dev/ttyS0 연결 실패: {e}")
                return # 스레드 실행 중단

            # 부모 클래스(socketlib)의 run_server()를 호출하여 소켓 서버 시작 및 연결 대기
            try:
                super().run_server() # self.run_server() 대신 super() 사용 가능, 혹은 직접 부모의 run_server 호출
                                     # 여기서는 socketlib의 run_server()를 명시적으로 호출하는 것이 더 명확할 수 있음.
                                     # socketlib의 run_server()는 self._connection을 설정함.
            except Exception as e:
                self._logger.error(f"{self._name} 오류: 소켓 서버 시작 실패: {e}")
                if hasattr(self, 'ser') and self.ser and self.ser.is_open:
                    self.ser.close() # 시리얼 포트가 열려있으면 닫기
                return


            self._logger.info(f"{self._name}: 서버 시작됨. 클라이언트로부터 메시지 대기 중...")
            while (self._running and self._connection): # 스레드 실행 중이고, 연결이 유효한 동안 반복
                try:
                    # socketlib의 my_recv를 통해 Msg 객체 수신
                    msg_obj: Msg | None = self.my_recv(4096) 
                    
                    if msg_obj is not None:
                        # 수신된 Msg 객체로부터 시리얼 명령 문자열 생성
                        # 예시: msg_obj.msg가 "click"이고, x, y 좌표가 있으면 "C,x,y\n" 형태의 명령 생성
                        #       msg_obj.msg가 "move" 이고, x, y 좌표가 있으면 "M,x,y\n" 형태의 명령 생성
                        # 이 부분은 실제 Pico 장치와 약속된 프로토콜에 따라 엄격하게 정의되어야 합니다.
                        serial_command_str = None
                        if msg_obj.msg == "mouse_click" and msg_obj.x is not None and msg_obj.y is not None:
                            # 예시: Pico가 "C,x좌표,y좌표\n" 형태의 클릭 명령을 기대한다고 가정
                            serial_command_str = f"C,{int(msg_obj.x)},{int(msg_obj.y)}\n"
                        elif msg_obj.msg == "mouse_move" and msg_obj.x is not None and msg_obj.y is not None:
                            # 예시: Pico가 "M,x좌표,y좌표\n" 형태의 이동 명령을 기대한다고 가정
                            serial_command_str = f"M,{int(msg_obj.x)},{int(msg_obj.y)}\n"
                        elif msg_obj.msg: # 기타 일반 메시지 (예: "reset", "start")
                            # Pico가 일반 문자열 명령도 받는다고 가정 (줄바꿈 추가)
                            serial_command_str = f"{msg_obj.msg.strip()}\n" 
                        
                        if serial_command_str:
                            self._logger.debug(f"{self._name} 서버: 수신된 Msg: {msg_obj}, 생성된 시리얼 명령: '{serial_command_str.strip()}'")
                            try:
                                self.ser.write(serial_command_str.encode('utf-8')) # UTF-8로 인코딩하여 전송
                                self._logger.info(f"{self._name} 서버: 시리얼 명령 '{serial_command_str.strip()}' 전송됨.")
                            except serial.SerialTimeoutException:
                                self._logger.error(f"{self._name} 오류: 시리얼 포트 쓰기 타임아웃 (명령: '{serial_command_str.strip()}').")
                            except serial.SerialException as e_ser:
                                self._logger.error(f"{self._name} 오류: 시리얼 쓰기 중 예외 발생 (명령: '{serial_command_str.strip()}'): {e_ser}")
                                # 심각한 시리얼 오류 시 연결을 닫고 루프를 종료할 수 있음
                                if self._connection: self._connection.close() # 소켓 연결도 닫기
                                self._running = False # 스레드 종료 플래그
                                break 
                        else:
                            self._logger.warn(f"{self._name} 서버: 수신된 Msg로부터 유효한 시리얼 명령을 생성할 수 없음: {msg_obj}")
                    
                    elif msg_obj is None: # my_recv가 None을 반환 (연결 문제 등)
                        self._logger.warn(f"{self._name} 서버: my_recv로부터 None 수신. 연결 확인 필요.")
                        if not self._connection: break # 연결이 없다면 루프 종료
                    else: # 예상치 못한 데이터 구조
                        self._logger.warn(f"{self._name} 서버: 예상치 못한 데이터 수신됨: {msg_obj}")

                except serial.SerialTimeoutException: # 시리얼 쓰기 타임아웃
                    self._logger.error(f"{self._name} 오류: 시리얼 포트 쓰기 타임아웃.")
                except serial.SerialException as e: # 기타 시리얼 통신 오류
                    self._logger.error(f"{self._name} 오류: 시리얼 통신 중 예외 발생: {e}")
                except AttributeError as e: # msg_obj가 None일 때 .msg 접근 시 발생 가능
                    self._logger.error(f"{self._name} 오류: 메시지 객체 속성 접근 오류 (msg_obj가 None일 수 있음): {e}")
                except Exception as e: # 기타 모든 예외
                    self._logger.error(f"{self._name} 서버: 메시지 수신/처리 루프 중 예기치 않은 오류: {e}")
                    # 심각한 오류 시 연결 상태 재확인 또는 짧은 대기
                    if not self._connection : break
                    time.sleep(0.5) 
                
                # 루프 주기 조절 (너무 빠른 반복 방지)
                # time.sleep(0.1) # 기존 코드의 슬립 유지 또는 필요에 따라 조절

            # 스레드 종료 시 시리얼 포트 정리
            if hasattr(self, 'ser') and self.ser and self.ser.is_open:
                self.ser.close()
                self._logger.info(f"{self._name}: 시리얼 포트 연결 종료됨.")
            self._logger.info(f"{self._name}: 서버 모드 실행 종료.")
                
        elif(self._mode == "client"):
            # 클라이언트 모드 로직
            self._logger.info(f"{self._name}: 클라이언트 모드 시작됨.")
            while (self._running): # 스레드 실행 중인 동안 반복
                try:
                    # 연결 상태 확인 및 (재)연결 시도
                    if self._connection is None:
                        super().run_client() # socketlib의 run_client() 호출하여 연결 시도
                        if self._connection is None: # 연결 실패 시
                             self._logger.warn(f"{self._name} 클라이언트: 서버 연결 실패. 5초 후 재시도합니다.")
                             time.sleep(5) # 5초 대기 후 재시도
                             continue # 루프의 처음으로 돌아가 연결 재시도

                    # 메시지 큐에서 전송할 Msg 객체 가져오기
                    if not self._message_queue.empty():
                        msg_to_send: Msg = self._message_queue.get()
                        
                        # socketlib의 my_send는 Msg 객체를 직접 받아 JSON으로 직렬화하여 전송합니다.
                        # main.py에서 mouse_queue에 넣는 데이터는 이미 Msg 객체여야 합니다.
                        # 예시 (main.py에서):
                        #   mouse_msg = Msg()
                        #   mouse_msg.msg = "mouse_click" # 또는 "mouse_move" 등 명령 유형
                        #   mouse_msg.x = 100 # x 좌표
                        #   mouse_msg.y = 200 # y 좌표
                        #   mouse_queue.put(mouse_msg)
                        # 이렇게 전달된 Msg 객체를 mouselib 클라이언트가 받아 서버로 전송합니다.
                        # mouselib 서버는 이 Msg 객체를 받아, msg_obj.msg (예: "mouse_click") 또는
                        # msg_obj.x, msg_obj.y 등을 사용하여 시리얼 명령을 구성해야 합니다.
                        # 현재 서버 로직은 msg_obj.msg만 사용하고 있으므로, 이것이 시리얼 명령 문자열이어야 합니다.
                        # 만약 main.py에서 `f"click,{msg.x},{msg.y}"` 형태의 문자열을 msg_obj.msg에 넣어 보낸다면 일관성이 있습니다.

                        status = self.my_send(msg_to_send) # my_send는 socketlib에서 상속됨
                        if status == 0:
                            self._logger.debug(f"{self._name} 클라이언트: 메시지 전송 성공: {msg_to_send}")
                        else:
                            self._logger.error(f"{self._name} 클라이언트: 메시지 전송 실패 (상태: {status}). 연결을 재설정합니다.")
                            if self._connection: self._connection.close()
                            self._connection = None # 연결 실패로 표시하여 재연결 유도
                            # 실패한 메시지는 다시 큐에 넣거나 별도 처리할 수 있으나, 현재는 유실됨.
                    else:
                        # 메시지 큐가 비어있으면 잠시 대기
                        time.sleep(0.1) # 폴링 간격 조절
                except socket.error as e: # 소켓 통신 오류 (예: 연결 끊김)
                    self._logger.error(f"{self._name} 클라이언트: 소켓 오류 발생: {e}")
                    if self._connection:
                        try: self._connection.close()
                        except socket.error: pass
                    self._connection = None # 연결 실패로 표시하여 재연결 유도
                    time.sleep(5) # 재연결 시도 전 잠시 대기
                except Exception as e: # 기타 예기치 않은 오류
                    self._logger.error(f"{self._name} 클라이언트: 예기치 않은 오류 발생: {e}")
                    # 오류 상황에 따라 연결을 재설정하거나 잠시 대기 후 계속
                    time.sleep(1)
            self._logger.info(f"{self._name}: 클라이언트 모드 실행 종료.")
        
        # 스레드 종료 시, 부모 클래스의 stop도 호출하여 정리 (이미 mouselib의 stop에서 호출될 수 있음)
        # super().stop() # 명시적으로 호출할 필요는 없을 수 있음, mouselib의 stop에서 처리

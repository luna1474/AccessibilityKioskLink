"""
메인 애플리케이션 실행 모듈입니다.
이 모듈은 웹 서버, 소켓 통신, OCR 처리, 마우스 제어 등 다양한 컴포넌트를 초기화하고,
이들 간의 상호작용을 관리하는 중앙 루프를 실행합니다.
각 컴포넌트는 별도의 스레드에서 동작하며, 큐(Queue)를 통해 메시지를 주고받습니다.
"""
from webserver import webserver
from socketlib import socketlib
from ocr import ocr # OCR 처리 클래스
from Msg import Msg # 통신 메시지 객체
from mouselib import mouselib # 마우스 제어 라이브러리

import time
import queue # 스레드 간 통신을 위한 큐 모듈
import cv2 # OpenCV 라이브러리 (이미지 처리)
# import logging # 로깅을 위해 추가 (print 대신 사용 권장)

# --- 애플리케이션 설정 상수 ---
# TODO: 이 값들은 외부 설정 파일(예: JSON, YAML, .env)에서 로드하는 것을 고려하십시오.
SOCKET_SERVER_IP = "172.30.101.158" # 소켓 통신 서버의 IP 주소
SOCKET_SERVER_PORT = 5050          # 소켓 통신 서버의 포트 번호

MOUSE_SERVER_IP = "172.30.101.158"  # 마우스 제어 서버의 IP 주소 (socket_model과 동일할 수 있음)
MOUSE_SERVER_PORT = 5051           # 마우스 제어 서버의 포트 번호

# 로거 설정 (기본 print 대신 사용 권장)
# logger = logging.getLogger(__name__)
# logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
# logger.info("애플리케이션 시작됨") # 시작 로그 예시

# --- 전역 큐(Queue) 정의 ---
# main_queue: 다른 스레드에서 발생한 주요 이벤트를 메인 루프로 전달하는 데 사용
main_queue = queue.Queue()
# webserver_queue: OCR 결과를 웹 서버로 전달하는 데 사용
webserver_queue = queue.Queue()
# socket_queue: 소켓 통신 스레드로 메시지(예: OCR 준비 완료)를 전달하는 데 사용
socket_queue = queue.Queue()
# mouse_queue: 마우스 제어 스레드로 마우스 좌표 등의 명령을 전달하는 데 사용
mouse_queue = queue.Queue()

# --- 컴포넌트 인스턴스 생성 ---
ocr_model = ocr() # OCR 처리 모델 인스턴스화
# 웹 서버 모델 인스턴스화 (메인 큐, 웹서버 수신 큐 전달)
webserver_model = webserver(main_queue, webserver_queue)
# 소켓 통신 클라이언트 모델 인스턴스화 (IP, 포트, 큐 전달)
socket_model = socketlib("client", SOCKET_SERVER_IP, SOCKET_SERVER_PORT, main_queue, socket_queue)
# 마우스 제어 클라이언트 모델 인스턴스화 (IP, 포트, 큐 전달)
# main_queue는 mouselib에서 현재 사용하지 않으므로 None 전달, mouse_queue는 mouselib이 읽을 큐
mouse_model = mouselib("client", MOUSE_SERVER_IP, MOUSE_SERVER_PORT, None, mouse_queue)

# --- 스레드 시작 ---
webserver_model.start() # 웹 서버 스레드 시작
socket_model.start()    # 소켓 통신 스레드 시작
mouse_model.start()     # 마우스 제어 스레드 시작

ocr_results_cache = [] # OCR 결과를 저장할 캐시 변수 (NameError 방지용)

# --- 메인 실행 루프 ---
print("메인 루프 시작됨. 종료하려면 'stop' 메시지를 main_queue에 전달하거나 Ctrl+C를 누르세요.")
try:
    while True:
        if not main_queue.empty(): # 메인 큐에 메시지가 있는지 확인
            msg: Msg = main_queue.get() # 큐에서 메시지 가져오기
            print(f"메인 큐 수신: {msg.msg}") # 수신된 메시지 내용 로깅 (logger.info 권장)

            if msg.msg == "stop": # "stop" 메시지 수신 시
                print("종료 메시지 수신. 모든 스레드를 중지합니다.")
                webserver_model.stop() # 웹 서버 스레드 중지 요청
                socket_model.stop()    # 소켓 통신 스레드 중지 요청
                mouse_model.stop()     # 마우스 제어 스레드 중지 요청
                break # 메인 루프 종료
            
            elif msg.msg == "photo": # "photo" 메시지 수신 시 (사진 데이터 처리)
                if msg.photo is not None:
                    # 수신된 사진을 파일로 저장 (디버깅 또는 확인용)
                    cv2.imwrite("received_image_from_socket.jpg", msg.photo)
                    print("수신된 사진을 'received_image_from_socket.jpg'로 저장했습니다.")
                    
                    # OCR 처리 수행
                    ocr_results_cache = ocr_model.process_ocr(msg.photo) # OCR 결과 캐시에 저장
                    # OCR 결과에서 텍스트 목록만 추출
                    name_list = [result[0] for result in ocr_results_cache if isinstance(result, (list, tuple)) and len(result) > 0]
                    
                    # 웹 서버 큐를 안전하게 비우고 새 데이터 삽입
                    # 기존: with webserver_queue.mutex: webserver_queue.queue.clear()
                    # 변경: 루프를 사용하여 get_nowait()으로 안전하게 비움
                    while not webserver_queue.empty():
                        try:
                            webserver_queue.get_nowait()
                        except queue.Empty: # 큐가 비었으면 루프 종료
                            break 
                    webserver_queue.put(name_list) # 웹 서버 큐에 새 이름 목록 삽입
                    print(f"웹 서버로 전송할 이름 목록: {name_list}")

                    # OCR 처리 준비 완료 메시지를 소켓 큐로 전송
                    ready_msg = Msg()
                    ready_msg.msg = "ready"
                    socket_queue.put(ready_msg)
                    print("OCR 처리 완료. 'ready' 메시지를 소켓으로 전송합니다.")
                else:
                    print("경고: 'photo' 메시지를 받았으나, 실제 사진 데이터가 없습니다.")

            elif msg.msg == "mouse": # "mouse" 메시지 수신 시 (마우스 이벤트 처리)
                try:
                    # msg.idx를 사용하여 ocr_results_cache에서 해당 OCR 결과 항목을 가져옴
                    # 변수명 'obj'를 'selected_ocr_result'로 변경하여 명확성 향상
                    selected_ocr_result = ocr_results_cache[msg.idx]
                    print(f"선택된 OCR 결과: {selected_ocr_result}") # (logger.info 권장)
                    
                    # 새로운 Msg 객체를 생성하여 마우스 제어 정보(좌표)를 담아 mouse_queue로 전송
                    mouse_command_msg = Msg()
                    mouse_command_msg.msg = "mouse_xy_coordinates" # 메시지 유형 명확화
                    # selected_ocr_result의 구조가 [텍스트, x좌표, y좌표]라고 가정
                    if isinstance(selected_ocr_result, (list, tuple)) and len(selected_ocr_result) >= 3:
                        mouse_command_msg.x = selected_ocr_result[1] # x 좌표
                        mouse_command_msg.y = selected_ocr_result[2] # y 좌표
                        mouse_queue.put(mouse_command_msg)
                        print(f"마우스 제어기로 좌표 전송: x={mouse_command_msg.x}, y={mouse_command_msg.y}")
                    else:
                        print(f"오류: 선택된 OCR 결과의 형식이 올바르지 않습니다: {selected_ocr_result}")

                except NameError: # ocr_results_cache가 아직 정의되지 않은 경우 (사진 처리가 먼저 수행되지 않음)
                    print(f"오류: 'ocr_results_cache'가 정의되지 않았습니다. 사진을 먼저 처리해야 합니다.")
                except IndexError: # msg.idx가 ocr_results_cache 범위를 벗어난 경우
                    print(f"오류: 인덱스 {msg.idx}가 OCR 결과 범위를 벗어났습니다. (결과 수: {len(ocr_results_cache)})")
                except TypeError: # ocr_results_cache가 구독 불가능한 타입(예: None)이거나, msg.idx가 정수가 아닐 경우
                    print(f"오류: OCR 결과 또는 인덱스 타입 오류. ocr_results_cache: {type(ocr_results_cache)}, msg.idx: {type(msg.idx)}")
                except Exception as e: # 기타 예기치 않은 오류
                    print(f"마우스 이벤트 처리 중 예기치 않은 오류 발생: {e}")
                    # 오류 발생 시 로깅할 추가 정보 (필요시 주석 해제)
                    # print(f"  ocr_results_cache 상태: {ocr_results_cache if 'ocr_results_cache' in locals() else '정의되지 않음'}")
                    # print(f"  msg.idx 상태: {msg.idx if hasattr(msg, 'idx') else '정의되지 않음'}")
        else:
            # 메인 큐가 비어있을 경우, CPU 사용을 줄이기 위해 잠시 대기
            time.sleep(0.1) # 대기 시간 조절 가능 (0.2초에서 0.1초로 변경)
except KeyboardInterrupt: # Ctrl+C로 종료 시
    print("\n사용자에 의해 프로그램 종료 요청됨 (KeyboardInterrupt). 모든 스레드를 중지합니다.")
    webserver_model.stop()
    socket_model.stop()
    mouse_model.stop()
finally:
    # --- 스레드 종료 대기 ---
    print("모든 스레드가 종료될 때까지 대기 중...")
    webserver_model.join() # 웹 서버 스레드 종료 대기
    socket_model.join()    # 소켓 통신 스레드 종료 대기
    mouse_model.join()     # 마우스 제어 스레드 종료 대기
    print("모든 스레드 종료됨. 프로그램 종료.")

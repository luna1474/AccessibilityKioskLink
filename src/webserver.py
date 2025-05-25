"""
이 모듈은 Flask 웹 서버 기능을 제공하는 `webserver` 클래스를 정의합니다.
웹 UI를 통해 OCR 결과 목록을 표시하고, 사용자의 클릭 이벤트를 받아
메인 애플리케이션 로직으로 전달하며, 주기적으로 데이터를 업데이트합니다.
또한, 원격으로 서버를 종료할 수 있는 엔드포인트를 제공합니다.
"""
from flask import Flask, request, render_template, jsonify
# import json # 현재 json 모듈은 직접 사용되지 않음
import Threadlib # 부모 클래스 Threadlib 임포트
from Msg import Msg # 통신을 위한 Msg 객체 임포트

class webserver(Threadlib.Threadlib):
    """
    Flask 기반의 웹 서버를 운영하는 스레드 클래스입니다.
    웹 UI를 제공하고, 사용자 상호작용을 처리하여 다른 컴포넌트와 통신합니다.
    """
    def __init__(self, main_queue, message_queue):
        """
        webserver 인스턴스를 초기화합니다.

        Args:
            main_queue (queue.Queue): 메인 애플리케이션 로직으로 사용자 액션(예: 클릭)을 전달하기 위한 큐.
            message_queue (queue.Queue): 웹 서버가 다른 컴포넌트로부터 데이터를 수신하기 위한 큐 (예: OCR 결과 목록).
        """
        super().__init__("Flask 웹서버", main_queue, message_queue) # 스레드 이름 변경 및 로거 기본 제공
        self.app = Flask(__name__) # Flask 애플리케이션 인스턴스 생성
        self.name_list = ["기본값1", "기본값2"] # 웹 페이지에 표시될 초기 이름 목록

        # Flask 라우트 정의
        self._register_routes()

    def _register_routes(self):
        """
        Flask 애플리케이션에 대한 모든 라우트를 등록합니다.
        이 메서드는 __init__에서 호출되어 라우트 설정을 중앙 집중화합니다.
        """
        @self.app.route('/')
        def index():
            """
            웹 애플리케이션의 메인 페이지를 렌더링합니다.
            'index.html' 템플릿을 사용하며, 현재 `self.name_list`를 템플릿에 전달합니다.

            Returns:
                str: 렌더링된 HTML 페이지.
            """
            # self._logger.debug(f"인덱스 페이지 요청됨. 현재 이름 목록: {self.name_list}")
            return render_template('index.html', names=self.name_list)

        @self.app.route('/click', methods=['POST'])
        def click():
            """
            사용자가 웹 페이지에서 특정 항목을 클릭했을 때 호출되는 엔드포인트입니다.
            클릭된 항목의 인덱스를 JSON으로 받아, 이를 `Msg` 객체에 담아 메인 큐로 전송합니다.

            Request JSON Body:
                {"index": <int>} - 클릭된 항목의 인덱스.

            Returns:
                tuple: 성공 시 빈 응답과 HTTP 상태 코드 204 (No Content).
                       오류 발생 시 (예: 잘못된 인덱스) 적절한 오류 응답을 반환할 수 있습니다 (현재는 미구현).
            """
            try:
                data = request.json # 요청 본문에서 JSON 데이터 추출
                clicked_name_index = data['index'] # JSON에서 'index' 키 값 추출
                
                # 클릭된 이름 정보 로깅 (주의: self.name_list 접근 시 IndexError 가능성)
                if 0 <= clicked_name_index < len(self.name_list):
                    clicked_name = self.name_list[clicked_name_index]
                    self._logger.info(f"클릭 이벤트 수신: 인덱스 {clicked_name_index}, 이름 '{clicked_name}'")
                else:
                    self._logger.warn(f"클릭 이벤트 수신: 잘못된 인덱스 {clicked_name_index} (name_list 크기: {len(self.name_list)})")
                    # 잘못된 인덱스에 대한 오류 처리를 추가할 수 있습니다 (예: return jsonify(error="Invalid index"), 400)
                    # 현재는 그대로 진행하여 IndexError를 유발하거나, 아래 Msg 생성에는 문제 없음.

                # 메인 큐로 전달할 Msg 객체 생성
                msg = Msg()
                msg.msg = "mouse" # 메시지 유형을 "mouse"로 설정 (마우스 클릭 이벤트 의미)
                msg.idx = clicked_name_index # 클릭된 항목의 인덱스 설정
                self._main_queue.put(msg) # 메인 큐에 메시지 삽입
                
                return '', 204 # 성공 응답 (내용 없음)
            except TypeError: # request.json이 None일 경우 (Content-Type이 application/json이 아닐 때 등)
                self._logger.error("클릭 이벤트 처리 오류: 요청 데이터가 JSON 형식이 아니거나 없습니다.")
                return jsonify(error="Invalid request data, JSON expected"), 400
            except KeyError: # data['index'] 접근 시 'index' 키가 없을 경우
                self._logger.error("클릭 이벤트 처리 오류: 요청 JSON에 'index' 키가 없습니다.")
                return jsonify(error="Missing 'index' in JSON data"), 400
            except Exception as e: # 기타 예외 처리
                self._logger.error(f"클릭 이벤트 처리 중 예기치 않은 오류 발생: {e}")
                return jsonify(error="An unexpected error occurred"), 500


        @self.app.route('/get_data')
        def get_data():
            """
            웹 페이지에서 주기적으로 호출되어 최신 데이터(이름 목록)를 가져오는 엔드포인트입니다.
            `self._message_queue` (웹 서버가 수신하는 큐)에 새 데이터가 있으면 `self.name_list`를 업데이트합니다.

            Returns:
                Response: 현재 `self.name_list`를 JSON 형식으로 반환합니다.
                          (Flask의 `jsonify`는 Response 객체를 생성합니다)
            """
            # 메시지 큐가 비어있지 않으면, 큐에서 새 이름 목록을 가져와 업데이트
            if not self._message_queue.empty():
                try:
                    # get_nowait() 사용하여 블로킹 없이 가져오기 시도
                    new_name_list = self._message_queue.get_nowait() 
                    self.name_list = new_name_list
                    self._logger.debug(f"웹 서버 데이터 업데이트됨: {self.name_list}")
                except queue.Empty: # 실시간으로 큐가 빌 수도 있음
                    self._logger.debug("데이터 가져오기 시도 중 메시지 큐가 비었습니다 (get_nowait).")
                    pass # 특별한 처리 없이 현재 목록 사용
            
            return jsonify(names=self.name_list) # 현재 이름 목록을 JSON으로 응답
        
        @self.app.route('/shutdown')
        def shutdown():
            """
            웹 서버 및 관련 스레드를 종료하기 위한 엔드포인트입니다.
            이 엔드포인트 호출 시 `self.stop()` (Threadlib의 메서드)이 실행됩니다.
            주의: 이 기능은 보안상 위험할 수 있으므로, 실제 운영 환경에서는 인증/인가 로직이 필요합니다.

            Returns:
                str: 종료 메시지.
            """
            self._logger.info("웹 서버 종료 요청 수신됨. 스레드 중지 시도...")
            self.stop() # Threadlib의 stop 메서드 호출하여 스레드 종료 플래그 설정
            # 실제 Flask 서버 종료는 run 메서드의 app.run()이 반환된 후 또는
            # 외부에서 SIGINT 등을 보내야 할 수 있습니다.
            # 여기서는 스레드 로직 종료만 담당합니다.
            return "웹 서버 종료 명령이 전달되었습니다."

    def run(self):
        """
        Flask 웹 서버를 실행합니다.
        `Threadlib`의 `run`을 오버라이드하며, 부모 클래스의 `run`을 먼저 호출한 후
        Flask 애플리케이션을 시작합니다.
        서버는 '0.0.0.0' 호스트에서 실행되어 모든 네트워크 인터페이스에서 접근 가능합니다.
        디버그 모드와 리로더는 비활성화되어 있습니다.
        """
        super().run() # 부모 클래스(Threadlib)의 run 메서드 호출 (로깅 등)
        try:
            # Flask 앱 실행. use_reloader=False는 Flask가 자체적으로 두 번 실행되는 것을 방지합니다.
            # debug=False는 보안 및 운영 환경을 위해 설정합니다.
            self.app.run(host='0.0.0.0', port=80, debug=False, use_reloader=False) # 포트 80번 명시 (기본 HTTP 포트)
            self._logger.info(f"{self._name}: Flask 애플리케이션 실행 시작됨 (호스트: 0.0.0.0, 포트: 80).")
        except Exception as e:
            self._logger.error(f"{self._name}: Flask 애플리케이션 실행 중 오류 발생: {e}")
        finally:
            self._logger.info(f"{self._name}: Flask 애플리케이션 실행 종료됨.")
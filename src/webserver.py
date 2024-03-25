from flask import Flask, request, render_template, jsonify
import json
import Threadlib
from Msg import Msg

class webserver(Threadlib.Threadlib):
    def __init__(self, main_queue, message_queue):
        super().__init__("Flask server", main_queue, message_queue)
        self.app = Flask(__name__)
        self.name_list = ["default1", "default2"]

        @self.app.route('/')
        def index():
            return render_template('index.html', names=self.name_list)

        @self.app.route('/click', methods=['POST'])
        def click():
            data = request.json
            clicked_name_index = data['index']
            clicked_name = self.name_list[clicked_name_index]
            print('Clicked name:', clicked_name)

            msg = Msg()
            msg.msg = "mouse"
            msg.idx = clicked_name_index
            self._main_queue.put(msg)
            return '', 204

        @self.app.route('/get_data')
        def get_data():
            if not self._message_queue.empty():
                self.name_list = self._message_queue.get()
            return jsonify(names=self.name_list)
        
        @self.app.route('/shutdown')
        def shutdown():
            self.stop()

    def run(self):
        super()
        self.app.run(host='0.0.0.0', debug=True, use_reloader=False)
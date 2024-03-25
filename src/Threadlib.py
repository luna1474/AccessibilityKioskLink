import threading
from loglib import loglib

class Threadlib(threading.Thread):
    def __init__(self, name: str, main_queue, message_queue):
        super(Threadlib, self).__init__()
        self._logger = loglib.get_logger()
        self._stop_event = threading.Event()
        self._main_queue = main_queue
        self._message_queue = message_queue
        self._running = True
        self._name = name
        self._logger.debug(self._name + ": Thread created")

    def run(self):
        self._logger.debug(self._name + ": Thread started")

    def stop(self):
        self._running = False
        self._stop_event.set()
        self._logger.debug(self._name + ": Thread stopped")
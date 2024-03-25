import logging

class loglib:
    # Configure the logging
    logging.basicConfig(filename="logfile.log",
                        format='%(asctime)s %(levelname)s:%(message)s',
                        level=logging.DEBUG,
                        datefmt='%m/%d/%Y %I:%M:%S %p')
    # Get the root logger
    log = logging.getLogger(__name__)

    @staticmethod
    def get_logger():
        return loglib.log

    @staticmethod
    def debug(text: str):
        loglib.log.debug(text)
    
    @staticmethod
    def info(text: str):
        loglib.log.info(text)
    
    @staticmethod
    def warning(text: str):
        loglib.log.warning(text)

    @staticmethod
    def error(text: str):
        loglib.log.error(text)
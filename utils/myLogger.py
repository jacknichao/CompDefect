import logging as root_logging
import os

# Set up the logger

path = os.path.dirname(os.path.realpath(__file__))

logger = root_logging.getLogger()
logger.setLevel(root_logging.INFO)

logger_format = root_logging.Formatter('%(asctime)s %(levelname)s: %(message)s', '%Y-%m-%d %H:%M:%S')

# logging_file_handler = root_logging.FileHandler(path+'/Log.log')
# logging_file_handler.setLevel(root_logging.INFO)
# logging_file_handler.setFormatter(logger_format)
# logger.addHandler(logging_file_handler)

logging_stream_handler = root_logging.StreamHandler()
logging_stream_handler.setLevel(root_logging.INFO)
logging_stream_handler.setFormatter(logger_format)
logger.addHandler(logging_stream_handler)

logging = root_logging

import logging
import logging.config
import yaml
import os

_FILE_HANDLER_CLASSES = [
        'logging.handlers.FileHandler',
        'logging.handlers.RotatingFileHandler',
        'logging.handlers.TimedRotatingFileHandler'
]

def add_human_logger():
    Human_LEVELV_NUM = 100
    logging.addLevelName(Human_LEVELV_NUM, "HUMAN")

    def human(self, message, *args, **kws):
        pass

    logging.Logger.human = human

def setup_logging(execution_data_folder_path, configFilePath='./logger_toolbox/logging.yaml',
                  defaultLevel=logging.INFO,
                  envKey='LOG_CFG'):
    """
    Setup logging configuration
    """

    if os.getenv(envKey, None):
        configFilePath = os.getenv(envKey, None)

    if os.path.exists(configFilePath):
        add_human_logger()

        with open(configFilePath, 'rt') as file:
            config = yaml.safe_load(file.read())
        _prepend_folder_path_to_file_handler(config, execution_data_folder_path)
        _prepend_pid_to_file_handler(config)
        logging.config.dictConfig(config)

    else:
        logging.basicConfig(level=defaultLevel)


def _prepend_folder_path_to_file_handler(config, execution_data_folder_path):

    for handler in config['handlers'].values():
        if handler['class'] in _FILE_HANDLER_CLASSES:
            handler['filename'] = os.path.join(execution_data_folder_path, handler['filename'])


def _prepend_pid_to_file_handler(config):

    for handler in config['handlers'].values():

        if handler['class'] in _FILE_HANDLER_CLASSES:

            filePath = handler['filename']
            folderPath = os.path.join(*os.path.split(filePath)[:-1])
            fileName = os.path.split(filePath)[-1]
            fileNameWithoutExtension = os.path.splitext(fileName)[0]
            fileExtension = os.path.splitext(fileName)[1]

            newFileName = 'pid{}_{}{}'.format(str(os.getpid()), fileNameWithoutExtension, fileExtension)
            newFolderPath = folderPath
            newFilePath = os.path.join(newFolderPath, newFileName)

            assert not os.path.exists(newFilePath)

            handler['filename'] = newFilePath

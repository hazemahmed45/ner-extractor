import os
from logging import Handler, LogRecord
import fsspec


class FileHandler(Handler):
    def __init__(self, user_unique_id: str, logs_dir="./logs"):
        """
        here we create the s3 client object to be use through the project
        """
        super(FileHandler, self).__init__()
        self.user_unique_id = user_unique_id
        self.log_dir = logs_dir
        self.fs: fsspec.AbstractFileSystem = fsspec.core.url_to_fs(self.log_dir)[0]
        if not self.fs.exists(self.log_dir):
            self.fs.makedirs(self.log_dir)
        self.log_filepath = f"{self.log_dir}{self.fs.sep}{self.user_unique_id}.log"

    def emit(self, record: LogRecord) -> None:
        if "user_unique_id" not in record.extra.keys():
            raise Exception(
                'did not pass unique user id to handler to log seperate request, pass the "user_unique_id" as an argument in the logger.log or logger.info'
            )
        user_unique_id = record.extra["user_unique_id"]
        if self.user_unique_id == user_unique_id:
            with fsspec.open(self.log_filepath, mode="a") as f:
                f.writelines(record.msg + "\n")
            return

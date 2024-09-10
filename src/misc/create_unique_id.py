import uuid
from datetime import datetime


def create_unique_user_id():
    return f"{str(uuid.uuid4())}_{datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}"

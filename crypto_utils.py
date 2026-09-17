import os
from dotenv import load_dotenv
load_dotenv()

import os
from cryptography.fernet import Fernet


def get_fernet():
    key = os.environ.get("CREDITFLOW_ENCRYPTION_KEY")
    if not key:
        raise RuntimeError("CREDITFLOW_ENCRYPTION_KEY environment variable is not set.")
    return Fernet(key.encode())

def encrypt_value(value):
    if value is None or value == "":
        return value
    return get_fernet().encrypt(value.encode()).decode()

def decrypt_value(value):
    if value is None or value == "":
        return value
    return get_fernet().decrypt(value.encode()).decode()
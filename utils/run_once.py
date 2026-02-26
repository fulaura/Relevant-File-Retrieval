import os
from functools import wraps

def run_once(flag_file="__run_once_flags__/default.flag"):
    os.makedirs(os.path.dirname(flag_file), exist_ok=True)

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if os.path.exists(flag_file):
                return
            result = func(*args, **kwargs)
            with open(flag_file, 'w') as f:
                f.write("1")
            return result
        return wrapper
    return decorator

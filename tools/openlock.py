import fcntl
import time
import contextlib


@contextlib.contextmanager
def openlock(*args, **kwargs):
    lock = open(*args, **kwargs)
    fcntl.lockf(lock, fcntl.LOCK_EX)
    try:
        yield lock
    finally:
        fcntl.lockf(lock, fcntl.LOCK_UN)
        lock.close()


with openlock('asdf', 'w+') as f:
    print('write')
    f.write('xxx')
    for i in range(10):
        print(i)
        time.sleep(1)

import sys
import time
import threading
from functools import wraps

def with_spinner(message="Thinking...", spinner_chars="|/-\\", refresh=0.1):
    """
    Decorator that shows a spinner and elapsed time while the wrapped function runs.
    Spinner is cleared when finished or on Ctrl+C.
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            stop_event = threading.Event()
            seconds_elapsed = 0
            last_tick = time.time()

            def spin():
                nonlocal seconds_elapsed, last_tick
                i = 0
                while not stop_event.is_set():
                    now = time.time()
                    if now - last_tick >= 1:
                        seconds_elapsed += 1
                        last_tick = now
                    char = spinner_chars[i % len(spinner_chars)]
                    sys.stdout.write(f"\r{char} {message} ({seconds_elapsed} seconds)")
                    sys.stdout.flush()
                    time.sleep(refresh)
                    i += 1
                # Clear the spinner line
                sys.stdout.write("\r" + " " * 60 + "\r")
                sys.stdout.flush()

            thread = threading.Thread(target=spin, daemon=True)
            thread.start()

            try:
                return func(*args, **kwargs)
            except KeyboardInterrupt:
                raise  # allow Ctrl+C to propagate
            finally:
                stop_event.set()
                thread.join()

        return wrapper

    return decorator

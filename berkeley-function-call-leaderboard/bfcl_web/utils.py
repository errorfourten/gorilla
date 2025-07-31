import sys
import time
import threading
from functools import wraps
from typing import List

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

class ModelStatusTracker:
    """
    Tracks the status of multiple model inference processes and displays which models
    are still waiting for responses.
    """
    def __init__(self, models: List[str], refresh=0.5, spinner_chars="|/-\\"):
        self.models = set(models)
        self.pending_models = set(models)
        self.completed_models = set()
        self.stop_event = threading.Event()
        self.seconds_elapsed = 0
        self.last_tick = time.time()
        self.refresh = refresh
        self.spinner_chars = spinner_chars
        self.thread = None
    
    def mark_completed(self, model_name: str):
        """Mark a model as having completed its inference."""
        if model_name in self.pending_models:
            self.pending_models.remove(model_name)
            self.completed_models.add(model_name)
            
            # Force an immediate display update
            if not self.stop_event.is_set() and self.thread and self.thread.is_alive():
                char = self.spinner_chars[0]  # Use first spinner char for update
                pending_str = ", ".join(sorted(self.pending_models)) if self.pending_models else "None"
                completed_str = ", ".join(sorted(self.completed_models))
                status = f"{char} Waiting on: {pending_str} | Completed: {completed_str} ({self.seconds_elapsed}s)"
                # Ensure the line is cleared before writing
                sys.stdout.write("\r" + " " * 120 + "\r")
                sys.stdout.write(status.ljust(120))
                sys.stdout.flush()
    
    def start(self):
        """Start the status tracking display."""
        self.thread = threading.Thread(target=self._display_status, daemon=True)
        self.thread.start()
    
    def stop(self):
        """Stop the status tracking display."""
        self.stop_event.set()
        if self.thread:
            self.thread.join()
        # Clear the status line completely
        sys.stdout.write("\r" + " " * 120 + "\r")  # Use a larger number to ensure full line clearing
        sys.stdout.flush()
        # Move cursor to the beginning of the line
        sys.stdout.write("\r")
        sys.stdout.flush()
    
    def _display_status(self):
        """Display the current status of model responses."""
        i = 0
        max_line_length = 0
        
        while not self.stop_event.is_set():
            if not self.pending_models:
                # All models have completed
                break
                
            now = time.time()
            if now - self.last_tick >= 1:
                self.seconds_elapsed += 1
                self.last_tick = now
                
            char = self.spinner_chars[i % len(self.spinner_chars)]
            pending_str = ", ".join(sorted(self.pending_models))
            completed_str = ", ".join(sorted(self.completed_models)) if self.completed_models else "None"
            
            # Create status message
            status = f"{char} Waiting on: {pending_str} | Completed: {completed_str} ({self.seconds_elapsed}s)"
            
            # Update max line length seen
            max_line_length = max(max_line_length, len(status) + 5)  # Add padding
            
            # Move cursor to beginning of line, print status, and pad with spaces to clear previous content
            sys.stdout.write("\r" + status.ljust(max_line_length))
            sys.stdout.flush()
            
            time.sleep(self.refresh)
            i += 1

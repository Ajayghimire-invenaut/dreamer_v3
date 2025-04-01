"""
Parallelization module.
Provides classes to run environments in separate processes for DreamerV3.
"""

import atexit
import enum
import os
import sys
import time
import traceback
import multiprocessing
import cloudpickle
from functools import partial
from typing import Any, Dict, Optional

class Message(enum.Enum):
    OK = 1
    CALLABLE = 2
    CALL = 3
    READ = 4
    STOP = 5
    ERROR = 6

class Worker:
    def __init__(self, function: Any, strategy: str = "spawn", state: bool = False) -> None:
        """
        Initialize a worker process for parallel execution.

        Args:
            function: The function to execute in the worker process
            strategy: Multiprocessing context strategy ('spawn' recommended for DreamerV3)
            state: Whether the worker maintains state (True for environments)
        """
        self.context = multiprocessing.get_context(strategy)
        self.parent_pipe, child_pipe = self.context.Pipe(duplex=True)
        serialized_function = cloudpickle.dumps(function)
        self.process = self.context.Process(
            target=self._loop,
            args=(child_pipe, serialized_function),
            daemon=True
        )
        self.process.start()
        self.next_id = 0
        self.results: Dict[int, Any] = {}
        atexit.register(self.close)

    def __call__(self, message: Message, payload: Optional[Any] = None) -> Any:
        """
        Send a message to the worker process and return a Future for the result.

        Args:
            message: The type of message to send
            payload: Data to send with the message

        Returns:
            Future object representing the result
        """
        call_id = self.next_id
        self.next_id += 1
        try:
            self.parent_pipe.send((message, call_id, payload))
        except Exception as error:
            raise RuntimeError(f"Failed to send message to worker: {error}")
        return Future(self._receive, call_id)

    def _receive(self, call_id: int) -> Any:
        """
        Receive and process results from the worker process.

        Args:
            call_id: Identifier for the call to retrieve

        Returns:
            Result from the worker process
        """
        while call_id not in self.results:
            try:
                if not self.parent_pipe.poll(1.0):  # Increased timeout for stability
                    continue
                message, cid, payload = self.parent_pipe.recv()
            except (OSError, EOFError):
                raise RuntimeError("Lost connection to worker process.")
            if message == Message.ERROR:
                raise Exception(payload)
            self.results[cid] = payload
        return self.results.pop(call_id)

    def _loop(self, pipe: Any, serialized_function: bytes) -> None:
        """
        Main loop for the worker process.

        Args:
            pipe: Communication pipe to the parent process
            serialized_function: Serialized function to execute
        """
        try:
            function = cloudpickle.loads(serialized_function)
            while True:
                if not pipe.poll(0.1):
                    continue
                message, call_id, payload = pipe.recv()
                if message == Message.STOP:
                    break
                try:
                    args, kwargs = payload if payload is not None else ([], {})
                    state, result = function(None, message, *args, **kwargs)
                    pipe.send((Message.OK, call_id, result))
                except Exception as error:
                    tb = traceback.format_exception(*sys.exc_info())
                    pipe.send((Message.ERROR, call_id, f"Worker error: {str(error)}\n{''.join(tb)}"))
                    break
        except Exception as error:
            pipe.send((Message.ERROR, 0, f"Worker initialization failed: {str(error)}"))
        finally:
            pipe.close()

    def close(self) -> None:
        """Cleanly shut down the worker process."""
        try:
            self.parent_pipe.send((Message.STOP, self.next_id, None))
            self.parent_pipe.close()
        except Exception:
            pass
        try:
            self.process.join(1.0)  # Increased join timeout for reliability
            if self.process.exitcode is None:
                os.kill(self.process.pid, 9)
        except Exception:
            pass

class Future:
    def __init__(self, receive: Any, call_id: int) -> None:
        """
        Represent a future result from a worker process.

        Args:
            receive: Function to retrieve the result
            call_id: Identifier for the call
        """
        self.receive = receive
        self.call_id = call_id
        self.complete = False
        self.result = None

    def __call__(self) -> Any:
        """
        Retrieve the result, blocking until available.

        Returns:
            The result from the worker process
        """
        if not self.complete:
            self.result = self.receive(self.call_id)
            self.complete = True
        return self.result

class Parallel:
    def __init__(self, constructor: Any, strategy: str = "spawn"):
        """
        Manage a parallel environment or object instance.

        Args:
            constructor: Callable to create the object (e.g., SingleEnvironment)
            strategy: Multiprocessing context strategy ('spawn' for DreamerV3)
        """
        self.worker = Worker(partial(self._respond, constructor), strategy, state=True)
        self.callable_cache: Dict[str, Any] = {}

    def __getattr__(self, attribute_name: str) -> Any:
        """
        Dynamically access attributes or methods of the parallel object.

        Args:
            attribute_name: Name of the attribute or method to access

        Returns:
            Callable or value depending on the attribute type
        """
        if attribute_name.startswith("_"):
            raise AttributeError(attribute_name)
        if attribute_name not in self.callable_cache:
            self.callable_cache[attribute_name] = self.worker(Message.CALLABLE, attribute_name)()
        if self.callable_cache[attribute_name]:
            return partial(self.worker, Message.CALL, attribute_name)
        else:
            return self.worker(Message.READ, attribute_name)()

    def close(self) -> None:
        """Shut down the parallel worker."""
        self.worker.close()

    @staticmethod
    def _respond(constructor: Any, current_state: Any, message: Message, attribute_name: str, *args, **kwargs) -> tuple:
        """
        Handle messages in the worker process.

        Args:
            constructor: Callable to initialize the state
            current_state: Current state of the object
            message: Type of message to process
            attribute_name: Attribute or method name
            *args: Positional arguments for method calls
            **kwargs: Keyword arguments for method calls

        Returns:
            Tuple of (updated_state, result)
        """
        current_state = current_state or constructor()
        if message == Message.CALLABLE:
            result = callable(getattr(current_state, attribute_name))
        elif message == Message.CALL:
            result = getattr(current_state, attribute_name)(*args, **kwargs)
        elif message == Message.READ:
            result = getattr(current_state, attribute_name)
        return current_state, result
"""
Parallelization module.
Provides classes to run environments in separate processes.
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
    def __init__(self, function: Any, strategy: str = "thread", state: bool = False) -> None:
        self.context = multiprocessing.get_context("spawn")
        self.parent_pipe, child_pipe = self.context.Pipe()
        serialized_function = cloudpickle.dumps(function)
        self.process = self.context.Process(target=self._loop, args=(child_pipe, serialized_function), daemon=True)
        self.process.start()
        self.next_id = 0
        self.results: Dict[int, Any] = {}
        atexit.register(self.close)

    def __call__(self, message: Message, payload: Optional[Any] = None) -> Any:
        call_id = self.next_id
        self.next_id += 1
        self.parent_pipe.send((message, call_id, payload))
        return Future(self._receive, call_id)

    def _receive(self, call_id: int) -> Any:
        while call_id not in self.results:
            try:
                message, cid, payload = self.parent_pipe.recv()
            except (OSError, EOFError):
                raise RuntimeError("Lost connection to worker.")
            if message == Message.ERROR:
                raise Exception(payload)
            self.results[cid] = payload
        return self.results.pop(call_id)

    def _loop(self, pipe: Any, serialized_function: bytes) -> None:
        function = cloudpickle.loads(serialized_function)
        while True:
            if not pipe.poll(0.1):
                continue
            message, call_id, payload = pipe.recv()
            if message == Message.STOP:
                return
            try:
                args, kwargs = payload if payload is not None else ([], {})
                state, result = function(None, message, *args, **kwargs)
                pipe.send((Message.OK, call_id, result))
            except Exception:
                tb = traceback.format_exception(*sys.exc_info())
                pipe.send((Message.ERROR, call_id, "".join(tb)))
                return

    def close(self) -> None:
        try:
            self.parent_pipe.send((Message.STOP, self.next_id, None))
            self.parent_pipe.close()
        except Exception:
            pass
        try:
            self.process.join(0.1)
            if self.process.exitcode is None:
                os.kill(self.process.pid, 9)
        except Exception:
            pass

class Future:
    def __init__(self, receive: Any, call_id: int) -> None:
        self.receive = receive
        self.call_id = call_id
        self.complete = False
        self.result = None

    def __call__(self) -> Any:
        if not self.complete:
            self.result = self.receive(self.call_id)
            self.complete = True
        return self.result

class Parallel:
    def __init__(self, constructor: Any, strategy: str):
        self.worker = Worker(partial(self._respond, constructor), strategy, state=True)
        self.callable_cache: Dict[str, Any] = {}

    def __getattr__(self, attribute_name: str) -> Any:
        if attribute_name.startswith("_"):
            raise AttributeError(attribute_name)
        if attribute_name not in self.callable_cache:
            self.callable_cache[attribute_name] = self.worker(Message.CALLABLE, attribute_name)()
        if self.callable_cache[attribute_name]:
            from functools import partial
            return partial(self.worker, Message.CALL, attribute_name)
        else:
            return self.worker(Message.READ, attribute_name)()

    def close(self) -> None:
        self.worker.close()

    @staticmethod
    def _respond(constructor: Any, current_state: Any, message: Message, attribute_name: str, *args, **kwargs) -> tuple:
        current_state = current_state or constructor
        if message == Message.CALLABLE:
            result = callable(getattr(current_state, attribute_name))
        elif message == Message.CALL:
            result = getattr(current_state, attribute_name)(*args, **kwargs)
        elif message == Message.READ:
            result = getattr(current_state, attribute_name)
        return current_state, result

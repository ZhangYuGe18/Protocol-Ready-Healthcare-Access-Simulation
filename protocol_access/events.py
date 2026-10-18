from dataclasses import dataclass, field
import heapq
import itertools


@dataclass(order=True)
class Event:
    time: float
    priority: int
    sequence: int
    kind: str = field(compare=False)
    agent: int = field(compare=False)


class EventCalendar:
    """Small deterministic event calendar used by model experiments."""

    def __init__(self):
        self._heap = []
        self._sequence = itertools.count()

    def schedule(self, time, kind, agent, priority=0):
        event = Event(float(time), int(priority), next(self._sequence), kind, int(agent))
        heapq.heappush(self._heap, event)
        return event

    def pop(self):
        return heapq.heappop(self._heap)

    def __bool__(self):
        return bool(self._heap)


@dataclass
class ResourcePool:
    capacity: int

    def __post_init__(self):
        if int(self.capacity) < 1:
            raise ValueError("Resource capacity must be positive")
        self.available = [0.0] * int(self.capacity)

    def acquire(self, request_time, duration):
        index = min(range(len(self.available)), key=self.available.__getitem__)
        start = max(float(request_time), float(self.available[index]))
        end = start + max(0.0, float(duration))
        self.available[index] = end
        return start, end

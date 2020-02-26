"""A simple queue to count continuous true elements
"""
from collections import deque


class DetectQueue:
    """A FIFO queue that simulates a filter
    """

    def __init__(self, maxlen=5):
        self.queue = deque()
        self.maxsize = maxlen
        self.size = 0
        self.count = 0  # max streak

    def append(self, val):
        self.queue.append(val)
        self.size += 1

        if self.size > self.maxsize:
            self.queue.popleft()
            self.size -= 1

        self.count = self.count + 1 if val else 0
        self.count = min(self.count, self.maxsize)

    def clear(self):
        self.queue.clear()
        self.count = 0
        self.size = 0

    def __repr__(self):
        return self.queue.__repr__()


if __name__ == "__main__":
    pass

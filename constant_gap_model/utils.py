import os
import sys

class Logger:
    def __init__(self, path, mode="w", write_to_terminal=True):
        assert mode in {"w", "a"}, "unknown mode for logger %s" % mode
        self.terminal = sys.stdout
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))
        if mode == "w" or not os.path.exists(path):
            self.log = open(path, "w")
        else:
            self.log = open(path, "a")

        self.write_to_terminal = write_to_terminal

    def write(self, message):
        if self.write_to_terminal:
            self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        # for python 3 compatibility.
        pass

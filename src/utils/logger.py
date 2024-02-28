from typing import *

import sys
import time


class AverageAggregator(object):
    def __init__(self):
        self._value = 0.
        self._count = 0

    @property
    def value(self):
        return self._value / self._count

    @value.setter
    def value(self, val: float):
        self._value += val
        self._count += 1

    def update(self, val: float, n=1):
        self._value += val
        self._count += n


class StatsLogger(object):
    __INSTANCE = None

    def __init__(self):
        if StatsLogger.__INSTANCE is not None:
            raise RuntimeError("StatsLogger should not be directly created")

        self._values = dict()
        self._loss = AverageAggregator()
        self._output_files = [sys.stdout]

    def add_output_file(self, f):
        self._output_files.append(f)

    @property
    def loss(self):
        return self._loss.value

    @loss.setter
    def loss(self, val: float):
        self._loss.value = val

    def update_loss(self, val: float, n=1):
        self._loss.update(val, n)

    def __getitem__(self, key: str):
        if key not in self._values:
            self._values[key] = AverageAggregator()
        return self._values[key]

    def clear(self):
        self._values.clear()
        self._loss = AverageAggregator()
        for f in self._output_files:
            if f.isatty():  # if the file stream is interactive
                print(file=f, flush=True)

    def print_progress(self, epoch: Union[int, str], iter: int, precision="{:.5f}"):
        fmt = "[{}] [epoch {:4d} iter {:3d}] | loss: " + precision
        msg = fmt.format(time.strftime("%Y-%m-%d %H:%M:%S"), epoch, iter, self._loss.value)
        for k,  v in self._values.items():
            msg += " | " + k + ": " + precision.format(v.value)
        for f in self._output_files:
            if f.isatty():  # if the file stream is interactive
                print(msg + "\b"*len(msg), end="", flush=True, file=f)
            else:
                print(msg, flush=True, file=f)

    @classmethod
    def instance(cls):
        if StatsLogger.__INSTANCE is None:
            StatsLogger.__INSTANCE = cls()
        return StatsLogger.__INSTANCE

"""Crude memory usage watcher."""

from array import array
import os
import multiprocessing
from tempfile import TemporaryFile
import time

from hostess.subutils import piped
import psutil


def dump_memlog(pid, templog, signal, interval=0.005):
    proc = psutil.Process(pid)
    with templog as stream:
        while True:
            if signal.value != 0:
                return
            stream.write(f"{proc.memory_info().rss}\n")
            time.sleep(interval)


def flush_memlog(templog, basemem):
    with templog as stream:
        lines = tuple(stream.readlines())
    templog.close()
    if len(lines) > 0:
        return max(map(int, lines[1:])) - basemem
    return float('nan')


memdump = piped(dump_memlog, block=False)


class Memwatcher:
    def __init__(self, pid, fake: bool = False):
        self.fake = fake
        self.pid, self.cache = pid, array('f', [float('nan')] * 2048)
        self.proc, self.cacheix = psutil.Process(pid), 0

    def __enter__(self):
        if self.fake is True:
            return
        self.basemem = psutil.Process(self.pid).memory_info().rss
        self.memlog = TemporaryFile(mode="a+")
        self.manager = multiprocessing.Manager()
        self.signal = self.manager.Value('sig', 0)
        self.memproc = multiprocessing.Process(
            target=dump_memlog, args=(self.pid, self.memlog, self.signal)
        )
        self.memproc.start()

    def __exit__(self, *_, **__):
        if self.fake is True:
            return
        if self.memproc is not None:
            self.signal.value = 1
            while self.memproc.is_alive():
                time.sleep(0.01)
            self.cacheix += 1
            self.cache[self.cacheix] = flush_memlog(self.memlog, self.basemem)
        self.memproc, self.signal, self.manager = None, None, None

    @property
    def last(self):
        if self.cacheix == 0:
            return float('nan')
        return self.cache[self.cacheix - 1]

    basemem = None
    memproc = None

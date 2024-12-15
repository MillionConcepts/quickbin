"""Crude memory usage watcher."""

from array import array
import multiprocessing
import time

import psutil


def dump_memlog(pid, signal, pipe, interval=0.01):
    proc = psutil.Process(pid)
    cache = []
    while True:
        if signal.value != 0:
            break
        cache.append(proc.memory_info().rss)
        time.sleep(interval)
    pipe.send(max(cache))


class Memwatcher:
    def __init__(self, pid, fake: bool = False):
        self.fake = fake
        self.pid, self.cache = pid, array('f', [float('nan')] * 2048)
        self.proc, self.cacheix = psutil.Process(pid), 0

    def __enter__(self):
        if self.fake is True:
            return
        self.basemem = psutil.Process(self.pid).memory_info().rss
        self.manager = multiprocessing.Manager()
        self.signal = self.manager.Value('sig', 0)
        self.pipe, pipe_terminus = multiprocessing.Pipe()
        self.memproc = multiprocessing.Process(
            target=dump_memlog, args=(self.pid, self.signal, pipe_terminus)
        )
        self.memproc.start()

    def __exit__(self, *_, **__):
        if self.fake is True:
            return
        if self.memproc is not None:
            self.signal.value = 1
            self.memproc.join()
            try:
                self.cache[self.cacheix] = self.pipe.recv()
            except EOFError:
                pass
            self.cacheix += 1
        self.memproc, self.signal, self.pipe = None, None, None

    @property
    def last(self):
        if self.cacheix == 0:
            return float('nan')
        return self.cache[self.cacheix - 1]

    basemem = None
    memproc = None
    pipe = None

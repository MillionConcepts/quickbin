from enum import Enum
from types import MappingProxyType as MPt

class Ops(Enum):
    count = 2
    sum = 4
    mean = 8
    std = 16
    median = 32
    min = 64
    max = 128

OPS = MPt(Ops.__members__)

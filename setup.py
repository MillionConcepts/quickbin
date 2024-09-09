from setuptools import Extension, setup

import numpy as np

_quickbin_core = Extension("quickbin._quickbin_core", sources=["quickbin/_quickbin_core.c"])

setup(
    ext_modules=[_quickbin_core],
    include_dirs=[np.get_include()],
)

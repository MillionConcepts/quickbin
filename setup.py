from setuptools import Extension, setup

import numpy as np

quickbin_core = Extension(
    "quickbin.quickbin_core",
    sources=[
        "quickbin/binning.c",
        "quickbin/iterators.c",
        "quickbin/quickbin_core.c",
    ]
)

setup(
    ext_modules=[quickbin_core],
    include_dirs=[np.get_include()],
)

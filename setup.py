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

quickbin_test_utils = Extension(
    "quickbin.tests.quickbin_test_utils",
    sources = ["quickbin/tests/quickbin_test_utils.c"]
)

setup(
    ext_modules=[quickbin_core, quickbin_test_utils],
    include_dirs=[np.get_include()],
)

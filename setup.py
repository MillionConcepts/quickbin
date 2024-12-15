from setuptools import Extension, setup

import numpy as np

quickbin_core = Extension(
    "quickbin._quickbin_core",
    sources=[
        "quickbin/binning.c",
        "quickbin/iterators.c",
        "quickbin/_quickbin_core.c",
    ]
)

quickbin_test_utils = Extension(
    "quickbin.tests._quickbin_test_utils",
    sources = ["quickbin/tests/_quickbin_test_utils.c"]
)

setup(
    ext_modules=[quickbin_core, quickbin_test_utils],
    include_dirs=[np.get_include()],
)

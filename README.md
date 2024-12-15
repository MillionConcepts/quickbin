# quickbin

This library provides a generalized histogram function, `bin2d()`, that is
intended as a nearly drop-in replacement for `binned_statistic_2d()` from
`scipy.stats`. It has 5-20x better performance for common tasks like 
calculating the binned mean of a large set of ungridded data samples, and
up to ~100x better in the best cases. 

## example of usage

```
import matplotlib.pyplot as plt
import numpy as np

from quickbin import bin2d()

y_coords = np.random.random(1000000)
x_coords = np.random.random(1000000)
temperatures = (np.sin(y_coords * x_coords * 8 * np.pi) + 2) * 100 

result = bin2d(y_coords, x_coords, temperatures, ("mean", "std"), n_bins=50)
bin_means, bin_stdevs = result["mean"], result["std"]

plt.imshow(temperatures, cmap='inferno')
```

## installation and dependencies

`quickbin` requires `setuptools` and `numpy`. Install it from source with
`python setup.py .` in the source root directory.

## tests

`quickbin` has a small test suite, which additionally depends on 
`pytest`. Run it by executing `pytest` in the source root directory.

## benchmarks

`quickbin` includes a submodule for benchmarking its time and memory 
performance against `binned_statistic_2d()`. It additionally depends on 
`scipy` and `psutil`. `notebooks/benchmark.ipynb` gives examples of usage.

## licensing

Pre-release code. May crash your computer. Not licensed for distribution.

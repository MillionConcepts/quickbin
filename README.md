# quickbin

This library provides a generalized histogram function, `bin2d()`, that is
intended as a nearly drop-in replacement for `binned_statistic_2d()` from
`scipy.stats`. It has 5-30x better performance for common tasks like 
calculating the binned mean of a large set of ungridded data samples, and
~100x better in the best cases. It also uses 2-3x less memory in most
cases, permitting the use of larger arrays and greater parallelism.

For instance, computing the binned mean and standard deviation of 
200000000 points on a 1000 x 1000 grid with two calls to 
`binned_statistic_2d()` takes a total of ~60 seconds with peak memory 
pressure of ~14.5 GB; a single call to `bin2d()` does it  in ~4 seconds 
with peak memory pressure of ~5 GB.

## example of use

```
import matplotlib.pyplot as plt  # note: not a quickbin dependency
import numpy as np

from quickbin import bin2d

# synthetic data representing measured emission intensity.
# 10000000 samples, averaging over a 500 x 500 grid. 
y_coords = np.random.random(10000000)
x_coords = np.random.random(10000000)
n_bins = 500
# add some spatial dependence
centered_distance = np.sqrt(
    (0.5 - x_coords) ** 2 + (0.5 - y_coords) ** 2
)
intensity = np.cos(centered_distance * np.pi * 4)
# add some spatially-biased noise
intensity += (
    (np.random.poisson(size=10000000) - 0.5) * abs(x_coords - 0.5)
)

result = bin2d(y_coords, x_coords, intensity, ("mean", "std"), n_bins=500)
intensity_mean, intensity_stdev = result["mean"], result["std"]

fig, axes = plt.subplots(1, 2)
axes[0].imshow(intensity_mean, cmap='inferno')
axes[1].imshow(intensity_stdev, cmap='winter')
axes[0].set_axis_off(), axes[1].set_axis_off()
fig.tight_layout(pad=0)
```

![quickbin example image](assets/quickbin_example.jpg)

## installation and dependencies

`quickbin` requires `setuptools` and `numpy`. Install it from source with
`pip install .` in the source root directory.

## tests

`quickbin` has a small test suite, which additionally depends on 
`pytest`. Run it by executing `pytest` in the source root directory.

## benchmarks

`quickbin` includes a submodule for benchmarking its time and memory 
performance against `binned_statistic_2d()`. It additionally depends on 
`scipy` and `psutil`. `notebooks/benchmark.ipynb` gives examples of usage.

## licensing

`quickbin` is subject to the BSD 3-Clause License, copyright of Million Concepts.
You may do almost anything you like with the code.

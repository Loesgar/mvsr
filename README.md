# Multi-Variant Segmented Regression

{TODO: Brief description goes here}

{TODO: Picture goes here}

{TODO: Citations go here}

## Overview

This library enables segmented regression (also referred to as piecewise regression or min-$\epsilon$ segmented regression). It generates a segmented regression function, based on models and input samples. The input samples consist of (typically noise-free) independent variables (x) and noisy dependent variables (y).

Contrary to other approaches, we do not only try to minimize the average prediction error, but also aim to accurately place the breakpoints (i.e. the start and end positions of the individual segments). This enables qualitative analysis of the breakpoints itself and can show at which exact point the behavior of the underlying data distribution changes, which is especially useful for time-series analysis.

The library is accurate and very fast, enabling the analysis of millions of samples in less than a second on a typical computer.

### Features

The following features are actively supported by this library:

- Segmented Regression using Multiple Algorithms:
  1. Fast and Accurate Heuristic (Default, $n\log{n}$ Runtime, Results Equal or **Very Close** to Optimal Solution)
  2. Exact Solution (Dynamic Program, $n^2$ Runtime, Guaranteed Minimal MSE Possible)
- Fixed Number of Segments
- Automatically Deduce the Number of Segments
- Multiple Input Dimensions
- Multiple Variants (i.e. Multiple Output Dimensions with Shared Breakpoints)

### Project Scope

The focus of this project is limited to the mentioned features. Especially the following featueres are currently considered out-of-scope.

#### Continuous Output Functions

We do not support enforcing continouity on the output regression function. While this restriction increases the mathematical complexity, we do not see any advantage. If the input data is continuous and the model fits the underlying data distribution, the result will be close to continuous anyways. If the input data is not continuous, it does not seem useful to enforcing continouity. Doing so will only increase the prediction error. Still, designing a postfix to enforce continoutiy is possible and would likely result in a very fast heuristic. If you have a reasonable use-case, consider creating a feature request or contact me directly.

Interpolation between two segments (over a non-continuous breakpoint) can be useful when predicting values, especially if there is a larger gap between the two neighbouting samples. This is supported in some languages (e.g. [Python](lang/python/README.md#Interpolation)).

#### Min-# Segmented Regression

Defining a fixed upper error limit and place as many segments as needed is not supported. While it is possible to achieve something like this using the algorithms, there are more effective algorithms to do this.

#### Online Analysis

Online analysis is currently not supported, although this might change in the future for automatically deduced segment counts.

However, depending on the number of samples and the speed at which new samples arrive, it can already be feasible to run the complete analysis step each time new data is gathered or when the new data does not fit to the previous regression.

#### Multidimensional Breakpoints

We do not support segmentation across multiple input dimensions (yet). Currently the data is only segmented along the input order. This means if there is not breakpoint placed in between two consecutive samples, the samples will belong to the same segment.

## Installation and Usage

While the algorithms are implemented in C++, the interface is exposed via a C-API and there are bindings for multiple languages, often already installable with the corresponding package managers. The exact usage differs between the languages. We currently support the following languages (see the individual documentation for usage instructions):

- [C](lang/c/README.md)
- [C++](lang/cpp/README.md)
- [Julia](lang/julia/README.md)
- [Python](lang/python/README.md)
- [R/Rlang](lang/rlang/README.md)
- [Rust](lang/rust/README.md)

<!--end-docs-->
## License and Contribution

The whole project is licensed under Mozilla Public License (MPL 2.0). This enables everyone to use, link and share this work (see [LICENSE](LICENSE)). We strongly encourage you to share any changes and contribute to this project directly.

If you are publishing a paper and use this library (e.g. to analyze your data), please consider to cite the following papers. They describe the analysis algorithm in more detail and evaluate the resulting speed and accuracy.

{TODO link cff file accordingly}

## Thanks

Special thanks for supporting this work goes to:

{TODO links and/or logos}

- SFB-FONDA (Collaborative Research Center CRC-1404)
- DFG (German Research Foundation)
- Technical University of Darmstadt
- Humboldt-University of Berlin

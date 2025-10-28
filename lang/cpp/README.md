# C++ (Native)

<!--hide-in-docs-->
This describes the native C++ version of the MVSR-project. Consider reading the [general documentation](../../README.md).

If you use this project to analyze you data, consider [citing our paper](../../README.md#license-and-contribution).

The C++ part is the native programming language for the mvsr library.
It is the most flexible, but also the most low-level API.
It allows you to use your own data-type and change the matrix functions, which can speedup the calculation for exceedingly large number of dimensions or variants.
If this is not needed, it is probably easier to use the [C-API](../c/README.md) from C++.

## Installation

No installation is needed, only the header files inside the `mvsr/src` folder.
While the `CMakeLists.txt` file inside the `mvsr` folder contains a project definition and additional compile options, the library is basically a header-only library.

## Usage

By including the `mvsr.hpp` file, the library provides a Regression class template, that can be instantiatet using a certain data type.
The data type is used to represent a single scalar value, typically `float`, `double`, or `long double`, but can also be a [custom data type](#custom-data-type).

After instantiating an object, it is possible to use the member functions.
They are described in the [C++ API reference](https://loesgar.github.io/mvsr/stable/lang/cpp/api-reference.html) and similar to the low-level C-API.
The exact usage can be seen in the implementation of the [C interface](../../mvsr/src/mvsr.cpp).

The typical steps are described in the [C-API documentation](../c/README.md#usage) and can be seen in the C helper functions in the `mvsr/inc/mvsr.h` file.

## Advanced Usage

The following features are only possible using the C++ API.

### Custom Data Type

If you do not want to use floats or doubles, you can instantiate the Regression class template with your own data type.

The type must be an arithmetic type, it must be able to do basic algebra operations and must be copyable.
This can enable the usage of smaller or larger floating point numbers, dynamically sized foats or rational number types in order to increase accuracy or improve performance.

### Custom Matrix Calculations

By default we provide custom algebra functions.
This has multiple reasons (e.g. special use cases, easier licensing, etc).
If you want to change these functions, simply replace the `mvsr_mat.hpp` file.
Especially solving the equation system can be sped-up significally using libraries like [Eigen](https://eigen.tuxfamily.org/).

Notice that this is only relevant for large numbers of dimensions and variants.
The best performing method depends on the data, the desired accuracy and the used hardware.

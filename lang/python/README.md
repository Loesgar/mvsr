# Python API

<!--hide-in-docs-->
This describes the Python API of the MVSR-project. Consider reading the [general documentation](../../README.md).

## Installation

## Usage

### Advanced Usage

<!--replace-in-docs ### [API Reference](project:./api-reference.rst) -->
### [API Reference](https://loesgar.github.io/msvr/python/api-reference)

### Internals

Data Preprocessing:

```mermaid
flowchart TD;
    X["X (Independent Variables)"]-->Regression;
    X-->Kernel[Kernel]-->LibMvsr["**LibMvsr**"];
    Y["Y (Dependent Variables)"]-->Regression;

    Y-->YNorm;
    subgraph KO["Kernel Object"];
    subgraph MV["Multivariant Operation"];
    YNorm[Variant Normalization]-->YWeighting["Variant Weighting"];
    Unweighting["Variant Unweighting"]-->Denorm["Variant Denormalization"];
    end
    Kernel;
    end

    LibMvsr-->Breakpoints-->Regression;
    LibMvsr-->Models-->Unweighting;
    Denorm-->Regression;
    YWeighting-->LibMvsr;
```

## Interpolation

Test text

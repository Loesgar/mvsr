# Python (MVSR)

This describes the python version of the MVSR-project. Consider reading the [general documentaion](../../README.md).

## Installation

## Usage

### Special Use-Cases

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

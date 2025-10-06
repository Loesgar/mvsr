=========================
MVSR Python API Reference
=========================

.. _numpy.typing.ArrayLike: https://numpy.org/doc/stable/reference/typing.html#numpy.typing.ArrayLike
.. _numpy.float32: https://numpy.org/doc/stable/reference/arrays.scalars.html#numpy.float32
.. _numpy.float64: https://numpy.org/doc/stable/reference/arrays.scalars.html#numpy.float64

-----------------------------
High-level Function Interface
-----------------------------

.. autofunction:: mvsr.mvsr

-------
Classes
-------

.. autoclass:: mvsr.Regression
.. autoclass-qualname:: mvsr.Kernel.Raw
    :no-members:

    .. automethod:: mvsr.Kernel.Raw.__call__
    .. automethod:: mvsr.Kernel.Raw.normalize
    .. automethod:: mvsr.Kernel.Raw.denormalize

.. autoclass-qualname:: mvsr.Kernel.Poly
    :show-inheritance:

    .. automethod:: mvsr.Kernel.Poly.__call__
    .. automethod:: mvsr.Kernel.Poly.interpolate

-----
Enums
-----

.. autoclass:: mvsr.Algorithm

.. autoclass:: mvsr.Interpolate

.. autoclass:: mvsr.Metric

.. autoclass:: mvsr.Score

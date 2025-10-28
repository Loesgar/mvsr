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

    .. automethod:: mvsr.Regression.__call__
    .. automethod:: mvsr.Regression.__len__

.. autoclass:: mvsr.Segment

    .. automethod:: mvsr.Segment.__call__

^^^^^^^
Kernels
^^^^^^^

.. autoclass-qualname:: mvsr.Kernel.Raw
    :no-members:

    .. automethod:: mvsr.Kernel.Raw.__call__
    .. automethod:: mvsr.Kernel.Raw.normalize
    .. automethod:: mvsr.Kernel.Raw.denormalize
    .. automethod:: mvsr.Kernel.Raw.interpolate

.. autoclass-qualname:: mvsr.Kernel.Poly
    :show-inheritance:

    .. automethod:: mvsr.Kernel.Poly.__call__

^^^^^
Enums
^^^^^

.. autoclass:: mvsr.Algorithm
.. autoclass:: mvsr.Score

-------------
Interpolation
-------------

.. autofunction-qualname:: mvsr.Interpolate.left
.. autofunction-qualname:: mvsr.Interpolate.right
.. autofunction-qualname:: mvsr.Interpolate.closest
.. autofunction-qualname:: mvsr.Interpolate.linear
.. autofunction-qualname:: mvsr.Interpolate.smooth

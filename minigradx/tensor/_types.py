import typing

import numpy.typing as npt

Numeric = typing.Union[int, float, bool]

NestedNumeric = typing.Union[
    Numeric, list["NestedNumeric"], tuple["NestedNumeric", ...]
]

ConvertableToTensor = typing.Union[NestedNumeric, npt.NDArray]

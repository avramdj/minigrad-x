import typing

Numeric = typing.Union[int, float, bool]

NestedNumeric = typing.Union[
    Numeric, list["NestedNumeric"], tuple["NestedNumeric", ...]
]

ConvertableToTensor = typing.Union[NestedNumeric]

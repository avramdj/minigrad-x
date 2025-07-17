from __future__ import annotations

from typing import Optional

from minigradx._C import (
    Dtype,
    TensorImpl,
    make_impl_from_data,
    make_impl_from_shape,
)
from minigradx.tensor._types import ConvertableToTensor


class Tensor:
    def __init__(
        self,
        data: Optional[ConvertableToTensor] = None,
        device: str = "cpu",
        dtype: Dtype = Dtype.Float32,
        requires_grad: bool = True,
    ) -> None:
        self._impl = make_impl_from_data(
            data=data,
            requires_grad=requires_grad,
            device=device,
            dtype=dtype,
        )

    def zeros(
        self,
        shape: tuple[int, ...],
        device: str = "cpu",
        dtype: Dtype = Dtype.Float32,
        requires_grad: bool = True,
    ) -> Tensor:
        return self._from_impl(
            make_impl_from_shape(
                shape=shape,
                requires_grad=requires_grad,
                device=device,
                dtype=dtype,
            )
        )

    def __getitem__(self, indices: tuple[int, ...] | list[int]) -> Tensor:
        return self._from_impl(self._impl.__getitem__(indices))

    @property
    def device(self) -> str:
        return self._impl.device

    # @property
    # def grad(self) -> Optional[Tensor]:
    #     if self._impl.grad is None:
    #         return None
    #     return Tensor._from_impl(self._impl.grad)

    # @property
    # def shape(self) -> tuple[int, ...]:
    #     return self._impl.shape

    @classmethod
    def _from_impl(cls, impl: TensorImpl, copy: bool = False) -> Tensor:
        tensor = cls(device=impl.device)
        if copy:
            tensor._impl = impl.clone()
        else:
            tensor._impl = impl

        return tensor

from __future__ import annotations

from typing import Optional

from minigradx._C import TensorImpl, make_tensor_impl
from minigradx.tensor.utils import ConvertableToTensor


class Tensor:
    def __init__(
        self, data: Optional[ConvertableToTensor] = None, device: str = "cpu"
    ) -> None:
        if data is not None:
            self._impl = make_tensor_impl(data, device)
        else:
            self._impl = make_tensor_impl([], device)

    def __getitem__(self, indices: tuple[int, ...] | list[int]) -> Tensor:
        return self._from_impl(self._impl.__getitem__(indices))

    @property
    def device(self) -> str:
        return self._impl.device

    # @property
    # def shape(self) -> tuple[int, ...]:
    #     return self._impl.shape

    @classmethod
    def _from_impl(cls, impl: TensorImpl) -> Tensor:
        tensor = cls(device=impl.device)
        tensor._impl = impl
        return tensor

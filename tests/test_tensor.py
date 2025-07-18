import numpy as np
import pytest

from minigradx import Tensor
from minigradx._C import Dtype


def create_nested_list(shape):
    if not shape:
        return 1.0
    size = np.prod(shape)
    return np.arange(size, dtype=np.float32).reshape(shape).tolist()


@pytest.fixture
def sample_data_2d():
    """Provides a 2D list for testing."""
    return [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]


@pytest.fixture
def sample_data_3d():
    """Provides a 3D list for testing."""
    return [[[1, 2], [3, 4]], [[5, 6], [7, 8]]]


class TestTensorCreation:
    """Tests for various ways of creating a Tensor."""

    def test_from_nested_list_2d(self, sample_data_2d):
        """Tests tensor creation from a 2D list."""
        tensor = Tensor(sample_data_2d, device="cpu", dtype=Dtype.Float32)
        assert tensor.shape == (2, 3)
        assert tensor.size == 6
        assert tensor.dtype == Dtype.Float32
        assert tensor.device == "cpu"
        assert tensor.requires_grad is True

    def test_from_nested_list_3d(self, sample_data_3d):
        """Tests tensor creation from a 3D list."""
        tensor = Tensor(sample_data_3d, dtype=Dtype.Int32, requires_grad=False)
        assert tensor.shape == (2, 2, 2)
        assert tensor.size == 8
        assert tensor.dtype == Dtype.Int32
        assert tensor.device == "cpu"
        assert tensor.requires_grad is False

    def test_from_numpy_array(self, sample_data_2d):
        """Tests tensor creation from a NumPy array."""
        np_array = np.array(sample_data_2d, dtype=np.float64)
        tensor = Tensor(np_array, dtype=Dtype.Float64)
        assert tensor.shape == (2, 3)
        assert tensor.size == 6
        assert tensor.dtype == Dtype.Float64

    def test_scalar_tensor(self):
        """Tests creation of a scalar tensor."""
        tensor = Tensor(42.0, dtype=Dtype.Float32)
        assert tensor.shape == ()
        assert tensor.size == 1
        assert tensor.dtype == Dtype.Float32

    def test_inconsistent_shape_error(self):
        """Tests that creation fails with inconsistent nested list shapes."""
        data = [[1, 2, 3], [4, 5]]
        with pytest.raises(ValueError, match="Inconsistent shapes at index 1"):
            Tensor(data)

    def test_zeros_factory(self):
        """Tests the zeros() factory method."""
        shape = (4, 5, 6)
        tensor = Tensor().zeros(
            shape, device="cpu", dtype=Dtype.Float32, requires_grad=False
        )
        assert tensor.shape == shape
        assert tensor.size == 4 * 5 * 6
        assert tensor.dtype == Dtype.Float32
        assert tensor.device == "cpu"
        assert tensor.requires_grad is False


class TestTensorProperties:
    """Tests for accessing Tensor properties."""

    @pytest.mark.parametrize(
        "shape, expected_size",
        [
            ((5,), 5),
            ((2, 3), 6),
            ((4, 2, 3), 24),
            ((), 1),
        ],
    )
    def test_shape_and_size(self, shape, expected_size):
        """Verify .shape and .size properties."""
        data = create_nested_list(shape)
        tensor = Tensor(data)
        assert tensor.shape == shape
        assert tensor.size == expected_size

    @pytest.mark.parametrize(
        "dtype_enum, requires_grad_val",
        [
            (Dtype.Float64, True),
            (Dtype.Int32, False),
            (Dtype.Bool, True),
        ],
    )
    def test_dtype_and_requires_grad(self, dtype_enum, requires_grad_val):
        """Verify .dtype and .requires_grad properties."""
        tensor = Tensor([1, 2], dtype=dtype_enum, requires_grad=requires_grad_val)
        assert tensor.dtype == dtype_enum
        assert tensor.requires_grad == requires_grad_val


class TestTensorOperations:
    """Tests for tensor operations."""

    def test_getitem_not_implemented(self, sample_data_2d):
        """
        Tests that __getitem__ raises an error, confirming the C++ binding works
        even if the feature is not yet implemented.
        """
        tensor = Tensor(sample_data_2d)
        # with pytest.raises(ValueError, match="Not implemented"):
        with pytest.raises(NotImplementedError):
            tensor[0]

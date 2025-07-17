from minigradx import Tensor


def test_cpu_tensor_repr():
    t = Tensor(device="cpu")
    assert t.device == "cpu"


def test_cuda_tensor_repr():
    t = Tensor(device="cuda")
    assert t.device == "cuda"

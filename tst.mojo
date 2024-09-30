from tensor import Tensor, TensorShape
from random import seed
from structured.strd.utils import matmul, load_csv, tensor_transpose, copy, add_bias
from structured.strd.nn import softmax, get_accuracy, get_predictions
alias type: DType = DType.float32
alias n: Int = 784
alias m_test: Int = 10000

fn main() raises -> None:
  var test_data: Tensor[type] = tensor_transpose[type, 0, 1](load_csv(
    "./structured/datasets/MNIST_CSV/mnist_test.csv"
  ))
  var Y: Tensor[type] = copy[type](TensorShape(1, m_test), test_data, 0)
  var X: Tensor[type] = copy[type, 255](TensorShape(n, m_test), test_data, m_test)

  seed()
  var W1: Tensor[type] = Tensor[type].randn(TensorShape(10, n), 0, 0.25).clip(-0.5, 0.5)
  var b1: Tensor[type] = Tensor[type].randn(TensorShape(10, 1), 0, 0.25).clip(-0.5, 0.5)
  var W2: Tensor[type] = Tensor[type].randn(TensorShape(10, 10), 0, 0.25).clip(-0.5, 0.5)
  var b2: Tensor[type] = Tensor[type].randn(TensorShape(10, 1), 0, 0.25).clip(-0.5, 0.5)

  var Z1: Tensor[type] = add_bias[type](matmul[type](W1, X), b1)
  var A1: Tensor[type] = Z1.clip(0, Scalar[type].MAX)
  var A2: Tensor[type] = softmax[type](add_bias[type](matmul[type](W2, A1), b2))

  print(get_predictions(A2))

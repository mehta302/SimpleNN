from tensor import Tensor, TensorShape
from random import seed
from math import exp
from structured.strd.utils import matmul, load_csv, tensor_transpose, copy, tensor_exp, tensor_sum1, div_bias1
from structured.strd.nn import forward_prop, get_accuracy, get_predictions
alias type: DType = DType.float32
alias n: Int = 784
alias m_test: Int = 10000

fn main() raises -> None:
  var test_data: Tensor[type] = tensor_transpose[type, 0, 1](load_csv(
    "./structured/datasets/MNIST_CSV/mnist_test.csv"
  ))
  var tst: Tensor[type] = copy[type](TensorShape(1, m_test), test_data, 0)
  var test: Tensor[type] = copy[type](TensorShape(n, m_test), test_data, m_test)
  test_data._take_data_ptr().free()

  seed()
  var W1: Tensor[type] = Tensor[type].randn(TensorShape(10, n), 0, 0.25).clip(-0.5, 0.5)
  var b1: Tensor[type] = Tensor[type].randn(TensorShape(10, 1), 0, 0.25).clip(-0.5, 0.5)
  var W2: Tensor[type] = Tensor[type].randn(TensorShape(10, 10), 0, 0.25).clip(-0.5, 0.5)
  var b2: Tensor[type] = Tensor[type].randn(TensorShape(10, 1), 0, 0.25).clip(-0.5, 0.5)

  var A2: Tensor[type] = forward_prop[type, m_test](W1, b1, W2, b2, test)[3]
  print(tensor_transpose[type, 0, 1](A2).argmax(axis=1))

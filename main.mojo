from tensor import Tensor, TensorShape
from random import seed
from structured.strd.utils import load_csv, tensor_transpose, copy
from structured.strd.nn import forward_prop, back_prop, update_weights, get_predictions, get_accuracy
alias type: DType = DType.float32
alias n: Int = 784
alias m_train: Int = 60000
alias m_test: Int = 10000

fn main() raises -> None:
  var train_data: Tensor[type] = tensor_transpose[type, 0, 1](load_csv(
    "./structured/datasets/MNIST_CSV/mnist_train.csv"
  ))

  var Y_train: Tensor[type] = copy[type](TensorShape(1, m_train), train_data, 0)
  var X_train: Tensor[type] = copy[type](TensorShape(n, m_train), train_data, m_train)
  train_data._take_data_ptr().free()

  seed()
  var W1: Tensor[type] = Tensor[type].randn(TensorShape(10, n), 0, 0.25).clip(-0.5, 0.5)
  var b1: Tensor[type] = Tensor[type].randn(TensorShape(10, 1), 0, 0.25).clip(-0.5, 0.5)
  var W2: Tensor[type] = Tensor[type].randn(TensorShape(10, 10), 0, 0.25).clip(-0.5, 0.5)
  var b2: Tensor[type] = Tensor[type].randn(TensorShape(10, 1), 0, 0.25).clip(-0.5, 0.5)

  var forwards: Tuple[Tensor[type], Tensor[type], Tensor[type], Tensor[type]]
  var backwards: Tuple[Tensor[type], Tensor[type], Tensor[type], Tensor[type]]

  @parameter
  for i in range(10):
    forwards = forward_prop[type, m_train](W1, b1, W2, b2, X_train)
    backwards = back_prop[type](forwards[0], forwards[1], forwards[3], W2, X_train, Y_train)
    update_weights(W1, b1, W2, b2, backwards[0], backwards[1], backwards[2], backwards[3], 0.1)

    print(i, get_accuracy[type](get_predictions(forwards[3]), Y_train)) 


  Y_train._take_data_ptr().free()
  X_train._take_data_ptr().free()

  var test_data: Tensor[type] = tensor_transpose[type, 0, 1](load_csv(
    "./structured/datasets/MNIST_CSV/mnist_test.csv"
  ))

  var Y_test: Tensor[type] = copy[type](TensorShape(1, n), test_data, 0)
  var X_test: Tensor[type] = copy[type](TensorShape(n, m_test), test_data, m_test)
  test_data._take_data_ptr().free()

  var test: Tensor[type] = forward_prop[type, m_test](
    W1, b1, W2, b2, X_test
  )[3]
  print("Final Accuracy: ", get_accuracy[type](get_predictions(test), Y_test))

from structured.strd.utils import tensor_exp, tensor_transpose, tensor_sum, add_bias, div_bias, matmul, tensor_sum1, div_bias1
from tensor import Tensor, TensorShape
from algorithm import max, parallelize

fn ReLU[type: DType](Z: Tensor[type]) -> Tensor[type]:
  return Z.clip(0, Scalar[type].MAX)

fn deriv_ReLU[type: DType](container: Tensor[type]) -> Tensor[type]:
  var output: Tensor[type] = Tensor[type](container.shape())
  @parameter
  fn derive(i: Int) -> None:
    output[i] = 1 if container[i] > 0 else 0
  parallelize[derive](container.num_elements(), container.num_elements())
  
  return output

fn softmax[type: DType](Z: Tensor[type]) raises -> Tensor[type]:
  var exp_Z: Tensor[type] = tensor_exp[type](Z / 10.0)
  return div_bias1[type](exp_Z, tensor_sum1[type](exp_Z))

fn one_hot[type: DType](Y: Tensor[type]) raises -> Tensor[type]:
  var m: Int = Y.num_elements()
  var one_hot_Y: Tensor[type] = Tensor[type](int(max(Y._to_buffer()))+1, m)
  @parameter
  fn encode(i: Int):
    one_hot_Y[(int(Y[i])*m)+i] = 1
  parallelize[encode](m, m)
  return one_hot_Y

fn forward_prop[type: DType, COLS: Int](
  W1: Tensor[type], b1: Tensor[type], W2: Tensor[type], b2: Tensor[type], X: Tensor[type]
) raises -> Tuple[Tensor[type], Tensor[type], Tensor[type], Tensor[type]]:
  var Z1: Tensor[type] = add_bias[type](matmul[type](W1, X), b1)
  var A1: Tensor[type] = ReLU[type](Z1)
  var Z2: Tensor[type] = add_bias[type](matmul[type](W2, A1), b2)
  var A2: Tensor[type] = softmax[type](Z2)
  return Z1, A1, Z2, A2

fn back_prop[type: DType](
  owned Z1: Tensor[type],
  owned A1: Tensor[type],
  A2: Tensor[type],
  W2: Tensor[type],
  X: Tensor[type],
  Y: Tensor[type]
) raises -> Tuple[Tensor[type], Tensor[type], Tensor[type], Tensor[type]]:
  var m: Int = Y.num_elements()
  var one_hot_Y: Tensor[type] = one_hot[type](Y)
  var dZ2: Tensor[type] = A2 - one_hot_Y
  var dW2: Tensor[type] = matmul[type](((1 / m) * dZ2), tensor_transpose[type, 0, 1](A1))
  var db2: Tensor[type] = (1 / m) * tensor_sum[type](dZ2)
  var dZ1: Tensor[type] = matmul[type](matmul[type](tensor_transpose[type, 0, 1](W2), dZ2), deriv_ReLU(Z1))
  var dW1: Tensor[type] = matmul[type](((1 / m) * dZ1), tensor_transpose[type, 0, 1](X))
  var db1: Tensor[type] = (1 / m) * tensor_sum[type](dZ1)
  return dW1, db1, dW2, db2

fn update_weights[type: DType](
  inout W1: Tensor[type], inout b1: Tensor[type], inout W2: Tensor[type], inout b2: Tensor[type],
  owned dW1: Tensor[type], owned db1: Tensor[type], owned dW2: Tensor[type], owned db2: Tensor[type], 
  owned alpha: SIMD[type, 1]
) raises -> None:
  W1 = matmul[type]((W1 - alpha), dW1)
  b1 = matmul[type]((b1 - alpha), db1)
  W2 = matmul[type]((W2 - alpha), dW2)
  b2 = matmul[type]((b2 - alpha), db2)

fn get_predictions[type: DType](owned A2: Tensor[type]) raises -> Tensor[type]:
  return tensor_transpose[type, 0, 1](A2).argmax(axis=1)

fn get_accuracy[type: DType](owned predictions: Tensor[type], Y: Tensor[type]) -> Float32:
  var sm: Int = 0
  var m: Int = Y.num_elements()
  @parameter
  fn summation(i: Int):
    if predictions[i] == Y[i]:
      sm += 1 
  parallelize[summation](m, m)
  return sm / m

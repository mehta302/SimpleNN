from tensor import Tensor, TensorShape
from algorithm import vectorize, parallelize
from random import seed, random_si64
from math import exp
alias float = DType.float32

# Loads a 2-d float32 tensor with data from a given csv file path 
fn load_csv(path: String) raises -> Tensor[float]:
  var container: Tensor[float]
  var lines: List[String]
  var elements: List[String]
  var ROWS: Int
  var COLS: Int

  with open(path, "r") as file:
    lines = file.read().split("\n")[:-1]
    ROWS = len(lines)
    COLS = len(lines[0].split(","))

    container = Tensor[float](TensorShape(ROWS, COLS))

    for i in range(ROWS):
      elements = lines[i].split(",")
      for j in range(COLS):
        container[((COLS * i) + j)] = atol(elements[j])
 
  return container

# Convert coordinate list to index value
fn Index(dimensions: List[Int], nums: List[Int]) -> Int:
  var idx: Int = 0
  var mul: Int = 1

  for i in range(len(dimensions)):
    idx += mul * nums[-i-1]
    mul *= dimensions[-i-1]

  return idx

# Convert index value to coordinate list
fn Coordinate(dimensions: List[Int], idx: Int) -> List[Int]:
  var nums: List[Int] = List[Int]()
  var div: Int = 1
  var index: Int = idx

  for i in dimensions:
    div *= i[]
  
  for i in dimensions:
    div /= i[]
    nums.append(index // div)
    index %= div

  return nums

# Transpose 2 dimensions of a tensor
fn tensor_transpose[type: DType, dim1: Int, dim2: Int](container: Tensor[type]) -> Tensor[type]:
  var original_shape: List[Int] = List[Int]()
  var new_shape: List[Int]
  var output: Tensor[type]
  var coordinates: List[Int]

  for i in range(container.shape().rank()):
    original_shape.append(container.dim(i))
  
  new_shape = original_shape
  new_shape[dim1], new_shape[dim2] = new_shape[dim2], new_shape[dim1]

  output = Tensor[type](TensorShape(new_shape))

  for i in range(container.num_elements()):
    coordinates = Coordinate(original_shape, i)
    coordinates[dim1], coordinates[dim2] = coordinates[dim2], coordinates[dim1]
    output[Index(new_shape, coordinates)] = container[i]
  
  return output


# SIMD shuffles in place the rows of a 2d Tensor given the sizes of its 2 dimensions
fn shuffle[type: DType, ROWS: Int, COLS: Int](inout container: Tensor[type]) -> None:
  seed()
  alias type_width: IntLiteral = simdwidthof[type]()
  var random_int: Int
  var replacement_value: SIMD[type, COLS]

  @parameter
  fn shuffler[simd_width: Int](idx: Int) -> None:
    random_int = int(random_si64(0, ROWS-1))
    replacement_value = container.load[width=COLS](idx*COLS)
    container.store[width=COLS](idx*COLS, container.load[width=COLS](random_int*COLS))
    container.store[width=COLS](random_int*COLS, replacement_value)

  vectorize[shuffler, type_width](ROWS)

# SIMD returns a Tensor with applied exp() to every element in given Tensor
fn tensor_exp[type: DType](container: Tensor[type]) -> Tensor[type]:
  alias type_width: IntLiteral = simdwidthof[type]()
  var exp_container: Tensor[type] = Tensor[type](container.shape())

  @parameter
  fn vec_exp[simd_width: Int](idx: Int) -> None:
    exp_container.store[width=simd_width](idx, exp(container.load[width=simd_width](idx)))
  
  vectorize[vec_exp, type_width](container.num_elements())
  return exp_container

# SIMD returns a scalar containing the total sum of all 2d Tensor elements along axis
fn tensor_sum[type: DType](container: Tensor[type]) -> Tensor[type]:
  alias type_width: IntLiteral = simdwidthof[type]()
  var ROWS: Int = container.dim(0)
  var COLS: Int = container.dim(1)
  var output: Tensor[type] = Tensor[type](TensorShape(ROWS, 1))

  @parameter
  fn calc_row(i: Int) -> None:
    @parameter
    fn vec_add[simd_width: Int](idx: Int) -> None:
      output[i] += container.load[width=simd_width]((COLS*i)+idx).reduce_add()
    vectorize[vec_add, type_width](COLS) 
  parallelize[calc_row](ROWS, ROWS)
  
  return output

fn tensor_sum1[type: DType](container: Tensor[type]) -> Tensor[type]:
  alias type_width = simdwidthof[type]()
  var ROWS: Int = container.dim(0)
  var COLS: Int = container.dim(1)
  var output: Tensor[type] = Tensor[type](TensorShape(1, COLS))

  @parameter
  fn calc_col(i: Int) -> None:
    @parameter
    fn vec_add[simd_width: Int](idx: Int) -> None:
      output.store[width=simd_width](idx, 
        output.load[width=simd_width](idx) + container.load[width=simd_width]((COLS*i)+idx)
      )
    vectorize[vec_add, type_width](COLS)
  parallelize[calc_col](ROWS, ROWS)

  return output

# SIMD add bias vector to 2d tensor
fn add_bias[type: DType](container: Tensor[type], bias: Tensor[type]) -> Tensor[type]:
  alias type_width: IntLiteral = simdwidthof[type]()
  var output: Tensor[type] = Tensor[type](container.shape())
  var COLS: Int = container.dim(1)
  var ROWS: Int = container.dim(0)

  @parameter
  fn calc_row(i: Int) -> None:
    @parameter
    fn add[simd_width: Int](j: Int):
      output.store[width=simd_width](i*COLS+j, container.load[width=simd_width](i*COLS+j) + bias[i])
    vectorize[add, type_width](COLS) 
  parallelize[calc_row](ROWS, ROWS)

  return output

# SIMD divide bias vector to 2d tensor
fn div_bias[type: DType](container: Tensor[type], bias: Tensor[type]) -> Tensor[type]:
  alias type_width: IntLiteral = simdwidthof[type]()
  var output: Tensor[type] = Tensor[type](container.shape())
  var COLS: Int = container.dim(1)
  var ROWS: Int = container.dim(0)

  @parameter
  fn calc_row(i: Int) -> None:
    @parameter
    fn add[simd_width: Int](j: Int):
      output.store[width=simd_width](i*COLS+j, container.load[width=simd_width](i*COLS+j) / bias[i])
    vectorize[add, type_width](COLS) 
  parallelize[calc_row](ROWS, ROWS)

  return output

fn div_bias1[type: DType](container: Tensor[type], bias: Tensor[type]) -> Tensor[type]:
  alias type_width: IntLiteral = simdwidthof[type]()
  var output: Tensor[type] = Tensor[type](container.shape())
  var COLS: Int = container.dim(1)
  var ROWS: Int = container.dim(0)

  @parameter
  fn calc_row(i: Int) -> None:
    @parameter
    fn add[simd_width: Int](j: Int):
      output.store[width=simd_width](i*COLS+j, container.load[width=simd_width](i*COLS+j) / bias.load[width=simd_width](j))
    vectorize[add, type_width](COLS) 
  parallelize[calc_row](ROWS, ROWS)

  return output

# SIMD Parallel computes matrix multiplication on 2d tensor. Cols of lhs must be equal to rows of rhs
fn matmul[type: DType](lhs: Tensor[type], rhs: Tensor[type]) -> Tensor[type]:
  alias type_width: IntLiteral = simdwidthof[type]()
  var N: Int = rhs.dim(1)
  var M: Int = lhs.dim(0)
  var K: Int = lhs.dim(1)
  var output: Tensor[type] = Tensor[type](TensorShape(M, N))

  @parameter
  fn calc_row(m: Int) -> None:
    for k in range(K):
      @parameter
      fn dot[simd_width: Int](n: Int) -> None:
        output.store[width=simd_width](m*N+n,
          rhs.load[width=simd_width](k*N+n).fma(
            lhs[m*K+k], output.load[width=simd_width](m*N+n)
          ) 
        )
      vectorize[dot, type_width](N)
  parallelize[calc_row](M, M)

  return output

# SIMD copy values from one tensor to another at a given offset
fn copy[type: DType](owned to: TensorShape, of: Tensor[type], offset: Int) -> Tensor[type]:
  alias type_width: IntLiteral = simdwidthof[type]()
  var output: Tensor[type] = Tensor[type](to)
  @parameter
  fn store[simd_width: Int](idx: Int):
    output.store[width=simd_width](idx, of.load[width=simd_width](idx+offset))
  vectorize[store, type_width](to.num_elements())

  return output

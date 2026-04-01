builtin.module {
  func.func @simple_function(%arg0: i32) -> i32 {
    %0 = arith.addi %arg0, %arg0 : i32 loc("source.mlir":10:5)
    %1 = arith.muli %0, %0 : i32 loc("source.mlir":12:3)
    return %0 : i32
  }
}
builtin.module {
  func.func @simple_function(%arg0: f32) -> f32 {
    %0 = arith.addf %arg0, %arg0 : f32 loc("source.mlir":10:5)
    %1 = arith.mulf %0, %0 : f32 loc("source.mlir":12:3)
    return %0 : f32
  }
}
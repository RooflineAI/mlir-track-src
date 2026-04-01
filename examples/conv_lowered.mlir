builtin.module {
  func.func @conv_paths(
      %input: tensor<1x4x4x1xf32>,
      %filter: tensor<1x1x1x1xf32>
  ) -> (tensor<1x4x4x1xf32>, tensor<1x4x4x1xf32>, tensor<1x4x4x1xf32>, tensor<1x4x4x1xf32>) {
    // Lowering for source conv 1: matmul-style path.
    %zero0 = arith.constant 0.0 : f32 loc("examples/conv_input.mlir":12:14)
    %pad0 = tensor.pad %input low[0, 0, 0, 0] high[0, 0, 0, 0] {
    ^bb0(%n: index, %h: index, %w: index, %c: index):
      tensor.yield %zero0 : f32 loc("examples/conv_input.mlir":12:14)
    } : tensor<1x4x4x1xf32> to tensor<1x4x4x1xf32> loc("examples/conv_input.mlir":12:14)
    %lhs0 = tensor.collapse_shape %pad0 [[0, 1, 2], [3]]
      : tensor<1x4x4x1xf32> into tensor<16x1xf32> loc("examples/conv_input.mlir":12:14)
    %rhs0 = tensor.collapse_shape %filter [[0, 1, 2], [3]]
      : tensor<1x1x1x1xf32> into tensor<1x1xf32> loc("examples/conv_input.mlir":12:14)
    %mat_init0 = tensor.empty() : tensor<16x1xf32> loc("examples/conv_input.mlir":12:14)
    %mat_fill0 = linalg.fill ins(%zero0 : f32) outs(%mat_init0 : tensor<16x1xf32>) -> tensor<16x1xf32> loc("examples/conv_input.mlir":12:14)
    %matmul0 = linalg.matmul ins(%lhs0, %rhs0 : tensor<16x1xf32>, tensor<1x1xf32>) outs(%mat_fill0 : tensor<16x1xf32>) -> tensor<16x1xf32> loc("examples/conv_input.mlir":12:14)
    %conv0 = tensor.expand_shape %matmul0 [[0, 1, 2], [3]] output_shape [1, 4, 4, 1]
      : tensor<16x1xf32> into tensor<1x4x4x1xf32> loc("examples/conv_input.mlir":12:14)

    // Lowering for source conv 2: direct convolution path.
    %zero1 = arith.constant 0.0 : f32 loc("examples/conv_input.mlir":18:14)
    %init1 = tensor.empty() : tensor<1x4x4x1xf32> loc("examples/conv_input.mlir":18:14)
    %fill1 = linalg.fill ins(%zero1 : f32) outs(%init1 : tensor<1x4x4x1xf32>) -> tensor<1x4x4x1xf32> loc("examples/conv_input.mlir":18:14)
    %conv1 = linalg.conv_2d_nhwc_hwcf
      {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>}
      ins(%input, %filter : tensor<1x4x4x1xf32>, tensor<1x1x1x1xf32>)
      outs(%fill1 : tensor<1x4x4x1xf32>) -> tensor<1x4x4x1xf32> loc("examples/conv_input.mlir":18:14)

    // Lowering for source conv 3: matmul-style path.
    %zero2 = arith.constant 0.0 : f32 loc("examples/conv_input.mlir":24:14)
    %pad2 = tensor.pad %input low[0, 0, 0, 0] high[0, 0, 0, 0] {
    ^bb0(%n: index, %h: index, %w: index, %c: index):
      tensor.yield %zero2 : f32 loc("examples/conv_input.mlir":24:14)
    } : tensor<1x4x4x1xf32> to tensor<1x4x4x1xf32> loc("examples/conv_input.mlir":24:14)
    %lhs2 = tensor.collapse_shape %pad2 [[0, 1, 2], [3]]
      : tensor<1x4x4x1xf32> into tensor<16x1xf32> loc("examples/conv_input.mlir":24:14)
    %rhs2 = tensor.collapse_shape %filter [[0, 1, 2], [3]]
      : tensor<1x1x1x1xf32> into tensor<1x1xf32> loc("examples/conv_input.mlir":24:14)
    %mat_init2 = tensor.empty() : tensor<16x1xf32> loc("examples/conv_input.mlir":24:14)
    %mat_fill2 = linalg.fill ins(%zero2 : f32) outs(%mat_init2 : tensor<16x1xf32>) -> tensor<16x1xf32> loc("examples/conv_input.mlir":24:14)
    %matmul2 = linalg.matmul ins(%lhs2, %rhs2 : tensor<16x1xf32>, tensor<1x1xf32>) outs(%mat_fill2 : tensor<16x1xf32>) -> tensor<16x1xf32> loc("examples/conv_input.mlir":24:14)
    %conv2 = tensor.expand_shape %matmul2 [[0, 1, 2], [3]] output_shape [1, 4, 4, 1]
      : tensor<16x1xf32> into tensor<1x4x4x1xf32> loc("examples/conv_input.mlir":24:14)

    // Lowering for source conv 4: direct convolution path.
    %zero3 = arith.constant 0.0 : f32 loc("examples/conv_input.mlir":30:14)
    %init3 = tensor.empty() : tensor<1x4x4x1xf32> loc("examples/conv_input.mlir":30:14)
    %fill3 = linalg.fill ins(%zero3 : f32) outs(%init3 : tensor<1x4x4x1xf32>) -> tensor<1x4x4x1xf32> loc("examples/conv_input.mlir":30:14)
    %conv3 = linalg.conv_2d_nhwc_hwcf
      {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>}
      ins(%input, %filter : tensor<1x4x4x1xf32>, tensor<1x1x1x1xf32>)
      outs(%fill3 : tensor<1x4x4x1xf32>) -> tensor<1x4x4x1xf32> loc("examples/conv_input.mlir":30:14)

    return %conv0, %conv1, %conv2, %conv3
      : tensor<1x4x4x1xf32>, tensor<1x4x4x1xf32>, tensor<1x4x4x1xf32>, tensor<1x4x4x1xf32>
  }
}

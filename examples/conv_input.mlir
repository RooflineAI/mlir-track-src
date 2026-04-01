builtin.module {
  func.func @conv_paths(
      %input: tensor<1x4x4x1xf32>,
      %filter: tensor<1x1x1x1xf32>
  ) -> (tensor<1x4x4x1xf32>, tensor<1x4x4x1xf32>, tensor<1x4x4x1xf32>, tensor<1x4x4x1xf32>) {
    %init0 = tensor.empty() : tensor<1x4x4x1xf32>
    %init1 = tensor.empty() : tensor<1x4x4x1xf32>
    %init2 = tensor.empty() : tensor<1x4x4x1xf32>
    %init3 = tensor.empty() : tensor<1x4x4x1xf32>

    // Source conv 1.
    %conv0 = linalg.conv_2d_nhwc_hwcf
      {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>}
      ins(%input, %filter : tensor<1x4x4x1xf32>, tensor<1x1x1x1xf32>)
      outs(%init0 : tensor<1x4x4x1xf32>) -> tensor<1x4x4x1xf32>

    // Source conv 2.
    %conv1 = linalg.conv_2d_nhwc_hwcf
      {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>}
      ins(%input, %filter : tensor<1x4x4x1xf32>, tensor<1x1x1x1xf32>)
      outs(%init1 : tensor<1x4x4x1xf32>) -> tensor<1x4x4x1xf32>

    // Source conv 3.
    %conv2 = linalg.conv_2d_nhwc_hwcf
      {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>}
      ins(%input, %filter : tensor<1x4x4x1xf32>, tensor<1x1x1x1xf32>)
      outs(%init2 : tensor<1x4x4x1xf32>) -> tensor<1x4x4x1xf32>

    // Source conv 4.
    %conv3 = linalg.conv_2d_nhwc_hwcf
      {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>}
      ins(%input, %filter : tensor<1x4x4x1xf32>, tensor<1x1x1x1xf32>)
      outs(%init3 : tensor<1x4x4x1xf32>) -> tensor<1x4x4x1xf32>

    return %conv0, %conv1, %conv2, %conv3
      : tensor<1x4x4x1xf32>, tensor<1x4x4x1xf32>, tensor<1x4x4x1xf32>, tensor<1x4x4x1xf32>
  }
}

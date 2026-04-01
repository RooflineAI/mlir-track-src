# mlir-track-src

`mlir-track-src` is a small command-line tool for tracking MLIR operations
across compiler pipeline stages using source locations. MLIR operations carry
location information, and when that information is preserved across rewrites and
lowerings, it can be used to associate operations between different IR
snapshots. This project turns that idea into a practical workflow for debugging
and understanding large MLIR pipelines.

The tool is especially useful when pass pipelines significantly restructure the
IR, for example by decomposing, expanding, or fusing operations. Instead of
manually comparing large `.mlir` dumps, `mlir-track-src` lets you inspect how
operations evolve across stages and ask questions such as: what did this
operation become, where did this later operation come from, and which operations
took a specific lowering path?

## Source Tracking

`mlir-track-src` provides an interactive source-tracking workflow for MLIR
files. Given an input MLIR file and another MLIR file from a later pipeline
stage, it builds indexes over the operations and uses their annotated source
locations to find corresponding operations across stages.

Here is a very simple example. Assume we have two MLIR files:
[examples/input.mlir](examples/input.mlir)
```
builtin.module {
  func.func @simple_function(%arg0: i32) -> i32 {
    %0 = arith.addi %arg0, %arg0 : i32 loc("source.mlir":10:5)
    %1 = arith.muli %0, %0 : i32 loc("source.mlir":12:3)
    return %0 : i32
  }
}
```
and:
[examples/track.mlir](examples/track.mlir)
```
builtin.module {
  func.func @simple_function(%arg0: f32) -> f32 {
    %0 = arith.addf %arg0, %arg0 : f32 loc("source.mlir":10:5)
    %1 = arith.mulf %0, %0 : f32 loc("source.mlir":12:3)
    return %0 : f32
  }
}
```

We want to see how a certain operation, here `%0 = arith.addi ...` is affected
by transformation in the pipeline.

First we start the tool:
```bash
mlir-track-src --input-mlir examples/input.mlir --track-mlir examples/track.mlir
```

The tool loads both MLIR modules, builds operation indexes, and opens an
interactive Python shell with helper commands for source-based tracking:
```
Building operation index...
Building tracking operation index...
Starting interactive shell...
Interactive shell. Type 'info' for available commands.
>>> info
Available commands:
   ...
```

We can use the `op_index` to look up a certain operation from the IR and using
the `show` and `track` commands we can see how the operation is transformed:
```
>>> show(op_index.get_ops_by_name("%0"))
%0 = arith.addi %arg0, %arg0 : i32`
>>> show(track(op_index.get_ops_by_name("%0")))
%0 = arith.addf %arg0, %arg0 : f32`
```

## Example: Analyzing A Lowering Decision

`mlir-track-src` can also be used to analyze lowering behavior across an entire
MLIR module.

The files [examples/conv_input.mlir](examples/conv_input.mlir)
and [examples/conv_lowered.mlir](examples/conv_lowered.mlir)
show a small synthetic example with four
`linalg.conv_2d_nhwc_hwcf` operations. In the later-stage snapshot, two of
those convolutions take a matmul-style path and two stay on a direct
convolution-style path.

Start the tool on the two pipeline snapshots:

```bash
mlir-track-src --input-mlir examples/conv_input.mlir --track-mlir examples/conv_lowered.mlir --src-remap-list %i=examples/conv_input.mlir
```

The `--src-remap` is needed here because `conv_lowered.mlir` preserves source
locations as `examples/conv_input.mlir:<line>:<col>`, while `--input-mlir`
resolves `%i` to the full input path.

Now we can ask which source convolutions later produce a `linalg.matmul`:

```
>>> len(op_index.get_ops_by_type("linalg.conv_2d_nhwc_hwcf"))
4

>>> matmul_path = []
>>> non_matmul_path = []

>>> for op in op_index.get_ops_by_type("linalg.conv_2d_nhwc_hwcf"):
...     tracked_ops = track(op)
...     has_matmul = any(o.op_name == "linalg.matmul" for o in tracked_ops)
...     if has_matmul:
...         matmul_path.append(op)
...     else:
...         non_matmul_path.append(op)

>>> [o.ssa_name for o in matmul_path]
['%4', '%6']
>>> [o.ssa_name for o in non_matmul_path]
['%5', '%7']
```

We can also inspect one example:
```
>>> show(track(matmul_path[0]))
... truncated for README ...
----------------------------------------
%matmul0 = linalg.matmul ins(%lhs0, %rhs0 : tensor<16x1xf32>, tensor<1x1xf32>) outs(%mat_fill0 : tensor<16x1xf32>) -> tensor<16x1xf32> loc("examples/conv_input.mlir":12:14)
----------------------------------------
... truncated for README ...
```

## Development

### Setup

```bash
python -m venv .venv
source .venv/bin/activate
````

Install the package in editable mode:
```bash
python -m pip install --upgrade pip
python -m pip install -e ".[test]"
```

When using a local build of the LLVM project, set the `PYTHONPATH` to point to
the MLIR Python bindings:
```bash
export LLVM_PROJECT_BUILD=<path-to-llvm-project>
export PYTHONPATH="$LLVM_PROJECT_BUILD/tools/mlir/python_packages/mlir_core:$PYTHONPATH"
```

You can now start the tool or run tests.

### Running Tests
```bash
pytest
```

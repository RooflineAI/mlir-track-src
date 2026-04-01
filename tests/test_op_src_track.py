import sys
from pathlib import Path

import pytest
from mlir_track_src.op_src_track import (
    shrink_to_children,
    shrink_to_parents,
    track,
    unique_ops,
)
from mlir_track_src.ops import OperationIndex
from mlir_track_src.src_loc import SourceRange, SourceReMap
from mlir.ir import Context, Module


def _create_simple_module(tmp_path: Path) -> Path:
    s = """
    builtin.module {
      func.func @simple_function(%arg0: i32) -> i32 {
        %0 = arith.addi %arg0, %arg0 : i32 loc("source.mlir":10:5)
        %1 = arith.muli %0, %0 : i32 loc("source.mlir":12:3)
        return %0 : i32
      }
    }
    """
    mlir_path = tmp_path / "module.mlir"
    mlir_path.write_text(s)
    return mlir_path


def _create_simple_track_module(tmp_path: Path) -> Path:
    s = """
    builtin.module {
      func.func @simple_function(%arg0: f32) -> f32 {
        %0 = arith.addf %arg0, %arg0 : f32 loc("source.mlir":10:5)
        %1 = arith.mulf %0, %0 : f32 loc("source.mlir":12:3)
        return %0 : f32
      }
    }
    """
    mlir_path = tmp_path / "track_module.mlir"
    mlir_path.write_text(s)
    return mlir_path


def _load_module(ctx: Context, module_path: Path) -> Module:
    mod = Module.parseFile(str(module_path), ctx)
    return mod


def test_unique_ops_func(tmp_path: Path) -> None:
    ctx = Context()
    mod = _load_module(ctx, _create_simple_module(tmp_path))
    op_index = OperationIndex.create(mod.operation, SourceReMap())

    a = op_index.get_ops_by_name("%0")[0]
    b = op_index.get_info_by_op(mod.operation)

    unique = unique_ops([a, a, b, a, b])
    assert len(unique) == 2
    assert unique[0].ssa_name == "%0"
    assert unique[1].op_name == "builtin.module"


def test_track_func(tmp_path: Path) -> None:
    ctx = Context()
    mod_input = _load_module(ctx, _create_simple_module(tmp_path))
    mod_track = _load_module(ctx, _create_simple_track_module(tmp_path))

    op_index = OperationIndex.create(mod_input.operation, SourceReMap())
    track_index = OperationIndex.create(mod_track.operation, SourceReMap())

    op_info = op_index.get_ops_by_name("%0")[0]

    # Track operations by op info
    tracked_ops = track(op_info, track_index)
    assert len(tracked_ops) == 1
    tracked_op = tracked_ops[0]
    assert tracked_op.ssa_name == "%0"
    assert tracked_op.op_name == "arith.addf"
    assert tracked_op.src_rng == SourceRange.from_str("source.mlir:10:5")

    # Track operations by source range
    src_rng = SourceRange.from_str("source.mlir:10:1-13:1")
    tracked_ops = track(src_rng, track_index)
    assert len(tracked_ops) == 2
    ssa_names = {op.op_name for op in tracked_ops}
    assert ssa_names == {"arith.addf", "arith.mulf"}

    # Track operations by op info list
    tracked_ops = track([op_info, op_info, op_info], track_index)
    assert len(tracked_ops) == 1
    assert tracked_ops[0].ssa_name == "%0"

    # Track with None
    assert track(None, track_index) == []


def test_shrink_to_parents_and_children_funcs(tmp_path: Path) -> None:
    ctx = Context()
    mod = _load_module(ctx, _create_simple_module(tmp_path))
    op_index = OperationIndex.create(mod.operation, SourceReMap())

    module_info = op_index.get_info_by_op(mod.operation)
    addi_info = op_index.get_ops_by_name("%0")[0]

    ops = [module_info, addi_info]
    shrunk_ops = shrink_to_parents(ops)
    assert len(shrunk_ops) == 1
    assert shrunk_ops[0].op_name == "builtin.module"
    shrunk_ops = shrink_to_children(ops)
    assert len(shrunk_ops) == 1
    assert shrunk_ops[0].ssa_name == "%0"


if __name__ == "__main__":
    args = sys.argv[1:]
    ret_code = pytest.main([__file__, "--verbose", "-vv"] + args)

    sys.exit(ret_code)

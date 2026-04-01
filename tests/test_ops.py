import sys
from pathlib import Path

import pytest
from mlir_track_src.ops import OperationIndex, OperationInfo
from mlir_track_src.src_loc import SourceLocation, SourceRange, SourceReMap
from mlir.ir import Context, Module, Operation, WalkOrder, WalkResult


def _get_op_by_name(module: Module, name: str) -> Operation | None:
    target_op = None

    def _cb(op: Operation) -> WalkResult:
        nonlocal target_op
        results = op.results
        if len(results) and results[0].get_name() == name:
            target_op = op
            return WalkResult.INTERRUPT
        # Prevent descending into linalg.generic ops
        if op.name == "linalg.generic":
            return WalkResult.SKIP
        return WalkResult.ADVANCE

    module.operation.walk(_cb, WalkOrder.PRE_ORDER)
    return target_op


def _create_simple_module(tmp_path: Path) -> Path:
    s = """
    builtin.module {
      func.func @simple_function(%arg0: i32) -> i32 {
        %0 = arith.addi %arg0, %arg0 : i32 loc("source.mlir":10:5)
        %1 = arith.muli %0, %0 : i32 loc("source.mlir":12:3)
        return %1 : i32
      }
    }
    """
    mlir_path = tmp_path / "module.mlir"
    mlir_path.write_text(s)
    return mlir_path


def _load_module(module_path: Path) -> tuple[Context, Module]:
    ctx = Context()
    mod = Module.parseFile(str(module_path), ctx)
    return ctx, mod


def test_get_combined_src() -> None:
    child_0 = OperationInfo(
        ssa_name="%0",
        src_rng=SourceRange.from_str("source.mlir:10:5"),
        op_name="arith.addi",
        unique_id="%0",
        op=None,
    )
    assert child_0.get_combined_src() == SourceRange.from_str("source.mlir:10:5")

    child_1 = OperationInfo(
        ssa_name="%1",
        src_rng=SourceRange.from_str("source.mlir:12:3"),
        op_name="arith.muli",
        unique_id="%1",
        op=None,
        children=[child_0],
    )
    assert child_1.get_combined_src() == SourceRange.from_str("source.mlir:10:5-12:3")

    parent = OperationInfo(
        ssa_name="",
        src_rng=SourceRange.unknown(),
        op_name="func.func",
        unique_id="func.func[1]",
        op=None,
        children=[child_1],
    )
    assert parent.get_combined_src() == SourceRange.from_str("source.mlir:10:5-12:3")


def test_create_operation_info(tmp_path: Path) -> None:
    mlir_path = _create_simple_module(tmp_path)
    ctx, mod = _load_module(mlir_path)
    src_remap = SourceReMap(disabled_files={mlir_path})

    op = _get_op_by_name(mod, "%0")
    assert op is not None

    # For named op with SSA name
    op_info = OperationInfo.create(op, src_remap)
    assert op_info.ssa_name == "%0"
    assert op_info.src_rng == SourceRange.from_str("source.mlir:10:5")
    assert op_info.op_name == "arith.addi"
    assert op_info.unique_id == "%0"
    assert op_info.parent is None
    assert op_info.children == []

    op_info = OperationInfo.create(
        op, SourceReMap({Path("source.mlir"): Path("mapped_source.mlir")})
    )
    assert op_info.src_rng == SourceRange.from_str("mapped_source.mlir:10:5")

    # For unnamed op
    op_info = OperationInfo.create(mod.operation, src_remap)
    assert op_info.ssa_name == ""
    assert op_info.op_name == "builtin.module"
    assert op_info.src_rng == SourceRange.unknown()
    assert op_info.parent is None
    assert op_info.children == []


def test_create_operation_index(tmp_path: Path) -> None:
    mlir_path = _create_simple_module(tmp_path)
    ctx, mod = _load_module(mlir_path)

    src_remap = SourceReMap(disabled_files={mlir_path})
    op_index = OperationIndex.create(mod.operation, src_remap)

    assert len(op_index) == 5, "Ops: module, func.func, arith.addi, arith.muli, return"

    expected_ids = [
        "builtin.module[0]",
        "builtin.module[0]/func.func[1]",
        "builtin.module[0]/func.func[1]/%0",
        "builtin.module[0]/func.func[1]/%1",
        "builtin.module[0]/func.func[1]/func.return[4]",
    ]
    assert expected_ids == list(
        uid for uid, _ in op_index
    ), "Operation IDs do not match expected"

    # Check get by unique id
    op_info = op_index.get_op("builtin.module[0]/func.func[1]/%0")
    assert op_info is not None
    assert op_info.ssa_name == "%0"
    assert op_info.src_rng == SourceRange.from_str("source.mlir:10:5")
    assert op_index.get_op("non_existent") is None

    # Check get by operation
    op_info = op_index.get_info_by_op(mod.operation)
    assert op_info is not None
    assert op_info.op_name == "builtin.module"
    assert len(op_info.children) == 1
    assert op_info.get_combined_src() == SourceRange.from_str("source.mlir:10:5-12:3")

    # Check get ops by name
    ops = op_index.get_ops_by_name("%0")
    assert len(ops) == 1
    assert ops[0].ssa_name == "%0"
    assert ops[0].src_rng == SourceRange.from_str("source.mlir:10:5")
    assert not op_index.get_ops_by_name("%non_existent")

    # Check get ops by src range
    ops = op_index.get_ops_by_src_range(SourceRange.from_str("source.mlir:10:5"))
    assert len(ops) == 1
    assert ops[0].ssa_name == "%0"
    assert not op_index.get_ops_by_src_range(SourceRange.from_str("source.mlir:20:1"))

    # Check get ops by src loc
    ops = op_index.get_ops_by_src_loc(SourceLocation.from_str("source.mlir:10:5"))
    assert len(ops) == 1
    assert ops[0].ssa_name == "%0"
    assert not op_index.get_ops_by_src_loc(SourceLocation.from_str("source.mlir:20:1"))


if __name__ == "__main__":
    args = sys.argv[1:]
    ret_code = pytest.main([__file__, "--verbose", "-vv"] + args)

    sys.exit(ret_code)

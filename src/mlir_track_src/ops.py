import dataclasses
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from mlir.ir import AsmState, Operation, WalkOrder, WalkResult

from mlir_track_src.src_loc import (
    SourceLocation,
    SourceRange,
    SourceReMap,
)


def _build_src_range_from_op_loc(
    op_location: Any, source_remap: SourceReMap
) -> SourceRange:
    if op_location.is_a_callsite():
        return _build_src_range_from_op_loc(op_location.callee, source_remap)

    if op_location.is_a_fused():
        child_locations = op_location.locations
        if not child_locations:
            return SourceRange.unknown()
        locations = [
            _build_src_range_from_op_loc(loc, source_remap) for loc in child_locations
        ]
        # Merge all child locations
        merged_range = locations[0]
        for loc in locations[1:]:
            merged_range = merged_range.merge(loc)
        return merged_range

    if op_location.is_a_name():
        return SourceRange.unknown()

    # Unknown location
    if not op_location.is_a_file():
        return SourceRange.unknown()

    file_path_str: str | None = op_location.filename
    if file_path_str is None or source_remap.is_file_disabled(file_path_str):
        return SourceRange.unknown()

    file_path = Path(file_path_str)

    file_path = source_remap.remap_file(file_path)
    start_line = op_location.start_line
    start_col = op_location.start_col
    end_line = op_location.end_line
    end_col = op_location.end_col

    return SourceRange(
        start=SourceLocation(file_path=file_path, line=start_line, column=start_col),
        end=SourceLocation(file_path=file_path, line=end_line, column=end_col),
    )


def _build_src_range_from_op(
    op: Operation,
    source_remap: SourceReMap,
) -> SourceRange:
    if not op.location:
        return SourceRange.unknown()
    return _build_src_range_from_op_loc(op.location, source_remap)


@dataclasses.dataclass
class OperationInfo:
    """
    Information about a specific operation in an MLIR module

    Note: There are two main reasons for this clas to exist:
    1. Provide an easy way to access meta information about an operation, such as its
       source location, SSA name, and unique id.
    2. The MLIR python bindings are extremely slow when accessing operation properties.
    """

    unique_id: str
    ssa_name: str
    src_rng: SourceRange
    op_name: str
    op: Operation
    parent: "OperationInfo | None" = None
    children: list["OperationInfo"] = dataclasses.field(default_factory=list)

    def get_combined_src(self) -> SourceRange:
        """
        Returns the combined source range of this operation and all its
        children.

        For operators that can nest other operations, such as e.g.
        "linalg.generic", this can be useful to resolve the full source range
        when the operation was a result of fusion.
        """
        combined_range = self.src_rng
        for child in self.children:
            combined_range = combined_range.try_merge(child.get_combined_src())
        return combined_range

    @classmethod
    def create(
        cls,
        op: Operation,
        src_remap: SourceReMap,
        id_prefix: str = "",
        alternative_name: str = "",
        asm_state: AsmState | None = None,
    ) -> "OperationInfo":
        ssa_name = ""
        if len(op.results) > 0:
            if asm_state:
                ssa_name = op.results[0].get_name(asm_state)
            else:
                ssa_name = op.results[0].get_name()
        unique_id = f"{id_prefix}{ssa_name or alternative_name}"
        return cls(
            unique_id=unique_id,
            ssa_name=ssa_name,
            src_rng=_build_src_range_from_op(op, src_remap),
            op_name=op.name,
            op=op,
        )

    def dump(self, /, indent: int = 0, content: bool = False) -> None:
        indent_str = " " * indent
        print(f"{indent_str}OperationInfo:")
        print(f"{indent_str}  SSA Name: {self.ssa_name}")
        print(f"{indent_str}  Operation Name: {self.op_name}")
        print(f"{indent_str}  Source Range: {self.src_rng}")
        if content:
            print(f"{indent_str}  Source Content:\n{self.src_rng.get_content()}")


@dataclasses.dataclass
class OperationIndex:
    """
    Index of operations in an MLIR module for quick lookup by various criteria.
    """

    # Mapping from unique op Id to OperationInfo
    operations: dict[str, OperationInfo]

    # Mapping from Operation to OperationInfo
    info_by_op: dict[Operation, OperationInfo]

    @staticmethod
    def _build_op_mapping(
        op: Operation, src_remap: SourceReMap, log_progress: bool
    ) -> tuple[dict[str, OperationInfo], dict[Operation, OperationInfo]]:
        operations: dict[str, OperationInfo] = {}
        info_by_op: dict[Operation, OperationInfo] = {}
        asm_state = AsmState(op)

        def _cb(op: Operation) -> WalkResult:
            if log_progress:
                print(f"Building operation index {len(operations) + 1}", end="\r")
            parent_info = info_by_op.get(op.parent, None)
            id_prefix = ""
            if parent_info is not None:
                id_prefix = f"{parent_info.unique_id}/"

            alternative_name = f"{op.name}[{len(info_by_op)}]"

            op_info = OperationInfo.create(
                op,
                src_remap,
                id_prefix=id_prefix,
                alternative_name=alternative_name,
                asm_state=asm_state,
            )
            if parent_info is not None:
                op_info.parent = parent_info
                parent_info.children.append(op_info)
            if op_info.unique_id:
                operations[op_info.unique_id] = op_info
            info_by_op[op] = op_info
            return WalkResult.ADVANCE

        op.walk(_cb, WalkOrder.PRE_ORDER)
        if log_progress:
            print(f"Built operation index with {len(operations)} operations.")

        return operations, info_by_op

    @classmethod
    def create(
        cls, op: Operation, src_remap: SourceReMap, log_progress: bool = False
    ) -> "OperationIndex":
        operations, info_by_op = cls._build_op_mapping(op, src_remap, log_progress)
        return cls(operations=operations, info_by_op=info_by_op)

    def __len__(self) -> int:
        return len(self.operations)

    def __iter__(self) -> Iterator[tuple[str, OperationInfo]]:
        return iter(self.operations.items())

    def get_op(self, op_id: str) -> OperationInfo | None:
        """
        Get OperationInfo by its unique id.
        """
        return self.operations.get(op_id, None)

    def get_info_by_op(self, op: Operation) -> OperationInfo | None:
        """
        Get OperationInfo by its MLIR Operation.
        """
        return self.info_by_op.get(op, None)

    def get_ops_by_name(self, ssa_name: str) -> list[OperationInfo]:
        """
        Get all OperationInfo instances with the given SSA name.
        """
        result: list[OperationInfo] = []
        for op_info in self.operations.values():
            if op_info.ssa_name == ssa_name:
                result.append(op_info)
        return result

    def get_ops_by_type(self, op_name: str) -> list[OperationInfo]:
        """
        Get all OperationInfo instances with the given operation name.
        """
        result: list[OperationInfo] = []
        for op_info in self.operations.values():
            if op_info.op_name == op_name:
                result.append(op_info)
        return result

    def get_ops_by_src_loc(self, src_loc: SourceLocation) -> list[OperationInfo]:
        """
        Get all OperationInfo instances that contain the given source location.
        """
        result: list[OperationInfo] = []
        for op_info in self.operations.values():
            if op_info.src_rng.contains(src_loc):
                result.append(op_info)
        return result

    def get_ops_by_src_range(self, src_rng: SourceRange) -> list[OperationInfo]:
        """
        Get all OperationInfo instances that overlap with the given source range.
        """
        result: list[OperationInfo] = []
        for op_info in self.operations.values():
            if op_info.src_rng.overlaps(src_rng):
                result.append(op_info)
        return result

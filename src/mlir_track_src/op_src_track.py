from collections.abc import Sequence

from mlir.ir import Operation

from mlir_track_src.ops import OperationIndex, OperationInfo
from mlir_track_src.src_loc import SourceRange


def unique_ops(
    op_infos: Sequence[OperationInfo | None],
) -> list[OperationInfo]:
    """
    Given a list of OperationInfo instances, return a list with duplicates
    removed, preserving the order of first occurrence.
    """

    seen_ops: set[Operation] = set()
    unique_op_infos: list[OperationInfo] = []
    for op_info in op_infos:
        if op_info and op_info.op not in seen_ops:
            unique_op_infos.append(op_info)
            seen_ops.add(op_info.op)
    return unique_op_infos


def track(
    oi: OperationInfo | None | list[OperationInfo] | SourceRange,
    track_op_index: OperationIndex,
    use_combined_src: bool = False,
) -> list[OperationInfo]:
    """
    Track operations from one module to another based on source information.

    Args:
        oi: The tracking input to find corresponding operations
        track_op_index: The OperationIndex of the module to track into
        use_combined_src: Only for OperationInfo input, whether to use the
                          combined source range (including children) for tracking

    Returns:
        A list of OperationInfo instances in the tracked module that correspond
        to the tracking input
    """

    if oi is None:
        return []
    if isinstance(oi, SourceRange):
        return track_op_index.get_ops_by_src_range(oi)
    if isinstance(oi, OperationInfo):
        if use_combined_src:
            return track_op_index.get_ops_by_src_range(oi.get_combined_src())
        return track_op_index.get_ops_by_src_range(oi.src_rng)
    result: list[OperationInfo] = []
    for single_oi in oi:
        result.extend(track(single_oi, track_op_index, use_combined_src))
    return unique_ops(result)


def _check_any_parent_in_set(op_info: OperationInfo, all_ops: set[Operation]) -> bool:
    parent = op_info.parent
    while parent is not None:
        if parent.op in all_ops:
            return True
        parent = parent.parent
    return False


def shrink_to_parents(ops: list[OperationInfo]) -> list[OperationInfo]:
    """
    Given a list of OperationInfo instances, return a list where child
    operations are removed if their parent operation is also in the list.
    """

    all_ops: set[Operation] = {oi.op for oi in ops}
    to_skip: set[Operation] = set()
    for oi in ops:
        if _check_any_parent_in_set(oi, all_ops):
            to_skip.add(oi.op)
    result: list[OperationInfo] = []
    for oi in ops:
        if oi.op not in to_skip:
            result.append(oi)
    return result


def _add_all_parents_to_set(
    op_info: OperationInfo,
    to_skip: set[Operation],
) -> None:
    parent = op_info.parent
    while parent is not None:
        to_skip.add(parent.op)
        parent = parent.parent


def shrink_to_children(ops: list[OperationInfo]) -> list[OperationInfo]:
    """
    Given a list of OperationInfo instances, return a list where parent
    operations are removed if their child operation is also in the list.
    """

    to_skip: set[Operation] = set()
    for oi in ops:
        _add_all_parents_to_set(oi, to_skip)
    result: list[OperationInfo] = []
    for oi in ops:
        if oi.op not in to_skip:
            result.append(oi)
    return result

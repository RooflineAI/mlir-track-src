# Copyright 2026 RooflineAI GmbH
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

try:
    from mlir.ir import Context, Module, AsmState, Operation, WalkOrder, WalkResult
except ImportError as e:
    try:
        from iree.compiler.ir import (
            Context,
            Module,
            AsmState,
            Operation,
            WalkOrder,
            WalkResult,
        )
    except ImportError:
        raise ImportError(
            "Failed to import MLIR Python bindings. "
            "Please ensure they are installed and accessible."
        ) from e

__all__ = [
    "Context",
    "Module",
    "AsmState",
    "Operation",
    "WalkOrder",
    "WalkResult",
]

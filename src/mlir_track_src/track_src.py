import atexit
import code
import readline
from collections.abc import Callable
from pathlib import Path
from typing import Annotated, Any, Optional

import typer
from mlir.ir import Context, Module

from mlir_track_src.op_src_track import shrink_to_children, shrink_to_parents, track
from mlir_track_src.ops import OperationIndex, OperationInfo
from mlir_track_src.src_loc import SourceRange, SourceReMap

InputMlirOption = Annotated[
    Path,
    typer.Option(
        help="Path to the input MLIR file to track the source location for",
        resolve_path=True,
        exists=True,
    ),
]

TrackMlirOption = Annotated[
    Optional[Path],
    typer.Option(
        help="Path to the MLIR file to track operations from the input MLIR file",
        resolve_path=True,
        exists=True,
    ),
]

SourceReMapOption = Annotated[
    Optional[list[str]],
    typer.Option(
        help="List of source file remappings in the format original_path=mapped_path.",
    ),
]


def _setup_readline() -> None:
    histfile = Path.home() / ".mlir_track_src_track_src_history"
    try:
        readline.read_history_file(histfile)
    except (FileNotFoundError, PermissionError):
        pass
    readline.set_history_length(1000)
    atexit.register(readline.write_history_file, histfile)


def _dump_infos(
    op_infos: list[OperationInfo], /, indent: int = 0, content: bool = False
) -> None:
    if not isinstance(op_infos, list) or not all(
        isinstance(info, OperationInfo) for info in op_infos
    ):
        print("Invalid input, expected list[OperationInfo]")
        return
    for op_info in op_infos:
        op_info.dump(indent=indent, content=content)


def _show_ops(ops: list[OperationInfo]) -> None:
    sep = ""
    for oi in sorted(shrink_to_parents(ops), key=lambda x: x.src_rng):
        print(sep, end="")
        print(str(oi.op))
        sep = "-" * 40 + "\n"


def _show(oi: OperationInfo | None | list[OperationInfo]) -> None:
    if oi is None:
        print("None")
    if not oi:
        print(oi)
    elif isinstance(oi, OperationInfo):
        print(str(oi.op))
    elif isinstance(oi, list) and all(isinstance(info, OperationInfo) for info in oi):
        _show_ops(oi)
    else:
        print("Invalid input to show function.")


def _show_src(oi: OperationInfo | None | list[OperationInfo], sep: str = "") -> None:
    if oi is None:
        return
    if not oi:
        return
    if isinstance(oi, OperationInfo):
        print(sep, end="")
        print(oi.src_rng.get_content())
        return
    if not isinstance(oi, list) or not all(
        isinstance(info, OperationInfo) for info in oi
    ):
        print("Invalid input to src function.")
    all_src_ranges = {i.src_rng for i in oi}
    for src_rng in sorted(all_src_ranges):
        print(sep, end="")
        print(src_rng.get_content())
        sep = "-" * 40 + "\n"


class ShellObject:
    def __init__(self, func: Callable, short_doc: str, doc: str = "") -> None:
        self._func = func
        self._short_doc = short_doc
        self.__doc__ = func.__doc__ or doc

    @property
    def short_doc(self) -> str:
        return self._short_doc

    def __call__(self, *args: Any, **kwargs: Any):
        return self._func(*args, **kwargs)

    def __repr__(self) -> str:
        help(self._func)
        return ""

    @staticmethod
    def help_override(obj: Any) -> None:
        if isinstance(obj, ShellObject):
            help(obj._func)
        else:
            help(obj)


class ReprObject:
    def __init__(self, content: str) -> None:
        self._content = content

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        print(self._content)

    def __repr__(self) -> str:
        return self._content


class ShellContext:
    def __init__(
        self,
        banner: str,
        commands: dict[str, ShellObject],
        extra_locals: dict[str, Any],
    ) -> None:
        self.banner = banner
        self._commands: dict[str, Any] = commands
        self._extra_locals = extra_locals
        self._extra_locals["info"] = ReprObject(
            "Available commands (type help(command_name) for details):\n"
            + "\n".join(f"  {name}: {cmd.short_doc}" for name, cmd in commands.items())
            + "\n\nExtra locals:\n"
            + "\n".join(
                f"  {name} - {type(val).__name__}" for name, val in extra_locals.items()
            )
        )
        self._extra_locals["help"] = ShellObject.help_override

    def interact(self) -> None:
        all_locals = dict(self._commands)
        all_locals.update(self._extra_locals)
        code.interact(banner=self.banner, local=all_locals)


def _wrap(callable: Callable, wrapped: Callable, defaults: list[str]) -> Callable:
    doc = wrapped.__doc__ or ""
    if defaults:
        doc = "Defaults:\n" + "\n".join(f"       {d}" for d in defaults) + "\n\n" + doc
    callable.__doc__ = doc
    return callable


def _start_shell(
    op_index: OperationIndex, track_op_index: OperationIndex | None
) -> None:
    def track_back(
        oi: OperationInfo | None | SourceRange | list[OperationInfo],
    ) -> list[OperationInfo]:
        return track(oi, op_index, use_combined_src=True)

    def related(
        oi: OperationInfo | None | SourceRange | list[OperationInfo],
    ) -> list[OperationInfo]:
        return track(oi, op_index)

    commands = {
        "dump_infos": ShellObject(
            func=_dump_infos,
            short_doc="Dump OperationInfo objects",
        ),
        "track_back": ShellObject(
            func=_wrap(
                track_back, track, defaults=["op_index", "use_combined_src=True"]
            ),
            short_doc="Track operations from tracking MLIR back to input MLIR",
        ),
        "shrink_to_parents": ShellObject(
            func=shrink_to_parents,
            short_doc="Shrink a list of OperationInfo to only include parents",
        ),
        "shrink_to_children": ShellObject(
            func=shrink_to_children,
            short_doc="Shrink a list of OperationInfo to only include children",
        ),
        "related": ShellObject(
            func=_wrap(related, track, defaults=["op_index", "use_combined_src=False"]),
            short_doc="Find related operations in the input MLIR (that have same source)",
        ),
        "show": ShellObject(
            func=_show,
            short_doc="Show the MLIR operations for the given OperationInfo(s)",
        ),
        "src": ShellObject(
            func=_show_src,
            short_doc="Show the source code for the given OperationInfo(s)",
        ),
    }
    if track_op_index is not None:

        def _track(
            oi: OperationInfo | None | SourceRange | list[OperationInfo],
        ) -> list[OperationInfo]:
            return track(oi, track_op_index)

        commands["track"] = ShellObject(
            func=_wrap(
                _track, track, defaults=["track_op_index", "use_combined_src=False"]
            ),
            short_doc="Track operations from input MLIR to tracking MLIR",
        )

    shell = ShellContext(
        banner="Interactive shell. Type 'info' for available commands.",
        commands=commands,
        extra_locals={
            "op_index": op_index,
            "track_op_index": track_op_index,
        },
    )
    shell.interact()


def track_src(
    input_mlir: InputMlirOption,
    track_mlir: TrackMlirOption = None,
    src_remap_list: SourceReMapOption = None,
) -> None:
    _setup_readline()

    src_remap = SourceReMap.create(
        pairs=src_remap_list or [],
        place_holders={
            "%i": str(input_mlir),
            "%t": str(track_mlir) if track_mlir else "",
        },
    )

    context = Context()

    print("Loading input MLIR module...")
    module: Module = Module.parseFile(str(input_mlir), context)
    track_module: Module | None = None
    if track_mlir is not None:
        print("Loading tracking MLIR module...")
        track_module = Module.parseFile(str(track_mlir), context)

    print("Building operation index...")
    op_index = OperationIndex.create(module.operation, src_remap, log_progress=True)
    track_op_index: OperationIndex | None = None
    if track_module is not None:
        print("Building tracking operation index...")
        track_op_index = OperationIndex.create(
            track_module.operation, src_remap, log_progress=True
        )

    print("Starting interactive shell...")
    _start_shell(op_index, track_op_index)

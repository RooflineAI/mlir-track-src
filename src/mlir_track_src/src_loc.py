import dataclasses
import re
from pathlib import Path
from typing import Self


@dataclasses.dataclass(frozen=True, eq=True)
class SourceLocation:
    """
    Represents a specific location in a source file.
    """

    file_path: Path
    line: int
    column: int
    is_unknown: bool = False

    def __post_init__(self) -> None:
        if self.line < 1:
            raise ValueError(f"Line number must be greater than or equal to 1: {self}")

        if self.column < 1:
            raise ValueError(
                f"Column number must be greater than or equal to 1: {self}"
            )

    def __str__(self) -> str:
        return f"{self.file_path}:{self.line}:{self.column}"

    def __lt__(self, other: "SourceLocation") -> bool:
        if self.file_path != other.file_path:
            return str(self.file_path) < str(other.file_path)
        if self.line != other.line:
            return self.line < other.line
        return self.column < other.column

    @classmethod
    def from_str(cls: type[Self], s: str) -> Self:
        ptrn = re.compile(r"^(.*):(\d+):(\d+)$")
        match = ptrn.match(s)
        if not match:
            raise ValueError(
                f"Invalid source location format: {s}. Expected format: file_path:line:column"
            )
        file_path, line, column = match.groups()
        return cls(
            file_path=Path(file_path),
            line=int(line),
            column=int(column),
        )

    @classmethod
    def from_strs(cls: type[Self], s: list[str]) -> list[Self]:
        return [cls.from_str(item) for item in s]

    @classmethod
    def unknown(cls: type[Self]) -> Self:
        return cls(file_path=Path("unknown"), line=1, column=1, is_unknown=True)


@dataclasses.dataclass(frozen=True, eq=True)
class SourceRange:
    # Start of the source range (inclusive)
    start: SourceLocation
    # End of the source range (inclusive)
    end: SourceLocation

    def __post_init__(self) -> None:
        if self.start.file_path != self.end.file_path:
            raise ValueError("Start and end locations must be in the same file.")
        if self.start.is_unknown != self.end.is_unknown:
            raise ValueError(
                "Start and end locations must both be unknown or both be known."
            )

    def __str__(self) -> str:
        if self.start == self.end:
            return str(self.start)
        return (
            f"{self.start.file_path}:{self.start.line}:{self.start.column}-"
            f"{self.end.line}:{self.end.column}"
        )

    def __lt__(self, other: "SourceRange") -> bool:
        if self.start != other.start:
            return self.start < other.start
        return self.end < other.end

    @classmethod
    def from_str(cls: type[Self], s: str) -> Self:
        try:
            loc = SourceLocation.from_str(s)
            return cls(start=loc, end=loc)
        except ValueError:
            pass
        ptrn = re.compile(r"^(.*):(\d+):(\d+)-(\d+):(\d+)$")
        match = ptrn.match(s)
        if not match:
            raise ValueError(
                f"Invalid source range format: {s}. "
                "Expected format: file_path:start_line:start_column-end_line:end_column"
            )
        file_path, start_line, start_column, end_line, end_column = match.groups()
        return cls(
            start=SourceLocation(
                file_path=Path(file_path),
                line=int(start_line),
                column=int(start_column),
            ),
            end=SourceLocation(
                file_path=Path(file_path),
                line=int(end_line),
                column=int(end_column),
            ),
        )

    @classmethod
    def from_strs(cls: type[Self], s: list[str]) -> list[Self]:
        return [cls.from_str(item) for item in s]

    @property
    def file_path(self) -> Path:
        return self.start.file_path

    @property
    def is_unknown(self) -> bool:
        return self.start.is_unknown

    def contains(self, loc: SourceLocation) -> bool:
        """
        Returns true if the given source location is contained within this
        source range.
        """

        if self.start.file_path != loc.file_path:
            return False
        if loc.line < self.start.line or loc.line > self.end.line:
            return False
        if loc.line == self.start.line and loc.column < self.start.column:
            return False
        if loc.line == self.end.line and loc.column > self.end.column:
            return False
        return True

    def overlaps(self, other: "SourceRange") -> bool:
        """
        Returns true if there is any overlap between other and self

        E.g.
            [-------self-------]
                        [----other----]
        """
        if self.start.file_path != other.start.file_path:
            return False
        if self.end.line < other.start.line or other.end.line < self.start.line:
            return False
        if self.end.line == other.start.line and self.end.column < other.start.column:
            return False
        if other.end.line == self.start.line and other.end.column < self.start.column:
            return False
        return True

    def merge(self, other: "SourceRange") -> "SourceRange":
        """
        Merges two SourceRanges into one that encompasses both

        E.g.
            [-------self-------]
                        [----other----]
        results in
            [-------------------------]

        or
            [-------self-------]
                                      [----other----]
        results in
            [---------------------------------------]
        """
        if self.is_unknown:
            return other
        if other.is_unknown:
            return self
        if self.start.file_path != other.start.file_path:
            raise ValueError(
                "Cannot merge SourceRanges from different files: "
                f"{self.start.file_path} != {other.start.file_path}"
            )

        start = self.start
        if (other.start.line < start.line) or (
            other.start.line == start.line and other.start.column < start.column
        ):
            start = other.start

        end = self.end
        if (other.end.line > end.line) or (
            other.end.line == end.line and other.end.column > end.column
        ):
            end = other.end

        return SourceRange(start=start, end=end)

    def try_merge(self, other: "SourceRange") -> "SourceRange":
        """
        Attempts to merge two SourceRanges. If they cannot be merged,
        returns self.
        """
        try:
            return self.merge(other)
        except ValueError:
            return self

    def get_content(self) -> str:
        """
        Returns the content of the source range from the file.

        FIXME with only source start locations reported we should always report
        entire lines
        """

        with self.file_path.open("r") as f:
            lines = f.readlines()

        if self.start.line == self.end.line:
            if self.start.column == self.end.column:
                return lines[self.start.line - 1]  # entire line
            line = lines[self.start.line - 1]
            return line[self.start.column - 1 : self.end.column - 1]

        content_lines = []
        # First line
        first_line = lines[self.start.line - 1]
        content_lines.append(first_line[self.start.column - 1 :])

        # Middle lines
        for line_num in range(self.start.line, self.end.line - 1):
            content_lines.append(lines[line_num])

        # Last line
        last_line = lines[self.end.line - 1]
        content_lines.append(last_line[: self.end.column - 1])

        return "".join(content_lines)

    @classmethod
    def unknown(cls: type[Self]) -> Self:
        return cls(start=SourceLocation.unknown(), end=SourceLocation.unknown())


class SourceReMap:
    """
    Represents a mapping from original source file paths to mapped source file
    paths.

    Additionally allows to mark certain files as disabled, meaning that any
    source locations in those files are treated as unknown.
    """

    def __init__(
        self,
        file_mappings: dict[Path, Path] | None = None,
        disabled_files: set[Path] | None = None,
    ) -> None:
        self._file_mappings: dict[Path, Path] = file_mappings or {}
        self._disabled_files: set[Path] = disabled_files or set()

    def add_file_mapping(self, original: Path, mapped: Path) -> None:
        if original in self._file_mappings:
            raise ValueError(f"Mapping for {original} already exists.")
        self._file_mappings[original] = mapped

    def add_disabled_file(self, file_path: str | Path) -> None:
        self._disabled_files.add(Path(file_path))

    def is_file_disabled(self, file_path: str | Path) -> bool:
        return Path(file_path) in self._disabled_files

    def remap_file(self, original: str | Path) -> Path:
        return self._file_mappings.get(Path(original), Path(original))

    @classmethod
    def create(
        cls, pairs: list[str], place_holders: dict[str, str] | None = None
    ) -> "SourceReMap":
        place_holders = place_holders or {}

        remap = cls()
        for pair in pairs:
            mapping = pair.split("=")
            if len(mapping) != 2:
                raise ValueError(
                    f"Invalid mapping format: {pair}. Expected format: original_path=mapped_path"
                )
            original_str, mapped_str = mapping
            original_str = place_holders.get(original_str, original_str)
            mapped_str = place_holders.get(mapped_str, mapped_str)
            original = Path(original_str.strip())
            mapped = Path(mapped_str.strip())
            remap.add_file_mapping(original, mapped)
        return remap

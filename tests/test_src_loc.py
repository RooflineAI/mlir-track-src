import sys
from pathlib import Path

import pytest
from mlir_track_src.src_loc import SourceLocation, SourceRange, SourceReMap


def test_source_location_invalid() -> None:
    with pytest.raises(ValueError):
        SourceLocation(file_path=Path("example.mlir"), line=0, column=5)
    with pytest.raises(ValueError):
        SourceLocation(file_path=Path("example.mlir"), line=10, column=0)


def test_source_location_str() -> None:
    loc = SourceLocation(file_path=Path("example.mlir"), line=10, column=5)
    assert str(loc) == "example.mlir:10:5"

    loc2 = SourceLocation.from_str("example.mlir:10:5")
    assert loc == loc2

    unknown_loc = SourceLocation.unknown()
    assert str(unknown_loc) == "unknown:1:1"
    assert unknown_loc.is_unknown


def test_source_location_sorting() -> None:
    locs = SourceLocation.from_strs(
        [
            "example.mlir:10:5",
            "example.mlir:2:15",
            "example.mlir:10:3",
        ]
    )
    locs.sort()
    assert [str(loc) for loc in locs] == [
        "example.mlir:2:15",
        "example.mlir:10:3",
        "example.mlir:10:5",
    ]


def test_source_location_equality() -> None:
    loc1 = SourceLocation.from_str("example.mlir:10:5")
    loc2 = SourceLocation.from_str("example.mlir:10:5")
    loc3 = SourceLocation.from_str("example.mlir:10:6")
    assert loc1 == loc2
    assert loc1 != loc3


def test_source_range_invalid() -> None:
    loc1 = SourceLocation.from_str("example.mlir:10:5")
    loc2 = SourceLocation.from_str("other.mlir:12:3")
    with pytest.raises(ValueError):
        SourceRange(start=loc1, end=loc2)

    assert SourceRange(start=loc1, end=loc1).file_path == Path("example.mlir")


def test_source_range_str() -> None:
    loc1 = SourceLocation.from_str("example.mlir:10:5")
    loc2 = SourceLocation.from_str("example.mlir:10:15")
    range1 = SourceRange(start=loc1, end=loc2)
    assert str(range1) == "example.mlir:10:5-10:15"
    assert range1 == SourceRange.from_str("example.mlir:10:5-10:15")

    loc3 = SourceLocation.from_str("example.mlir:12:3")
    range2 = SourceRange(start=loc3, end=loc3)
    assert str(range2) == "example.mlir:12:3"
    assert range2 == SourceRange.from_str("example.mlir:12:3")

    unknown_range = SourceRange.unknown()
    assert str(unknown_range) == "unknown:1:1"
    assert unknown_range.is_unknown


def test_source_range_sorting() -> None:
    range_list = SourceRange.from_strs(
        [
            "example.mlir:10:5",
            "example.mlir:10:5-10:15",
            "example.mlir:9:20-10:2",
            "example.mlir:10:4",
            "example.mlir:10:5-11:1",
        ]
    )
    range_list.sort()
    assert [str(r) for r in range_list] == [
        "example.mlir:9:20-10:2",
        "example.mlir:10:4",
        "example.mlir:10:5",
        "example.mlir:10:5-10:15",
        "example.mlir:10:5-11:1",
    ]


def test_source_range_equality() -> None:
    range1 = SourceRange.from_str("example.mlir:10:5-10:15")
    range2 = SourceRange.from_str("example.mlir:10:5-10:15")
    range3 = SourceRange.from_str("example.mlir:10:5-11:1")
    assert range1 == range2
    assert range1 != range3


def test_source_range_contains() -> None:
    range1 = SourceRange.from_str("example.mlir:10:5-10:15")
    loc_inside = SourceLocation.from_str("example.mlir:10:10")
    loc_outside_line_after = SourceLocation.from_str("example.mlir:11:1")
    loc_outside_line_before = SourceLocation.from_str("example.mlir:9:20")
    loc_outside_column_after = SourceLocation.from_str("example.mlir:10:16")
    loc_outside_column_before = SourceLocation.from_str("example.mlir:10:4")
    loc_different_file = SourceLocation.from_str("other.mlir:10:10")

    assert range1.contains(loc_inside)
    assert not range1.contains(loc_outside_line_after)
    assert not range1.contains(loc_outside_line_before)
    assert not range1.contains(loc_outside_column_after)
    assert not range1.contains(loc_outside_column_before)
    assert not range1.contains(loc_different_file)


def test_source_range_overlaps() -> None:
    range1 = SourceRange.from_str("example.mlir:10:5-10:15")
    range_overlap_start = SourceRange.from_str("example.mlir:10:10-10:20")
    range_overlap_end = SourceRange.from_str("example.mlir:10:1-10:10")
    range_contained = SourceRange.from_str("example.mlir:10:7-10:12")
    range_no_overlap_before = SourceRange.from_str("example.mlir:9:3-9:30")
    range_no_overlap_after = SourceRange.from_str("example.mlir:11:3-11:30")
    range_different_file = SourceRange.from_str("other.mlir:10:5-10:15")

    assert range1.overlaps(range_overlap_start)
    assert range_overlap_start.overlaps(range1)
    assert range1.overlaps(range_overlap_end)
    assert range_overlap_end.overlaps(range1)
    assert range1.overlaps(range_contained)
    assert range_contained.overlaps(range1)
    assert not range1.overlaps(range_no_overlap_before)
    assert not range_no_overlap_before.overlaps(range1)
    assert not range1.overlaps(range_no_overlap_after)
    assert not range_no_overlap_after.overlaps(range1)
    assert not range1.overlaps(range_different_file)
    assert not range_different_file.overlaps(range1)


def test_source_range_overlap_range_loc() -> None:
    range1 = SourceRange.from_str("example.mlir:10:5")
    range2 = SourceRange.from_str("example.mlir:10:10")
    range3 = SourceRange.from_str("example.mlir:10:5-10:15")

    assert not range1.overlaps(range2)
    assert not range2.overlaps(range1)
    assert range1.overlaps(range3)
    assert range3.overlaps(range1)


def test_source_range_merge() -> None:
    # Overlapping ranges
    range1 = SourceRange.from_str("example.mlir:10:5-10:15")
    range2 = SourceRange.from_str("example.mlir:10:10-10:20")
    merged = range1.merge(range2)
    assert merged == SourceRange.from_str("example.mlir:10:5-10:20")

    # Non-overlapping ranges
    range3 = SourceRange.from_str("example.mlir:11:4-11:10")
    merged2 = range1.merge(range3)
    assert merged2 == SourceRange.from_str("example.mlir:10:5-11:10")

    # Merging with unknown range
    range_unknown = SourceRange.unknown()
    assert range1.merge(range_unknown) == range1
    assert range_unknown.merge(range1) == range1


def test_source_range_try_merge() -> None:
    range1 = SourceRange.from_str("example.mlir:10:5-10:15")
    range2 = SourceRange.from_str("example.mlir:10:10-10:20")
    range3 = SourceRange.from_str("other.mlir:11:4-11:10")

    merged = range1.try_merge(range2)
    assert merged == SourceRange.from_str("example.mlir:10:5-10:20")

    not_merged = range1.try_merge(range3)
    assert not_merged == range1


def test_source_range_get_content(tmp_path: Path) -> None:
    # Create a temporary file with known contents
    temp_file = tmp_path / "temp_example.mlir"
    contents = [
        "1:3456789\n",
        "2:3456789\n",
        "3:3456789\n",
        "4:3456789\n",
    ]
    temp_file.write_text("".join(contents))

    # Single line range
    range1 = SourceRange.from_str(f"{temp_file}:2:1-2:6")
    extracted = range1.get_content()
    expected = "2:345"
    assert extracted == expected

    # Single location
    range2 = SourceRange.from_str(f"{temp_file}:3:1")
    extracted2 = range2.get_content()
    expected2 = "3:3456789\n"
    assert extracted2 == expected2

    # Multi-line range
    range3 = SourceRange.from_str(f"{temp_file}:2:5-4:4")
    extracted3 = range3.get_content()
    expected3 = "56789\n3:3456789\n4:3"
    assert extracted3 == expected3


def test_source_remap_add_mapping() -> None:
    remap = SourceReMap()
    path1 = Path("a.mlir")
    path2 = Path("b.mlir")
    remap.add_file_mapping(path1, path2)
    assert remap.remap_file(path1) == path2
    with pytest.raises(ValueError):
        remap.add_file_mapping(path1, Path("c.mlir"))


def test_source_remap_no_mapping() -> None:
    remap = SourceReMap()
    path1 = Path("a.mlir")
    assert remap.remap_file(path1) == path1


def test_source_remap_disabled_file() -> None:
    remap = SourceReMap()
    remap.add_disabled_file("disabled.mlir")
    assert remap.is_file_disabled("disabled.mlir")
    assert not remap.is_file_disabled("enabled.mlir")


def test_source_remap_create() -> None:
    remap = SourceReMap.create(
        pairs=[
            "source.mlir=mapped.mlir",
            "%t=remapped_tmp_file.mlir",
            "some_model.mlir=%s",
        ],
        place_holders={"%t": "tmp_file.mlir", "%s": "final_model.mlir"},
    )
    assert remap.remap_file(Path("source.mlir")) == Path("mapped.mlir")
    assert remap.remap_file(Path("tmp_file.mlir")) == Path("remapped_tmp_file.mlir")
    assert remap.remap_file(Path("some_model.mlir")) == Path("final_model.mlir")
    assert remap.remap_file(Path("unmapped.mlir")) == Path("unmapped.mlir")


if __name__ == "__main__":
    args = sys.argv[1:]
    ret_code = pytest.main([__file__, "--verbose", "-vv"] + args)

    sys.exit(ret_code)

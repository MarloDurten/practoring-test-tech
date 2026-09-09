"""Тесты раскладки папок отчётов."""

from __future__ import annotations

from pathlib import Path

from app.report_paths import build_run_layout, source_slug


def test_source_slug_file_with_spaces() -> None:
    slug = source_slug(r"C:\data\exam clip.mkv")
    assert slug == "exam_clip"


def test_source_slug_camera() -> None:
    assert source_slug(0) == "camera_0"
    assert source_slug("1") == "camera_1"


def test_layout_separates_csv_json_marked(tmp_path: Path) -> None:
    layout = build_run_layout(tmp_path, "clip name.mp4", "balanced", run_stamp="20260909_150000")
    layout.create()
    assert layout.source_slug == "clip_name"
    assert layout.csv_path.parent.name == "csv"
    assert layout.json_path.parent.name == "json"
    assert layout.marked_path.parent.name == "marked"
    assert layout.run_dir == tmp_path / "clip_name" / "20260909_150000"
    assert layout.csv_dir.is_dir()
    assert layout.json_dir.is_dir()
    assert layout.marked_dir.is_dir()


def test_two_sources_do_not_share_folder(tmp_path: Path) -> None:
    a = build_run_layout(tmp_path, "exam_a.mkv", "fast", run_stamp="t1")
    b = build_run_layout(tmp_path, "exam_b.mkv", "fast", run_stamp="t1")
    assert a.source_dir != b.source_dir

"""Раскладка каталогов отчётов: один ролик / один прогон — без смешивания файлов."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


_UNSAFE = re.compile(r"[^\w.\-]+", re.UNICODE)


def source_slug(source: str | int) -> str:
    """Стабильное имя папки для источника: файл → stem, камера → camera_N."""
    if isinstance(source, int) or str(source).strip().isdigit():
        return f"camera_{int(source)}"
    stem = Path(str(source)).stem.strip() or "source"
    slug = _UNSAFE.sub("_", stem).strip("._")
    return slug or "source"


@dataclass(frozen=True)
class RunLayout:
    """Куда класть артефакты одного прогона."""

    root: Path
    source_dir: Path
    run_dir: Path
    csv_dir: Path
    json_dir: Path
    marked_dir: Path
    logs_dir: Path
    csv_path: Path
    json_path: Path
    marked_path: Path
    log_path: Path
    source_slug: str
    run_stamp: str
    profile: str

    def create(self) -> "RunLayout":
        for folder in (self.csv_dir, self.json_dir, self.marked_dir, self.logs_dir):
            folder.mkdir(parents=True, exist_ok=True)
        return self

    def as_dict(self) -> dict:
        return {
            "root": str(self.root),
            "source_dir": str(self.source_dir),
            "run_dir": str(self.run_dir),
            "csv_dir": str(self.csv_dir),
            "json_dir": str(self.json_dir),
            "marked_dir": str(self.marked_dir),
            "logs_dir": str(self.logs_dir),
            "csv_path": str(self.csv_path),
            "json_path": str(self.json_path),
            "marked_path": str(self.marked_path),
            "log_path": str(self.log_path),
            "source_slug": self.source_slug,
            "run_stamp": self.run_stamp,
            "profile": self.profile,
        }


def build_run_layout(
    output_root: str | Path,
    source: str | int,
    profile: str,
    run_stamp: str | None = None,
    marked_path: str | None = None,
    marked_ext: str = ".mp4",
) -> RunLayout:
    """
    reports/
      <имя_видео_или_camera_N>/
        <YYYYMMDD_HHMMSS>/
          csv/frames_<profile>.csv
          json/summary_<profile>.json
          marked/marked_<profile>.mp4
          logs/run.log
    """
    root = Path(output_root)
    slug = source_slug(source)
    stamp = run_stamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    source_dir = root / slug
    run_dir = source_dir / stamp
    csv_dir = run_dir / "csv"
    json_dir = run_dir / "json"
    marked_dir = run_dir / "marked"
    logs_dir = run_dir / "logs"
    layout = RunLayout(
        root=root,
        source_dir=source_dir,
        run_dir=run_dir,
        csv_dir=csv_dir,
        json_dir=json_dir,
        marked_dir=marked_dir,
        logs_dir=logs_dir,
        csv_path=csv_dir / f"frames_{profile}.csv",
        json_path=json_dir / f"summary_{profile}.json",
        marked_path=Path(marked_path) if marked_path else marked_dir / f"marked_{profile}{marked_ext}",
        log_path=logs_dir / "run.log",
        source_slug=slug,
        run_stamp=stamp,
        profile=profile,
    )
    return layout

"""CLI: нагрузочный прогон анализа видео с веб-камеры или файла."""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

import cv2

from app.analyzers.pipeline import AnalysisPipeline, draw_overlay
from app.config import QUALITY_PROFILES, AppConfig, ModuleToggles
from app.human_report import format_human_report
from app.media import COMPRESSION_PRESETS, open_video_writer, preset_by_id
from app.metrics import format_console_summary
from app.report_paths import build_run_layout
from app.video_capture import marked_output_fps


def build_parser() -> argparse.ArgumentParser:
    profiles = ", ".join(QUALITY_PROFILES)
    parser = argparse.ArgumentParser(
        prog="python -m app.main",
        description=(
            "Тестовое приложение для оценки серверной нагрузки при анализе видео: "
            "отвлечение от экрана и сигналы подозрительных (возможно AI-generated) кадров. "
            "Разрешение камеры подбирать не нужно — кадры нормализуются автоматически."
        ),
    )
    parser.add_argument(
        "--source",
        default=None,
        help="Индекс веб-камеры (0, 1, ...) или путь к видеофайлу. Если не указан — интерактивное меню.",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Выбор видео и сжатия стрелками в терминале (режим по умолчанию без --source).",
    )
    parser.add_argument(
        "--dir",
        default=None,
        help="Папка, в которой искать видео для интерактивного выбора (по умолчанию текущая).",
    )
    parser.add_argument(
        "--profile",
        choices=list(QUALITY_PROFILES),
        default="balanced",
        help=f"Профиль качества: {profiles}.",
    )
    parser.add_argument(
        "--resize-mode",
        choices=["letterbox", "center_crop"],
        default="letterbox",
        help="Сохранение aspect ratio: letterbox (padding) или center_crop.",
    )
    parser.add_argument(
        "--max-duration",
        type=float,
        default=600.0,
        help="Максимальная длительность обработки в секундах (по умолчанию 600 = 10 мин).",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Опциональный лимит числа прочитанных кадров (удобно для смоука).",
    )
    parser.add_argument(
        "--warmup-frames",
        type=int,
        default=5,
        help="Сколько первых обработанных кадров считать прогревом MediaPipe.",
    )
    parser.add_argument(
        "--output-dir",
        default="reports",
        help="Корневой каталог отчётов. Внутри создаются папки источника и прогона.",
    )
    parser.add_argument(
        "--flat-output",
        action="store_true",
        help="Писать CSV/JSON/marked прямо в --output-dir, без подпапок на источник.",
    )
    parser.add_argument(
        "--preview",
        action="store_true",
        help="Показать окно OpenCV с оверлеем (локальный прогон, не для сервера).",
    )
    parser.add_argument(
        "--save-marked",
        action="store_true",
        help="Сохранить видео с маркировками (bbox, взгляд, distraction, AI-сигналы).",
    )
    parser.add_argument(
        "--marked-path",
        default=None,
        help="Путь к marked-видео. По умолчанию reports/<видео>/<прогон>/marked/...",
    )
    parser.add_argument(
        "--codec",
        choices=[p.id for p in COMPRESSION_PRESETS],
        default="mp4v",
        help="Формат сжатия marked-видео: mp4v, avc1, xvid, mjpg.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )

    toggles = parser.add_argument_group("Модули (можно выключать по одному)")
    toggles.add_argument("--no-face", action="store_true", help="Выключить face tracking.")
    toggles.add_argument("--no-gaze", action="store_true", help="Выключить gaze estimation.")
    toggles.add_argument(
        "--no-distraction", action="store_true", help="Выключить distraction heuristics."
    )
    toggles.add_argument(
        "--no-ai-heuristics",
        action="store_true",
        help="Выключить ai_generated_frame_heuristics.",
    )
    return parser


def config_from_args(args: argparse.Namespace) -> AppConfig:
    source: str | int = args.source
    if str(source).isdigit():
        source = int(source)

    modules = ModuleToggles(
        face_tracking=not args.no_face,
        gaze_estimation=not args.no_gaze,
        distraction_heuristics=not args.no_distraction,
        ai_generated_frame_heuristics=not args.no_ai_heuristics,
    )
    if not modules.face_tracking and modules.gaze_estimation:
        logging.getLogger(__name__).warning(
            "Gaze включён, но face tracking выключен — взгляд будет недоступен."
        )

    return AppConfig(
        profile=args.profile,
        resize_mode=args.resize_mode,
        source=source,
        max_duration_sec=args.max_duration,
        max_frames=args.max_frames,
        warmup_frames=args.warmup_frames,
        output_dir=args.output_dir,
        preview=args.preview,
        save_marked=bool(args.save_marked or args.marked_path),
        marked_path=args.marked_path,
        flat_output=bool(args.flat_output),
        compression_id=args.codec,
        modules=modules,
    )


def setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level),
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def prepare_stdio() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass
    if hasattr(sys.stderr, "reconfigure"):
        try:
            sys.stderr.reconfigure(encoding="utf-8")
        except Exception:
            pass
    if sys.platform == "win32":
        try:
            import ctypes

            handle = ctypes.windll.kernel32.GetStdHandle(-11)
            mode = ctypes.c_uint()
            if ctypes.windll.kernel32.GetConsoleMode(handle, ctypes.byref(mode)):
                ctypes.windll.kernel32.SetConsoleMode(handle, mode.value | 0x0004)
        except Exception:
            pass


def run_session(cfg: AppConfig, quiet_logs: bool = False) -> int:
    log = logging.getLogger("app.main")
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    try:
        codec = preset_by_id(cfg.compression_id)
    except KeyError:
        codec = preset_by_id("mp4v")

    if cfg.flat_output:
        out_dir = Path(cfg.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        csv_path = out_dir / f"frames_{cfg.profile}_{stamp}.csv"
        json_path = out_dir / f"summary_{cfg.profile}_{stamp}.json"
        marked_path = (
            Path(cfg.marked_path)
            if cfg.marked_path
            else out_dir / f"marked_{cfg.profile}_{stamp}{codec.extension}"
        )
        run_dir = out_dir
        layout_info = {"mode": "flat", "run_dir": str(run_dir)}
    else:
        layout = build_run_layout(
            cfg.output_dir,
            cfg.source,
            cfg.profile,
            run_stamp=stamp,
            marked_path=cfg.marked_path,
            marked_ext=codec.extension,
        ).create()
        csv_path = layout.csv_path
        json_path = layout.json_path
        marked_path = layout.marked_path
        run_dir = layout.run_dir
        layout_info = layout.as_dict()
        layout_info["mode"] = "isolated"
    if cfg.save_marked:
        marked_path.parent.mkdir(parents=True, exist_ok=True)

    log.info("Папка прогона: %s", run_dir)

    pipeline = AnalysisPipeline(cfg)
    preview_enabled = bool(cfg.preview)
    save_marked = bool(cfg.save_marked)
    marked_writer: cv2.VideoWriter | None = None
    used_fourcc = codec.fourcc

    def on_frame(bgr, analysis, record) -> None:
        nonlocal marked_writer, marked_path, used_fourcc
        vis = draw_overlay(bgr, analysis, record)
        if save_marked:
            if marked_writer is None:
                h, w = vis.shape[:2]
                info = pipeline._source_info or {}
                fps = marked_output_fps(info.get("reported_fps"), cfg.frame_stride)
                marked_writer, used_fourcc, marked_path = open_video_writer(
                    marked_path, fps, (w, h), codec.fourcc
                )
                log.info(
                    "Пишу маркировки в %s (%.1f fps, %dx%d, fourcc=%s)",
                    marked_path,
                    fps,
                    w,
                    h,
                    used_fourcc,
                )
            marked_writer.write(vis)
        if preview_enabled:
            cv2.imshow("proctoring-load-test", vis)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                raise KeyboardInterrupt

    log.info(
        "Запуск: source=%s profile=%s mode=%s working=%sx%s stride=%s save_marked=%s codec=%s",
        cfg.source,
        cfg.profile,
        cfg.resize_mode,
        cfg.target_size[0],
        cfg.target_size[1],
        cfg.frame_stride,
        save_marked,
        codec.id,
    )

    need_callback = preview_enabled or save_marked
    try:
        summary = pipeline.run(on_frame=on_frame if need_callback else None)
    except FileNotFoundError as exc:
        log.error("%s", exc)
        print(exc)
        return 2
    except KeyboardInterrupt:
        log.warning("Остановлено пользователем")
        summary = pipeline.metrics.summarize(
            extra={
                "profile": cfg.profile,
                "source": str(cfg.source),
                "resize_mode": cfg.resize_mode,
                "interrupted": True,
            }
        )
    except Exception:
        log.exception("Критическая ошибка пайплайна")
        return 1
    finally:
        if marked_writer is not None:
            marked_writer.release()
        if preview_enabled:
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass

    pipeline.metrics.write_csv(csv_path)
    summary["csv_path"] = str(csv_path.resolve())
    summary["json_path"] = str(json_path.resolve())
    summary["report_dir"] = str(Path(run_dir).resolve())
    summary["report_layout"] = layout_info
    summary["compression"] = (
        {
            "id": codec.id,
            "label": codec.label,
            "fourcc": used_fourcc,
            "extension": Path(marked_path).suffix,
        }
        if save_marked
        else {"label": "не сохранялось"}
    )
    if save_marked and Path(marked_path).exists():
        summary["marked_video_path"] = str(Path(marked_path).resolve())
    pipeline.metrics.write_json(json_path, summary)

    print(format_human_report(summary))
    if not quiet_logs:
        print(format_console_summary(summary))
        print(f"REPORT_DIR: {run_dir}")
        print(f"CSV:  {csv_path}")
        print(f"JSON: {json_path}")
        if save_marked:
            print(f"MARKED: {marked_path}")
    return 0


def main(argv: list[str] | None = None) -> int:
    prepare_stdio()
    parser = build_parser()
    args = parser.parse_args(argv)

    interactive = bool(args.interactive or args.source is None)
    if interactive:
        from app.interactive import run_interactive

        scan = Path(args.dir) if args.dir else Path.cwd()
        return run_interactive(scan)

    setup_logging(args.log_level)
    cfg = config_from_args(args)
    return run_session(cfg, quiet_logs=False)


if __name__ == "__main__":
    sys.exit(main())

"""Оркестрация: захват → нормализация → preprocess → infer → postprocess → метрики."""

from __future__ import annotations

import logging
import time
from typing import Callable, Optional

import cv2
import numpy as np

from app.analyzers.ai_heuristics import AiGeneratedHeuristics
from app.analyzers.distraction import DistractionHeuristics
from app.analyzers.face_tracking import FaceTracker
from app.analyzers.gaze_estimation import GazeEstimator
from app.analyzers.postprocessing import FrameAnalysis, Postprocessor
from app.analyzers.preprocessing import FramePreprocessor
from app.config import AppConfig
from app.metrics import FrameRecord, MetricsCollector
from app.normalization import FrameNormalizer
from app.video_capture import CapturedFrame, VideoSource

logger = logging.getLogger(__name__)


def _estimate_duration(source_info: dict | None, records: list) -> float:
    info = source_info or {}
    fps = float(info.get("reported_fps") or 0.0)
    hinted = int(info.get("frame_count_hint") or 0)
    if fps > 1.0 and hinted > 0:
        return hinted / fps
    if records:
        return float(records[-1].timestamp_sec)
    n = len(records)
    if fps > 1.0 and n:
        return n / fps
    return 0.0


class AnalysisPipeline:
    def __init__(self, config: AppConfig) -> None:
        self.config = config
        w, h = config.target_size
        self.normalizer = FrameNormalizer(w, h, mode=config.resize_mode)
        self.preprocessor = FramePreprocessor(clahe=False)
        self.face_tracker: FaceTracker | None = None
        self.gaze_estimator = GazeEstimator(enabled=config.modules.gaze_estimation)
        self.distraction = DistractionHeuristics(
            thresholds=config.distraction,
            enabled=config.modules.distraction_heuristics,
        )
        self.ai_heuristics = AiGeneratedHeuristics(
            thresholds=config.ai,
            enabled=config.modules.ai_generated_frame_heuristics,
        )
        self.post = Postprocessor()
        self.metrics = MetricsCollector()
        self._processed_index = 0
        self._source_info = None

    def _ensure_face_tracker(self) -> FaceTracker:
        # Тяжёлую инициализацию MediaPipe откладываем, пока источник точно открыт.
        if self.face_tracker is None:
            self.face_tracker = FaceTracker(enabled=self.config.modules.face_tracking)
        return self.face_tracker

    def run(
        self,
        on_frame: Optional[Callable[[np.ndarray, FrameAnalysis, FrameRecord], None]] = None,
    ) -> dict:
        cfg = self.config
        self.metrics.start()
        processed_after_warmup = 0

        try:
            with VideoSource(
                source=cfg.source,
                max_duration_sec=cfg.max_duration_sec,
                max_frames=cfg.max_frames,
                hint_width=cfg.target_size[0],
                hint_height=cfg.target_size[1],
            ) as source:
                self._source_info = source.info.as_dict()
                self._ensure_face_tracker()
                logger.info(
                    "Старт пайплайна: profile=%s stride=%s working=%sx%s modules=%s",
                    cfg.profile,
                    cfg.frame_stride,
                    cfg.target_size[0],
                    cfg.target_size[1],
                    cfg.modules,
                )
                for captured in source.frames():
                    record, analysis, preview = self._handle_frame(captured)
                    self.metrics.add(record)
                    if (
                        not record.skipped
                        and record.processed_index >= cfg.warmup_frames
                    ):
                        processed_after_warmup += 1
                    if on_frame is not None and analysis is not None and preview is not None:
                        on_frame(preview, analysis, record)
        finally:
            self.metrics.stop()
            self.close()

        extra = {
            "profile": cfg.profile,
            "source": str(cfg.source),
            "resize_mode": cfg.resize_mode,
            "frame_stride": cfg.frame_stride,
            "warmup_frames": cfg.warmup_frames,
            "modules": {
                "face_tracking": cfg.modules.face_tracking,
                "gaze_estimation": cfg.modules.gaze_estimation,
                "distraction_heuristics": cfg.modules.distraction_heuristics,
                "ai_generated_frame_heuristics": cfg.modules.ai_generated_frame_heuristics,
            },
            "source_info": self._source_info,
            "processed_after_warmup": processed_after_warmup,
            "source_duration_sec": round(_estimate_duration(self._source_info, self.metrics.records), 4),
        }
        return self.metrics.summarize(extra=extra)

    def _handle_frame(
        self, captured: CapturedFrame
    ) -> tuple[FrameRecord, Optional[FrameAnalysis], Optional[np.ndarray]]:
        t0 = time.perf_counter()
        stride = max(1, self.config.frame_stride)
        should_process = (captured.source_index % stride) == 0

        resources = self.metrics.sample_resources()
        t_cap = time.perf_counter()

        if not should_process:
            process_ms = (time.perf_counter() - t0) * 1000.0
            record = self._empty_record(
                captured,
                skipped=True,
                process_ms=process_ms,
                capture_ms=(t_cap - t0) * 1000.0,
                resources=resources,
            )
            return record, None, None

        # --- нормализация ---
        t_n0 = time.perf_counter()
        working, norm = self.normalizer.process(captured.image)
        t_n1 = time.perf_counter()

        # --- предобработка ---
        pre = self.preprocessor.process(working)
        t_p1 = time.perf_counter()

        # --- CV-инференс ---
        face = self._ensure_face_tracker().infer(pre.rgb)
        gaze = self.gaze_estimator.infer(face)
        distraction = self.distraction.infer(face, gaze, captured.timestamp_sec)
        ai = self.ai_heuristics.infer(pre.gray, face)
        t_i1 = time.perf_counter()

        analysis = self.post.merge(face, gaze, distraction, ai, norm)
        t_post = time.perf_counter()

        process_ms = (t_post - t0) * 1000.0
        record = FrameRecord(
            source_frame_index=captured.source_index,
            processed_index=self._processed_index,
            skipped=False,
            timestamp_sec=captured.timestamp_sec,
            source_width=captured.source_width,
            source_height=captured.source_height,
            working_width=norm.working_width,
            working_height=norm.working_height,
            process_ms=process_ms,
            capture_ms=(t_n0 - t0) * 1000.0,
            normalize_ms=(t_n1 - t_n0) * 1000.0,
            preprocess_ms=(t_p1 - t_n1) * 1000.0,
            infer_ms=(t_i1 - t_p1) * 1000.0,
            postprocess_ms=(t_post - t_i1) * 1000.0,
            cpu_process_percent=resources["cpu_process_percent"],
            cpu_system_percent=resources["cpu_system_percent"],
            ram_mb=resources["ram_mb"],
            gpu_util_percent=resources["gpu_util_percent"],
            vram_mb=resources["vram_mb"],
            face_detected=face.detected,
            face_confidence=face.confidence,
            head_yaw=face.head_yaw,
            head_pitch=face.head_pitch,
            head_roll=face.head_roll,
            gaze_yaw=gaze.yaw_deg,
            gaze_pitch=gaze.pitch_deg,
            ear=face.mean_ear,
            blink=distraction.blink,
            distraction=distraction.distracted,
            distraction_reasons=analysis.reasons_csv(),
            ai_suspicion_score=ai.score,
            ai_suspicious=ai.suspicious,
            ai_signals=analysis.ai_signals_csv(),
        )
        self._processed_index += 1
        return record, analysis, pre.bgr

    def _empty_record(
        self,
        captured: CapturedFrame,
        skipped: bool,
        process_ms: float,
        capture_ms: float,
        resources: dict,
    ) -> FrameRecord:
        w, h = self.config.target_size
        return FrameRecord(
            source_frame_index=captured.source_index,
            processed_index=-1,
            skipped=skipped,
            timestamp_sec=captured.timestamp_sec,
            source_width=captured.source_width,
            source_height=captured.source_height,
            working_width=w,
            working_height=h,
            process_ms=process_ms,
            capture_ms=capture_ms,
            normalize_ms=0.0,
            preprocess_ms=0.0,
            infer_ms=0.0,
            postprocess_ms=0.0,
            cpu_process_percent=resources["cpu_process_percent"],
            cpu_system_percent=resources["cpu_system_percent"],
            ram_mb=resources["ram_mb"],
            gpu_util_percent=resources["gpu_util_percent"],
            vram_mb=resources["vram_mb"],
            face_detected=False,
            face_confidence=0.0,
            head_yaw=None,
            head_pitch=None,
            head_roll=None,
            gaze_yaw=None,
            gaze_pitch=None,
            ear=None,
            blink=False,
            distraction=False,
            distraction_reasons="",
            ai_suspicion_score=0.0,
            ai_suspicious=False,
            ai_signals="",
        )

    def close(self) -> None:
        if self.face_tracker is not None:
            self.face_tracker.close()
        self.metrics.close()


def draw_overlay(bgr: np.ndarray, analysis: FrameAnalysis, record: FrameRecord) -> np.ndarray:
    """Необязательный preview: bbox, углы, флаги. Не влияет на бенчмарк инференса."""
    vis = bgr.copy()
    face = analysis.face
    if face.detected and face.bbox is not None:
        x, y, w, h = face.bbox
        color = (0, 165, 255) if analysis.distraction.distracted else (0, 200, 0)
        if analysis.ai.suspicious:
            color = (0, 0, 255)
        cv2.rectangle(vis, (x, y), (x + w, y + h), color, 2)
        if face.landmarks_px is not None:
            for pt in face.landmarks_px[::8]:
                cv2.circle(vis, (int(pt[0]), int(pt[1])), 1, (255, 255, 0), -1)

    lines = [
        f"src {record.source_width}x{record.source_height} -> {record.working_width}x{record.working_height}",
        f"lat {record.process_ms:.1f} ms  infer {record.infer_ms:.1f} ms",
        f"face {int(face.detected)} conf {face.confidence:.2f} yaw {face.head_yaw or 0:.1f}",
        f"gaze yaw {analysis.gaze.yaw_deg or 0:.1f} pitch {analysis.gaze.pitch_deg or 0:.1f}",
        f"distraction {analysis.distraction.distracted} {analysis.reasons_csv()[:48]}",
        f"ai {analysis.ai.score:.2f} {analysis.ai_signals_csv()[:48]}",
    ]
    y = 18
    for line in lines:
        cv2.putText(vis, line, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(vis, line, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (240, 240, 240), 1, cv2.LINE_AA)
        y += 18
    return vis

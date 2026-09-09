"""Трекинг лица и 3D-позы головы.

MediaPipe 0.10.x до ~0.10.14: mp.solutions.face_mesh
MediaPipe 0.10.30+: только Tasks API (FaceLandmarker) + файл .task

Приложение выбирает бэкенд само. Модель FaceLandmarker скачивается один раз
в app/models/ при первом запуске (нужен интернет).
"""

from __future__ import annotations

import logging
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)

try:
    import mediapipe as mp

    _MP_AVAILABLE = True
except Exception:  # noqa: BLE001
    mp = None  # type: ignore
    _MP_AVAILABLE = False

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
LANDMARKER_NAME = "face_landmarker.task"
LANDMARKER_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "face_landmarker/face_landmarker/float16/latest/face_landmarker.task"
)

# Индексы Face Mesh / FaceLandmarker (478 точек, включая iris).
NOSE_TIP = 1
CHIN = 152
LEFT_EYE_OUTER = 33
LEFT_EYE_INNER = 133
RIGHT_EYE_OUTER = 263
RIGHT_EYE_INNER = 362
LEFT_MOUTH = 61
RIGHT_MOUTH = 291

LEFT_EAR_IDX = (33, 160, 158, 133, 153, 144)
RIGHT_EAR_IDX = (362, 385, 387, 263, 373, 380)

LEFT_IRIS_CENTER = 468
RIGHT_IRIS_CENTER = 473

_FACE_3D = np.array(
    [
        (0.0, 0.0, 0.0),
        (0.0, -330.0, -65.0),
        (-225.0, 170.0, -135.0),
        (225.0, 170.0, -135.0),
        (-150.0, -150.0, -125.0),
        (150.0, -150.0, -125.0),
    ],
    dtype=np.float64,
)


@dataclass
class FaceResult:
    detected: bool = False
    confidence: float = 0.0
    landmarks_norm: Optional[np.ndarray] = None
    landmarks_px: Optional[np.ndarray] = None
    visibilities: Optional[np.ndarray] = None
    bbox: Optional[tuple[int, int, int, int]] = None
    head_yaw: Optional[float] = None
    head_pitch: Optional[float] = None
    head_roll: Optional[float] = None
    left_ear: Optional[float] = None
    right_ear: Optional[float] = None
    mean_ear: Optional[float] = None
    left_iris_norm: Optional[tuple[float, float]] = None
    right_iris_norm: Optional[tuple[float, float]] = None
    landmark_count: int = 0
    extra: dict = field(default_factory=dict)


def ensure_landmarker_model(dest_dir: Path = MODELS_DIR) -> Path:
    """Скачать face_landmarker.task, если файла ещё нет."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    path = dest_dir / LANDMARKER_NAME
    if path.exists() and path.stat().st_size > 100_000:
        return path
    logger.info("Скачиваю MediaPipe FaceLandmarker: %s", LANDMARKER_URL)
    tmp = path.with_suffix(".task.part")
    try:
        with urllib.request.urlopen(LANDMARKER_URL, timeout=60) as resp, tmp.open("wb") as fh:
            while True:
                chunk = resp.read(1024 * 256)
                if not chunk:
                    break
                fh.write(chunk)
        tmp.replace(path)
    except Exception as exc:
        if tmp.exists():
            tmp.unlink(missing_ok=True)
        raise RuntimeError(
            "Не удалось скачать модель FaceLandmarker. "
            "Скачайте вручную:\n"
            f"  {LANDMARKER_URL}\n"
            f"и положите в {path}\n"
            f"Причина: {exc}"
        ) from exc
    logger.info("Модель сохранена: %s (%.1f МБ)", path, path.stat().st_size / 1e6)
    return path


def _eye_aspect_ratio(landmarks: np.ndarray, idx: tuple[int, ...]) -> Optional[float]:
    if landmarks.shape[0] <= max(idx):
        return None
    p1, p2, p3, p4, p5, p6 = (landmarks[i] for i in idx)
    vert1 = np.linalg.norm(p2 - p6)
    vert2 = np.linalg.norm(p3 - p5)
    hor = np.linalg.norm(p1 - p4)
    if hor < 1e-6:
        return None
    return float((vert1 + vert2) / (2.0 * hor))


def _rotation_matrix_to_euler(rot: np.ndarray) -> tuple[float, float, float]:
    sy = float(np.sqrt(rot[0, 0] ** 2 + rot[1, 0] ** 2))
    singular = sy < 1e-6
    if not singular:
        pitch = float(np.degrees(np.arctan2(rot[2, 1], rot[2, 2])))
        yaw = float(np.degrees(np.arctan2(-rot[2, 0], sy)))
        roll = float(np.degrees(np.arctan2(rot[1, 0], rot[0, 0])))
    else:
        pitch = float(np.degrees(np.arctan2(-rot[1, 2], rot[1, 1])))
        yaw = float(np.degrees(np.arctan2(-rot[2, 0], sy)))
        roll = 0.0
    return yaw, pitch, roll


def _rvec_to_euler(rvec: np.ndarray) -> tuple[float, float, float]:
    rot, _ = cv2.Rodrigues(rvec)
    return _rotation_matrix_to_euler(rot)


class FaceTracker:
    """MediaPipe Face Mesh / FaceLandmarker + pose + EAR."""

    def __init__(
        self,
        max_faces: int = 1,
        refine_landmarks: bool = True,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        enabled: bool = True,
        model_path: Optional[Path] = None,
    ) -> None:
        self.enabled = enabled
        self.refine_landmarks = refine_landmarks
        self._backend = "none"
        self._mesh = None
        self._landmarker = None
        if not enabled:
            return
        if not _MP_AVAILABLE:
            raise RuntimeError(
                "mediapipe не установлен. Установите: pip install mediapipe\n"
                "Рекомендуется Python 3.11; на 3.12 нужен mediapipe>=0.10.30 (Tasks API)."
            )
        if hasattr(mp, "solutions"):
            self._init_solutions(
                max_faces, refine_landmarks, min_detection_confidence, min_tracking_confidence
            )
        else:
            self._init_tasks(
                max_faces,
                min_detection_confidence,
                min_tracking_confidence,
                model_path,
            )

    def _init_solutions(
        self,
        max_faces: int,
        refine_landmarks: bool,
        min_detection_confidence: float,
        min_tracking_confidence: float,
    ) -> None:
        self._mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=max_faces,
            refine_landmarks=refine_landmarks,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self._backend = "solutions"
        logger.info("FaceTracker: MediaPipe solutions.face_mesh")

    def _init_tasks(
        self,
        max_faces: int,
        min_detection_confidence: float,
        min_tracking_confidence: float,
        model_path: Optional[Path],
    ) -> None:
        from mediapipe.tasks.python import BaseOptions
        from mediapipe.tasks.python.vision import (
            FaceLandmarker,
            FaceLandmarkerOptions,
            RunningMode,
        )

        path = Path(model_path) if model_path else ensure_landmarker_model()
        options = FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(path)),
            running_mode=RunningMode.IMAGE,
            num_faces=max_faces,
            min_face_detection_confidence=min_detection_confidence,
            min_face_presence_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=True,
        )
        self._landmarker = FaceLandmarker.create_from_options(options)
        self._backend = "tasks"
        logger.info("FaceTracker: MediaPipe Tasks FaceLandmarker (%s)", path.name)

    def infer(self, rgb: np.ndarray) -> FaceResult:
        if not self.enabled:
            return FaceResult(extra={"disabled": True})
        if rgb is None or rgb.size == 0:
            return FaceResult()
        try:
            if self._backend == "solutions":
                return self._infer_solutions(rgb)
            if self._backend == "tasks":
                return self._infer_tasks(rgb)
            raise RuntimeError("FaceTracker не инициализирован")
        except Exception as exc:  # noqa: BLE001
            logger.exception("Face tracker упал: %s", exc)
            return FaceResult(extra={"error": str(exc)})

    def _infer_solutions(self, rgb: np.ndarray) -> FaceResult:
        results = self._mesh.process(rgb)
        if not results.multi_face_landmarks:
            return FaceResult()
        lm = results.multi_face_landmarks[0]
        pts_norm = np.array([[p.x, p.y] for p in lm.landmark], dtype=np.float32)
        vis = np.array(
            [getattr(p, "visibility", 1.0) or 0.0 for p in lm.landmark],
            dtype=np.float32,
        )
        presence = np.array(
            [getattr(p, "presence", 1.0) or 0.0 for p in lm.landmark],
            dtype=np.float32,
        )
        confidence = float(np.clip(np.mean(np.maximum(vis, presence)), 0.0, 1.0))
        return self._pack(rgb.shape[1], rgb.shape[0], pts_norm, vis, confidence)

    def _infer_tasks(self, rgb: np.ndarray) -> FaceResult:
        if rgb.dtype != np.uint8:
            rgb = np.clip(rgb, 0, 255).astype(np.uint8)
        rgb = np.ascontiguousarray(rgb)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self._landmarker.detect(mp_image)
        if not result.face_landmarks:
            return FaceResult()
        lm = result.face_landmarks[0]
        pts_norm = np.array([[p.x, p.y] for p in lm], dtype=np.float32)
        vis = np.array(
            [getattr(p, "visibility", 1.0) if getattr(p, "visibility", None) is not None else 1.0 for p in lm],
            dtype=np.float32,
        )
        presence = np.array(
            [getattr(p, "presence", 1.0) if getattr(p, "presence", None) is not None else 1.0 for p in lm],
            dtype=np.float32,
        )
        confidence = float(np.clip(np.mean(np.maximum(vis, presence)), 0.0, 1.0))
        packed = self._pack(rgb.shape[1], rgb.shape[0], pts_norm, vis, confidence)

        matrices = getattr(result, "facial_transformation_matrixes", None) or []
        if len(matrices) > 0:
            mat = np.array(matrices[0], dtype=np.float64)
            if mat.shape[0] >= 3 and mat.shape[1] >= 3:
                packed.head_yaw, packed.head_pitch, packed.head_roll = _rotation_matrix_to_euler(mat[:3, :3])
                packed.extra["pose_source"] = "facial_transformation_matrix"
        return packed

    def _pack(
        self,
        width: int,
        height: int,
        pts_norm: np.ndarray,
        vis: np.ndarray,
        confidence: float,
    ) -> FaceResult:
        pts_px = np.stack([pts_norm[:, 0] * width, pts_norm[:, 1] * height], axis=1)
        xs, ys = pts_px[:, 0], pts_px[:, 1]
        x0, y0 = int(np.clip(xs.min(), 0, width - 1)), int(np.clip(ys.min(), 0, height - 1))
        x1, y1 = int(np.clip(xs.max(), 0, width - 1)), int(np.clip(ys.max(), 0, height - 1))
        bbox = (x0, y0, max(1, x1 - x0), max(1, y1 - y0))

        left_ear = _eye_aspect_ratio(pts_px, LEFT_EAR_IDX)
        right_ear = _eye_aspect_ratio(pts_px, RIGHT_EAR_IDX)
        ears = [e for e in (left_ear, right_ear) if e is not None]
        mean_ear = float(sum(ears) / len(ears)) if ears else None

        yaw = pitch = roll = None
        try:
            yaw, pitch, roll = self._estimate_head_pose(pts_px, width, height)
        except Exception as exc:  # noqa: BLE001
            logger.debug("solvePnP pose недоступна: %s", exc)

        left_iris = right_iris = None
        if pts_norm.shape[0] > RIGHT_IRIS_CENTER:
            left_iris = (float(pts_norm[LEFT_IRIS_CENTER, 0]), float(pts_norm[LEFT_IRIS_CENTER, 1]))
            right_iris = (float(pts_norm[RIGHT_IRIS_CENTER, 0]), float(pts_norm[RIGHT_IRIS_CENTER, 1]))

        return FaceResult(
            detected=True,
            confidence=confidence,
            landmarks_norm=pts_norm,
            landmarks_px=pts_px,
            visibilities=vis,
            bbox=bbox,
            head_yaw=yaw,
            head_pitch=pitch,
            head_roll=roll,
            left_ear=left_ear,
            right_ear=right_ear,
            mean_ear=mean_ear,
            left_iris_norm=left_iris,
            right_iris_norm=right_iris,
            landmark_count=int(pts_norm.shape[0]),
        )

    def _estimate_head_pose(
        self, pts_px: np.ndarray, width: int, height: int
    ) -> tuple[float, float, float]:
        needed = [NOSE_TIP, CHIN, LEFT_EYE_OUTER, RIGHT_EYE_OUTER, LEFT_MOUTH, RIGHT_MOUTH]
        if pts_px.shape[0] <= max(needed):
            raise RuntimeError("Недостаточно landmarks для solvePnP")
        image_points = np.array([pts_px[i] for i in needed], dtype=np.float64)
        focal = float(width)
        cam = np.array(
            [[focal, 0.0, width / 2.0], [0.0, focal, height / 2.0], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        dist = np.zeros((4, 1), dtype=np.float64)
        ok, rvec, _tvec = cv2.solvePnP(
            _FACE_3D, image_points, cam, dist, flags=cv2.SOLVEPNP_ITERATIVE
        )
        if not ok:
            raise RuntimeError("solvePnP не сошёлся")
        return _rvec_to_euler(rvec)

    def close(self) -> None:
        if self._mesh is not None:
            try:
                self._mesh.close()
            except Exception:
                pass
            self._mesh = None
        if self._landmarker is not None:
            try:
                self._landmarker.close()
            except Exception:
                pass
            self._landmarker = None

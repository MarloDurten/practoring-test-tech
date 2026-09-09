"""Конфигурация профилей качества, порогов эвристик и переключателей модулей."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Tuple


ResizeMode = Literal["letterbox", "center_crop"]
QualityProfileName = Literal["high_quality", "balanced", "fast"]

# Внутренние рабочие разрешения. Камера/файл могут быть любыми —
# кадр всегда приводится к этим размерам после нормализации.
QUALITY_PROFILES: dict[str, dict] = {
    "high_quality": {
        "width": 1280,
        "height": 720,
        "frame_stride": 1,  # обрабатываем каждый кадр
        "description": "Максимальное качество: все кадры, 1280x720",
    },
    "balanced": {
        "width": 640,
        "height": 480,
        "frame_stride": 3,  # 1 кадр из 3
        "description": "Баланс качества и нагрузки: 1/3 кадров, 640x480",
    },
    "fast": {
        "width": 640,
        "height": 360,
        "frame_stride": 5,  # 1 кадр из 5
        "description": "Минимальная нагрузка: 1/5 кадров, 640x360",
    },
}

# Если исходный кадр больше этого числа пикселей — сначала cheap downscale,
# затем letterbox/crop. Не опираемся на cap.set() камеры.
MAX_INPUT_PIXELS = 1920 * 1080

# Верхняя граница длительности входного ролика (и сессии с камеры).
DEFAULT_MAX_DURATION_SEC = 10 * 60

# Если OpenCV не сообщает FPS контейнера/камеры, и захват, и marked-видео
# используют один и тот же fallback, чтобы timestamp_sec и длительность ролика совпадали.
DEFAULT_FALLBACK_FPS = 30.0


@dataclass
class ModuleToggles:
    """Включение/выключение отдельных стадий пайплайна."""

    face_tracking: bool = True
    gaze_estimation: bool = True
    distraction_heuristics: bool = True
    ai_generated_frame_heuristics: bool = True


@dataclass
class DistractionThresholds:
    """Пороги простых эвристик отвлечения от экрана."""

    # Лицо отсутствует N обработанных кадров подряд.
    missing_face_frames: int = 15
    # Взгляд отклонён от оси камеры (градусы).
    gaze_yaw_deg: float = 25.0
    gaze_pitch_deg: float = 20.0
    # Поворот головы (градусы, solvePnP).
    head_yaw_deg: float = 30.0
    head_pitch_deg: float = 25.0
    # Eye Aspect Ratio: ниже порога — глаз закрыт.
    ear_closed: float = 0.21
    # Сколько обработанных кадров с закрытыми глазами считаем «закрыл глаза».
    eyes_closed_frames: int = 12
    # Аномальная частота морганий (в минуту, по скользящему окну).
    blink_rate_high_per_min: float = 45.0
    blink_rate_low_per_min: float = 4.0
    blink_window_sec: float = 30.0
    # Минимум секунд наблюдения, прежде чем судить о низкой частоте морганий.
    blink_min_observe_sec: float = 20.0


@dataclass
class AiHeuristicThresholds:
    """Пороги сигналов «подозрительный / возможно AI-generated кадр»."""

    # Средняя видимость landmarks MediaPipe.
    low_confidence: float = 0.45
    # Скачок координат landmarks (нормированных 0..1) между соседними кадрами.
    landmark_jump: float = 0.12
    # Нестабильность: std смещения по окну.
    landmark_jitter_std: float = 0.035
    # Laplacian variance лица слишком низкая (излишне гладкое / GAN-like).
    face_too_smooth: float = 18.0
    # Laplacian слишком высокая относительно фона (артефакты / шум).
    face_artifact_ratio: float = 3.5
    # Подозрительный кадр, если суммарный score выше порога.
    suspicion_score: float = 0.55
    # Размер окна для оценки стабильности landmarks.
    jitter_window: int = 8


@dataclass
class AppConfig:
    """Полная конфигурация одного прогона."""

    profile: QualityProfileName = "balanced"
    resize_mode: ResizeMode = "letterbox"
    source: str | int = 0
    max_duration_sec: float = DEFAULT_MAX_DURATION_SEC
    max_frames: int | None = None
    warmup_frames: int = 5
    output_dir: str = "reports"
    preview: bool = False
    save_marked: bool = False
    marked_path: str | None = None
    flat_output: bool = False
    compression_id: str = "mp4v"
    camera_index: int = 0
    modules: ModuleToggles = field(default_factory=ModuleToggles)
    distraction: DistractionThresholds = field(default_factory=DistractionThresholds)
    ai: AiHeuristicThresholds = field(default_factory=AiHeuristicThresholds)

    @property
    def target_size(self) -> Tuple[int, int]:
        spec = QUALITY_PROFILES[self.profile]
        return int(spec["width"]), int(spec["height"])

    @property
    def frame_stride(self) -> int:
        return int(QUALITY_PROFILES[self.profile]["frame_stride"])

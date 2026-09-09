"""Эвристики отвлечения пользователя от экрана.

Не «детектор читерства», а набор простых правил для нагрузочного теста:
- лицо отсутствует N кадров;
- взгляд отклонен больше порога;
- голова повернута в сторону;
- аномальная частота морганий / глаза закрыты слишком долго.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Optional

from app.analyzers.face_tracking import FaceResult
from app.analyzers.gaze_estimation import GazeResult
from app.config import DistractionThresholds


@dataclass
class DistractionResult:
    distracted: bool = False
    reasons: list[str] = field(default_factory=list)
    missing_face_streak: int = 0
    blink: bool = False
    blink_rate_per_min: float = 0.0
    eyes_closed_streak: int = 0


class DistractionHeuristics:
    def __init__(self, thresholds: Optional[DistractionThresholds] = None, enabled: bool = True) -> None:
        self.enabled = enabled
        self.t = thresholds or DistractionThresholds()
        self._missing_streak = 0
        self._eyes_closed_streak = 0
        self._blink_open = True
        self._blink_times: deque[float] = deque()

    def reset(self) -> None:
        self._missing_streak = 0
        self._eyes_closed_streak = 0
        self._blink_open = True
        self._blink_times.clear()

    def infer(
        self,
        face: FaceResult,
        gaze: GazeResult,
        timestamp_sec: float,
    ) -> DistractionResult:
        if not self.enabled:
            return DistractionResult()

        reasons: list[str] = []
        blink = False

        if not face.detected:
            self._missing_streak += 1
            self._eyes_closed_streak = 0
            self._blink_open = True
            if self._missing_streak >= self.t.missing_face_frames:
                reasons.append(f"face_missing:{self._missing_streak}")
        else:
            self._missing_streak = 0

            if face.head_yaw is not None and abs(face.head_yaw) > self.t.head_yaw_deg:
                reasons.append(f"head_yaw:{face.head_yaw:.1f}")
            if face.head_pitch is not None and abs(face.head_pitch) > self.t.head_pitch_deg:
                reasons.append(f"head_pitch:{face.head_pitch:.1f}")

            if gaze.available:
                if gaze.yaw_deg is not None and abs(gaze.yaw_deg) > self.t.gaze_yaw_deg:
                    reasons.append(f"gaze_yaw:{gaze.yaw_deg:.1f}")
                if gaze.pitch_deg is not None and abs(gaze.pitch_deg) > self.t.gaze_pitch_deg:
                    reasons.append(f"gaze_pitch:{gaze.pitch_deg:.1f}")

            ear = face.mean_ear
            if ear is not None:
                closed = ear < self.t.ear_closed
                if closed:
                    self._eyes_closed_streak += 1
                    if self._blink_open:
                        blink = True
                        self._blink_open = False
                        self._blink_times.append(timestamp_sec)
                else:
                    self._eyes_closed_streak = 0
                    self._blink_open = True

                if self._eyes_closed_streak >= self.t.eyes_closed_frames:
                    reasons.append(f"eyes_closed:{self._eyes_closed_streak}")

        window = self.t.blink_window_sec
        while self._blink_times and timestamp_sec - self._blink_times[0] > window:
            self._blink_times.popleft()
        rate = 0.0
        if timestamp_sec > 1.0:
            observed = max(min(timestamp_sec, window), 1e-3)
            rate = len(self._blink_times) * (60.0 / observed)

        if face.detected and timestamp_sec >= self.t.blink_min_observe_sec:
            if rate > self.t.blink_rate_high_per_min:
                reasons.append(f"blink_rate_high:{rate:.1f}")
            elif rate < self.t.blink_rate_low_per_min and timestamp_sec >= self.t.blink_min_observe_sec:
                reasons.append(f"blink_rate_low:{rate:.1f}")

        return DistractionResult(
            distracted=bool(reasons),
            reasons=reasons,
            missing_face_streak=self._missing_streak,
            blink=blink,
            blink_rate_per_min=float(rate),
            eyes_closed_streak=self._eyes_closed_streak,
        )

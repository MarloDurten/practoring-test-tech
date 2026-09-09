"""Постобработка: агрегация событий кадра в удобную структуру для логов."""

from __future__ import annotations

from dataclasses import dataclass, field

from app.analyzers.ai_heuristics import AiHeuristicResult
from app.analyzers.distraction import DistractionResult
from app.analyzers.face_tracking import FaceResult
from app.analyzers.gaze_estimation import GazeResult
from app.normalization import NormalizationInfo


@dataclass
class FrameAnalysis:
    face: FaceResult
    gaze: GazeResult
    distraction: DistractionResult
    ai: AiHeuristicResult
    norm: NormalizationInfo
    extra: dict = field(default_factory=dict)

    def reasons_csv(self) -> str:
        return "|".join(self.distraction.reasons)

    def ai_signals_csv(self) -> str:
        return "|".join(self.ai.signals)


class Postprocessor:
    """Сейчас — тонкая фасовка результатов. Сюда же кладут NMS/сглаживание в prod."""

    def merge(
        self,
        face: FaceResult,
        gaze: GazeResult,
        distraction: DistractionResult,
        ai: AiHeuristicResult,
        norm: NormalizationInfo,
    ) -> FrameAnalysis:
        return FrameAnalysis(
            face=face,
            gaze=gaze,
            distraction=distraction,
            ai=ai,
            norm=norm,
        )

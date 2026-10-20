"""Request / response models for the ParkiScanML API."""
from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field


class SessionStart(BaseModel):
    session_id: str
    participant_id: Optional[str] = None
    language: Optional[str] = "en"
    client_time_utc: Optional[str] = None
    session_dir: Optional[str] = None
    source: Optional[str] = "unity"


class SessionStop(BaseModel):
    session_id: str


class TaskSubmit(BaseModel):
    record: dict[str, Any]
    raw_csv_path: Optional[str] = None
    language: Optional[str] = None


class SpiralPoint(BaseModel):
    t: float
    x: float
    y: float
    pressure: Optional[float] = None


class SpiralRequest(BaseModel):
    session_id: Optional[str] = None
    points: list[SpiralPoint] = Field(min_length=1)
    center: Optional[tuple[float, float]] = None
    reference: Optional[list[tuple[float, float]]] = None


class Trial(BaseModel):
    correct: Optional[bool] = None
    rt_ms: Optional[float] = None
    target: Optional[Any] = None
    response: Optional[Any] = None
    omitted: Optional[bool] = False
    span: Optional[int] = None


class CognitionRequest(BaseModel):
    session_id: Optional[str] = None
    task: str = "unspecified"
    trials: list[Trial]
    completion_time_s: Optional[float] = None


class GaitRequest(BaseModel):
    session_id: Optional[str] = None
    # option A: stride series in seconds
    left_stride_s: Optional[list[float]] = None
    right_stride_s: Optional[list[float]] = None
    stride_s: Optional[list[float]] = None
    # option B: accelerometer
    t: Optional[list[float]] = None
    ax: Optional[list[float]] = None
    ay: Optional[list[float]] = None
    az: Optional[list[float]] = None
    # option C: pose ankle heights
    left_ankle_y: Optional[list[float]] = None
    right_ankle_y: Optional[list[float]] = None


class PoseFrame(BaseModel):
    t: float
    landmarks: list[list[float]]


class PoseRequest(BaseModel):
    session_id: Optional[str] = None
    frames: list[PoseFrame]


class GazeSample(BaseModel):
    t: float
    x: Optional[float] = None
    y: Optional[float] = None
    valid: bool = True


class GazeTarget(BaseModel):
    t_on: float
    t_off: Optional[float] = None
    x: float
    y: float


class GazeRequest(BaseModel):
    session_id: Optional[str] = None
    samples: list[GazeSample]
    units: str = "norm"
    screen_width_deg: float = 40.0
    targets: Optional[list[GazeTarget]] = None


class ImuBatch(BaseModel):
    session_id: str
    packets: list[dict[str, Any]] = Field(min_length=1, max_length=5000)


class TtsRequest(BaseModel):
    text: str = Field(min_length=1, max_length=600)
    language: str = "en"

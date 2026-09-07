"""Central constants. Values traceable to the AutoVLA paper are cited inline."""

from dataclasses import dataclass, field

# --- Trajectory / tokenization (paper Sec. 3.1 + Appendix A) -----------------
DT = 0.5                 # seconds per action token; data resampled to 2 Hz
HORIZON_S = 5.0          # planning horizon
N_ACTION_TOKENS = 10     # HORIZON_S / DT
HISTORY_FRAMES = 4       # current + 3 preceding frames at 2 Hz (2 s of history)
CODEBOOK_SIZE = 2048     # K = 2048 balances reconstruction vs codebook usage
KDISK_DELTA = 0.05       # metres; no two codebook segments closer than this

# --- Ego vehicle footprint --------------------------------------------------
# nuScenes ego platform (Renault Zoe). Used for the contour distance that
# K-disk clustering is defined over.
EGO_LENGTH = 4.084
EGO_WIDTH = 1.730
CONTOUR_POINTS = 16      # points sampled along the bbox perimeter

# --- Kinematic bicycle model limits (physical feasibility check) ------------
WHEELBASE = 2.588
MAX_STEER = 0.61         # rad (~35 deg)
MAX_ACCEL = 3.0          # m/s^2
MAX_DECEL = -6.0         # m/s^2
MAX_SPEED = 25.0         # m/s (~90 km/h, urban upper bound)
MAX_CURVATURE = 0.2      # 1/m; ~5 m turning radius

# --- Cameras (paper Sec. 3.1: front, front-left, front-right) ---------------
CAMERAS = ("CAM_FRONT_LEFT", "CAM_FRONT", "CAM_FRONT_RIGHT")

# --- High-level navigation instructions ------------------------------------
COMMANDS = ("Go Straight", "Turn Left", "Turn Right", "Stop")


@dataclass
class TokenizerConfig:
    dt: float = DT
    horizon_s: float = HORIZON_S
    codebook_size: int = CODEBOOK_SIZE
    delta: float = KDISK_DELTA
    ego_length: float = EGO_LENGTH
    ego_width: float = EGO_WIDTH
    contour_points: int = CONTOUR_POINTS

    @property
    def n_tokens(self) -> int:
        return int(round(self.horizon_s / self.dt))


@dataclass
class DataConfig:
    root: str = "data"
    history_frames: int = HISTORY_FRAMES
    cameras: tuple = field(default_factory=lambda: CAMERAS)
    sample_rate_hz: float = 2.0

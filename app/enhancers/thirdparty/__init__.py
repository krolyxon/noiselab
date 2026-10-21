"""Wrappers around third-party denoisers: optional comparison baselines (CLAUDE.md §14).

Every module here imports its library lazily, inside process() / load_model(), so
importing this package is cheap and never fails when a library is missing.
Availability is reported by AlgoSpec.availability().
"""

from .deepfilter_ import DeepFilterNet3
from .demucs_ import Demucs
from .noisereduce_ import NoiseReduce
from .nsnet2_ import NsNet2
from .rnnoise_ import RnNoise
from .webrtc_ import WebRtcNs

ENHANCERS = (NoiseReduce(), WebRtcNs(), RnNoise(), NsNet2(), DeepFilterNet3(), Demucs())

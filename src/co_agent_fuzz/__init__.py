"""Co-AgentFuzz: collaborative, feedback-directed prompt fuzzing."""

from .config import load_config
from .agents.coordinator import CoAgentCoordinator

__all__ = ["CoAgentCoordinator", "load_config"]
__version__ = "0.1.0"

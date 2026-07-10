"""
Memory decay models — biologically-inspired forgetting functions.

Each model computes a retention factor R ∈ [0, 1] given:
    - delta_t: time since last observation
    - visit_count: number of times the cell was observed

R = 1.0 means full retention (no forgetting).
R = 0.0 means complete forgetting (reset to unknown).

The retention factor is multiplied against log-odds values,
pulling them toward 0 (maximum uncertainty / P=0.5).
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import Union


class DecayModel(ABC):
    """Abstract base class for memory decay models."""

    @abstractmethod
    def compute_retention(self, delta_t: Union[float, np.ndarray],
                          visit_count: Union[int, np.ndarray]
                          ) -> Union[float, np.ndarray]:
        """
        Compute retention factor.

        Args:
            delta_t: Time since last observation (scalar or array).
            visit_count: Number of times observed (scalar or array).

        Returns:
            Retention factor in [0, 1] (scalar or array, matching input).
        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name for this decay model."""
        pass

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class NoDecay(DecayModel):
    """
    No decay — control baseline.

    R = 1.0 always (perfect memory, standard SLAM).
    """

    def compute_retention(self, delta_t, visit_count):
        if isinstance(delta_t, np.ndarray):
            return np.ones_like(delta_t, dtype=np.float64)
        return 1.0

    @property
    def name(self) -> str:
        return "No Decay (Control)"


class ExponentialDecay(DecayModel):
    """
    Exponential decay — Ebbinghaus forgetting curve.

    R(Δt) = e^(-λ · Δt)

    Fast initial forgetting with asymptotic approach to zero.
    Single parameter λ controls decay rate.
    """

    def __init__(self, decay_lambda: float = 0.01):
        self.decay_lambda = decay_lambda

    def compute_retention(self, delta_t, visit_count):
        return np.exp(-self.decay_lambda * delta_t)

    @property
    def name(self) -> str:
        return f"Exponential (λ={self.decay_lambda})"

    def __repr__(self) -> str:
        return f"ExponentialDecay(λ={self.decay_lambda})"


class PowerLawDecay(DecayModel):
    """
    Power-law decay — Jost's Law.

    R(Δt) = (Δt + 1)^(-β)

    Slower long-term decay than exponential. Better fits human
    memory data for long retention intervals.
    """

    def __init__(self, beta: float = 0.5):
        self.beta = beta

    def compute_retention(self, delta_t, visit_count):
        return np.power(delta_t + 1.0, -self.beta)

    @property
    def name(self) -> str:
        return f"Power-Law (β={self.beta})"

    def __repr__(self) -> str:
        return f"PowerLawDecay(β={self.beta})"


class AdaptiveDecay(DecayModel):
    """
    Adaptive decay — visit-frequency-modulated forgetting.

    R(Δt) = e^(-λ/S · Δt)
    where S = min(1 + α · n_visits, S_max)

    Stability S increases with visit count (spaced repetition).
    Frequently observed landmarks resist decay.
    Most biologically plausible model.
    """

    def __init__(self, decay_lambda: float = 0.01,
                 alpha: float = 0.3,
                 stability_max: float = 10.0):
        self.decay_lambda = decay_lambda
        self.alpha = alpha
        self.stability_max = stability_max

    def compute_retention(self, delta_t, visit_count):
        stability = np.minimum(1.0 + self.alpha * visit_count, self.stability_max)
        return np.exp(-self.decay_lambda / stability * delta_t)

    @property
    def name(self) -> str:
        return f"Adaptive (λ={self.decay_lambda}, α={self.alpha})"

    def __repr__(self) -> str:
        return (f"AdaptiveDecay(λ={self.decay_lambda}, α={self.alpha}, "
                f"S_max={self.stability_max})")


class AggressiveDecay(DecayModel):
    """
    Aggressive exponential decay — very high λ.

    R(Δt) = e^(-λ · Δt) with λ >> typical values.

    Tests extreme forgetting. Expected to degrade performance.
    """

    def __init__(self, decay_lambda: float = 0.2):
        self.decay_lambda = decay_lambda

    def compute_retention(self, delta_t, visit_count):
        return np.exp(-self.decay_lambda * delta_t)

    @property
    def name(self) -> str:
        return f"Aggressive (λ={self.decay_lambda})"

    def __repr__(self) -> str:
        return f"AggressiveDecay(λ={self.decay_lambda})"


class ThresholdDecay(DecayModel):
    """
    Threshold decay — binary forgetting.

    Cells with |log_odds| below a certainty threshold after time
    has passed are fully forgotten (R=0). Otherwise retained (R=1).

    This models a "remember or forget completely" mechanism.
    """

    def __init__(self, time_threshold: float = 100.0,
                 certainty_boost_per_visit: float = 10.0):
        """
        Args:
            time_threshold: Base time before forgetting occurs.
            certainty_boost_per_visit: Extra time per visit before forgetting.
        """
        self.time_threshold = time_threshold
        self.certainty_boost_per_visit = certainty_boost_per_visit

    def compute_retention(self, delta_t, visit_count):
        effective_threshold = (self.time_threshold +
                               self.certainty_boost_per_visit * visit_count)
        if isinstance(delta_t, np.ndarray):
            return np.where(delta_t > effective_threshold, 0.0, 1.0)
        return 0.0 if delta_t > effective_threshold else 1.0

    @property
    def name(self) -> str:
        return f"Threshold (t={self.time_threshold})"

    def __repr__(self) -> str:
        return (f"ThresholdDecay(t_thresh={self.time_threshold}, "
                f"boost={self.certainty_boost_per_visit})")


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def create_decay_model(model_type: str, **kwargs) -> DecayModel:
    """
    Factory function to create a decay model by name.

    Args:
        model_type: One of 'none', 'exponential', 'power_law',
                    'adaptive', 'aggressive', 'threshold'.
        **kwargs: Parameters forwarded to the model constructor.
    """
    models = {
        "none": NoDecay,
        "exponential": ExponentialDecay,
        "power_law": PowerLawDecay,
        "adaptive": AdaptiveDecay,
        "aggressive": AggressiveDecay,
        "threshold": ThresholdDecay,
    }
    if model_type not in models:
        raise ValueError(
            f"Unknown decay model '{model_type}'. "
            f"Available: {list(models.keys())}"
        )
    return models[model_type](**kwargs)

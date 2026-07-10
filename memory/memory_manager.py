"""
Memory Manager — applies decay models to the occupancy grid on schedule.

Tracks decay statistics and identifies cells that have decayed below
the uncertainty threshold as re-exploration candidates.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from config.settings import DecayConfig, DecayModelType
from memory.decay_models import (
    DecayModel, create_decay_model,
    NoDecay, ExponentialDecay, PowerLawDecay,
    AdaptiveDecay, AggressiveDecay, ThresholdDecay
)
from slam.occupancy_grid import DecayingOccupancyGrid


class MemoryManager:
    """
    Manages memory decay application and tracking.

    Responsibilities:
        - Apply the configured decay model at regular intervals
        - Track which cells have decayed below uncertainty threshold
        - Report decay statistics (for metrics collection)
        - Identify re-exploration candidates
    """

    def __init__(self, decay_config: DecayConfig):
        """
        Args:
            decay_config: Decay configuration with model type and parameters.
        """
        self.config = decay_config
        self.decay_model = self._create_model(decay_config)

        # Statistics
        self._total_decay_applications = 0
        self._total_cells_decayed = 0
        self._cells_fully_forgotten = 0

    def _create_model(self, config: DecayConfig) -> DecayModel:
        """Create the appropriate decay model from config."""
        model_type = config.model_type

        if model_type == DecayModelType.NONE:
            return NoDecay()
        elif model_type == DecayModelType.EXPONENTIAL:
            return ExponentialDecay(decay_lambda=config.decay_lambda)
        elif model_type == DecayModelType.POWER_LAW:
            return PowerLawDecay(beta=config.power_beta)
        elif model_type == DecayModelType.ADAPTIVE:
            return AdaptiveDecay(
                decay_lambda=config.decay_lambda,
                alpha=config.adaptive_alpha,
                stability_max=config.stability_max,
            )
        elif model_type == DecayModelType.AGGRESSIVE:
            return AggressiveDecay(decay_lambda=config.aggressive_lambda)
        elif model_type == DecayModelType.THRESHOLD:
            return ThresholdDecay(
                time_threshold=config.certainty_threshold * 1000,
                certainty_boost_per_visit=10.0,
            )
        else:
            raise ValueError(f"Unknown decay model type: {model_type}")

    # ------------------------------------------------------------------
    # Decay application
    # ------------------------------------------------------------------

    def should_apply_decay(self, timestep: int) -> bool:
        """Check if decay should be applied at this timestep."""
        if isinstance(self.decay_model, NoDecay):
            return False
        return timestep > 0 and timestep % self.config.decay_interval == 0

    def apply_decay(self, grid: DecayingOccupancyGrid, current_time: float):
        """
        Apply decay to the occupancy grid.

        Uses the vectorized decay path for performance.

        Args:
            grid: The occupancy grid to decay.
            current_time: Current simulation timestamp.
        """
        if isinstance(self.decay_model, NoDecay):
            return

        # Snapshot before decay (for stats)
        abs_lo_before = np.abs(grid.log_odds)

        # Vectorized decay application
        grid.apply_decay_vectorized(
            current_time,
            self.decay_model.compute_retention
        )

        # Update statistics
        abs_lo_after = np.abs(grid.log_odds)
        self._total_decay_applications += 1

        # Count cells that moved toward uncertainty
        threshold = self.config.uncertainty_threshold
        decayed_mask = (abs_lo_before > threshold) & (abs_lo_after <= threshold)
        self._cells_fully_forgotten += int(np.sum(decayed_mask))

        changed = np.sum(np.abs(grid.log_odds - np.sign(grid.log_odds) *
                                abs_lo_before) > 1e-6)
        self._total_cells_decayed += int(changed)

    # ------------------------------------------------------------------
    # Re-exploration candidates
    # ------------------------------------------------------------------

    def get_reexploration_candidates(self, grid: DecayingOccupancyGrid
                                      ) -> np.ndarray:
        """
        Identify cells that were once observed but have decayed
        below the uncertainty threshold.

        Returns:
            Boolean mask of shape grid.shape — True for re-explore candidates.
        """
        was_observed = grid.last_observed >= 0
        is_uncertain = np.abs(grid.log_odds) < self.config.uncertainty_threshold
        return was_observed & is_uncertain

    def get_reexploration_count(self, grid: DecayingOccupancyGrid) -> int:
        """Count cells needing re-exploration."""
        return int(np.sum(self.get_reexploration_candidates(grid)))

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def get_decay_stats(self, grid: DecayingOccupancyGrid) -> Dict:
        """
        Return current decay statistics.

        Returns dict with:
            - model_name: Decay model identifier
            - total_applications: Number of times decay was applied
            - total_cells_decayed: Cumulative cells affected
            - cells_fully_forgotten: Cells that dropped below threshold
            - reexploration_candidates: Current cells needing re-exploration
            - memory_usage: Current memory usage breakdown
        """
        return {
            "model_name": self.decay_model.name,
            "total_applications": self._total_decay_applications,
            "total_cells_decayed": self._total_cells_decayed,
            "cells_fully_forgotten": self._cells_fully_forgotten,
            "reexploration_candidates": self.get_reexploration_count(grid),
            "memory_usage": grid.get_memory_usage(),
        }

    def reset_stats(self):
        """Reset cumulative statistics."""
        self._total_decay_applications = 0
        self._total_cells_decayed = 0
        self._cells_fully_forgotten = 0

    def __repr__(self) -> str:
        return (
            f"MemoryManager(model={self.decay_model}, "
            f"interval={self.config.decay_interval}, "
            f"applications={self._total_decay_applications})"
        )

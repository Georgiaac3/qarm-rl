"""
Curriculum learning scheduler for progressive task difficulty.

Gradually increases complexity:
- Phase 1: Few easy objects → grasp practice
- Phase 2: More objects, mixed difficulty
- Phase 3: Many heavy objects → challenging scenarios

improves convergence
"""

from dataclasses import dataclass


@dataclass
class CurriculumPhase:
    """Definition of a curriculum learning phase."""

    name: str
    n_objects: int  # Number of objects in bin
    object_types: list  # Allowed object types
    n_steps: int  # How many training steps to stay in this phase
    reward_bonus: float  # Bonus multiplier for rewards (v. shaping)
    difficulty_level: float  # For logging/analysis


class CurriculumScheduler:
    """
    Progressive curriculum that adapts based on performance.

    Phases:
    1. Early (0-50k steps): Few easy objects, high reward bonus
    2. Mid (50k-150k steps): More objects, mixed types
    3. Late (150k+): Max complexity, large objects, heavy items
    """

    PHASES = [
        CurriculumPhase(
            name="PHASE_1_EASY",
            n_objects=2,
            object_types=["small_plastic", "small_metal"],
            n_steps=50_000,
            reward_bonus=1.5,  # +50% reward shaping
            difficulty_level=0.2,
        ),
        CurriculumPhase(
            name="PHASE_2_MEDIUM",
            n_objects=5,
            object_types=[
                "small_plastic",
                "small_metal",
                "medium_plastic",
                "medium_metal",
            ],
            n_steps=100_000,
            reward_bonus=1.0,  # Normal rewards
            difficulty_level=0.6,
        ),
        CurriculumPhase(
            name="PHASE_3_HARD",
            n_objects=8,
            object_types=[
                "small_plastic",
                "small_metal",
                "medium_plastic",
                "medium_metal",
                "large_plastic",
                "large_metal",
                "heavy",
            ],
            n_steps=100_000,
            reward_bonus=0.8,  # Slightly reduced (harder task)
            difficulty_level=1.0,
        ),
    ]

    def __init__(self, total_steps: int):
        """
        Initialize curriculum scheduler.

        Args:
            total_steps: Total training timesteps
        """
        self.total_steps = total_steps
        self.current_step = 0
        self.phase_idx = 0
        self.current_phase = self.PHASES[0]

    def update(self, step: int) -> CurriculumPhase:
        """
        Update curriculum based on training step.

        Args:
            step: Current training step

        Returns:
            Current phase configuration
        """
        self.current_step = step

        # Find which phase we're in
        accumulated_steps = 0
        for idx, phase in enumerate(self.PHASES):
            accumulated_steps += phase.n_steps
            if step < accumulated_steps:
                self.phase_idx = idx
                self.current_phase = phase
                break
        else:
            # Stay in last phase if we've passed all phases
            self.phase_idx = len(self.PHASES) - 1
            self.current_phase = self.PHASES[-1]

        return self.current_phase

    def get_config(self) -> dict:
        """Get environment config for current phase."""
        return {
            "n_objects": self.current_phase.n_objects,
            "object_types": self.current_phase.object_types,
            "reward_bonus": self.current_phase.reward_bonus,
        }

    def get_progress(self) -> str:
        """Get human-readable progress string."""
        phase_name = self.current_phase.name
        phase_pct = (
            (self.current_step % self.current_phase.n_steps) / self.current_phase.n_steps * 100
        )
        difficulty = self.current_phase.difficulty_level
        return f"{phase_name} ({phase_pct:.1f}% | difficulty={difficulty:.1f})"


# Example usage
if __name__ == "__main__":
    scheduler = CurriculumScheduler(total_steps=250_000)

    steps = [0, 25_000, 50_000, 100_000, 150_000, 200_000, 250_000]

    print("Curriculum Learning Schedule:")
    print("=" * 70)
    for step in steps:
        phase = scheduler.update(step)
        print(f"Step {step:7d}: {scheduler.get_progress()}")
        print(f"           Objects: {phase.n_objects}, Types: {phase.object_types[:2]}...")

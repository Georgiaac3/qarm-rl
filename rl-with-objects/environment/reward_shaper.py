"""
Advanced reward shaping for bin picking task.

Provides:
1. Distance-based grasping bonus
2. Velocity-based throwing bonus
3. Cumulative object removal reward
4. Step efficiency bonus
5. Stability penalties
"""


class RewardShaper:
    """
    Advanced reward calculation with multiple components.

    Rewards:
    - Grasp quality: Based on distance to actual object
    - Throw velocity: Higher velocity = higher reward (when grasped)
    - Object removal: Bonus for clearing objects
    - Efficiency: Penalty for wasted steps
    - Stability: Penalty for collisions
    """

    # Reward thresholds and multipliers
    GRASP_BONUS = 10.0
    THROW_BONUS = 50.0
    Bin_CLEAR_BONUS = 100.0
    STEP_PENALTY = -1.0
    COLLISION_PENALTY = -5.0

    # Distance thresholds for grasp quality
    PERFECT_GRASP_DIST = 0.002  # Within 2mm = perfect grasp
    GOOD_GRASP_DIST = 0.01  # Within 1cm = good grasp

    # Velocity thresholds for throwing
    MIN_THROW_VEL = 0.5  # m/s minimum to count as throw
    GOOD_THROW_VEL = 2.0  # m/s good throw
    EXCELLENT_THROW_VEL = 5.0  # m/s excellent throw

    def __init__(self, curriculum_bonus: float = 1.0):
        """
        Initialize reward shaper.

        Args:
            curriculum_bonus: Multiplier for all rewards (from curriculum)
        """
        self.curriculum_bonus = curriculum_bonus
        self.last_n_objects = None
        self.total_reward = 0.0

    def calculate_reward(
        self,
        grasp_success: bool,
        grasp_distance: float,  # Distance to actual object
        grasped_object,  # The object (if grasped)
        thrown: bool,
        throw_distance: float,  # Distance from bin
        throw_velocity: float,  # m/s
        objects_remaining: int,
        step_count: int,
        collision: bool = False,
    ) -> dict:
        """
        Calculate multi-component reward.

        Returns:
            Dict with breakdown: {
                'total': float,
                'grasp': float,
                'throw': float,
                'removal': float,
                'step': float,
                'collision': float,
                'components': dict
            }
        """
        reward_parts = {}
        total_reward = 0.0

        # ===== GRASP COMPONENT =====
        grasp_reward = 0.0
        if grasp_success:
            # Base grasp bonus
            grasp_reward = self.GRASP_BONUS

            # Quality bonus (distance to actual object)
            if grasp_distance < self.PERFECT_GRASP_DIST:
                grasp_reward *= 1.5  # Perfect grasp bonus
            elif grasp_distance < self.GOOD_GRASP_DIST:
                grasp_reward *= 1.2  # Good grasp bonus

            # Object mass bonus (heavier objects = harder to grasp)
            if grasped_object and hasattr(grasped_object, "mass"):
                mass_bonus = 1.0 + grasped_object.mass / 0.1  # Scale 0-2x
                grasp_reward *= mass_bonus

        reward_parts["grasp"] = grasp_reward * self.curriculum_bonus
        total_reward += reward_parts["grasp"]

        # ===== THROW COMPONENT =====
        throw_reward = 0.0
        if thrown and throw_velocity >= self.MIN_THROW_VEL:
            throw_reward = self.THROW_BONUS

            # Velocity bonus (higher is better, up to EXCELLENT)
            if throw_velocity >= self.EXCELLENT_THROW_VEL:
                throw_reward *= 1.5
            elif throw_velocity >= self.GOOD_THROW_VEL:
                throw_reward *= 1.2

            # Distance bonus (far throw is good)
            if throw_distance > 0.25:  # Outside bin bounds
                throw_reward *= 1.1

        reward_parts["throw"] = throw_reward * self.curriculum_bonus
        total_reward += reward_parts["throw"]

        # ===== OBJECT REMOVAL COMPONENT =====
        removal_reward = 0.0
        if self.last_n_objects is not None:
            objects_removed = self.last_n_objects - objects_remaining
            if objects_removed > 0:
                # Reward for each object removed (cumulative progress)
                removal_reward = objects_removed * 20.0

        # Bin cleared bonus
        if objects_remaining == 0:
            removal_reward = self.Bin_CLEAR_BONUS * self.curriculum_bonus

        self.last_n_objects = objects_remaining
        reward_parts["removal"] = removal_reward * self.curriculum_bonus
        total_reward += reward_parts["removal"]

        # ===== EFFICIENCY PENALTY =====
        reward_parts["step"] = self.STEP_PENALTY * self.curriculum_bonus
        total_reward += reward_parts["step"]

        # ===== COLLISION PENALTY =====
        collision_reward = self.COLLISION_PENALTY if collision else 0.0
        reward_parts["collision"] = collision_reward * self.curriculum_bonus
        total_reward += reward_parts["collision"]

        self.total_reward += total_reward

        return {
            "total": total_reward,
            "components": reward_parts,
            "cumulative": self.total_reward,
        }

    def reset(self):
        """Reset tracking state for new episode."""
        self.last_n_objects = None
        self.total_reward = 0.0


# Example: Integrate into environment
def integrate_reward_shaper_into_env(env, reward_shaper):
    """
    Monkey-patch the environment's step() to use reward shaper.

    This is an example - you'd want to cleanly integrate this instead.
    """
    original_step = env.step

    def new_step(action):
        obs, reward, terminated, truncated, info = original_step(action)

        # Enhanced reward calculation (your existing logic here)
        # reward_dict = reward_shaper.calculate_reward(...)
        # reward = reward_dict['total']
        # info.update({'reward_breakdown': reward_dict['components']})

        return obs, reward, terminated, truncated, info

    env.step = new_step
    return env


if __name__ == "__main__":
    # Test reward shaping
    shaper = RewardShaper(curriculum_bonus=1.5)

    # Scenario 1: Perfect grasp of light object
    result = shaper.calculate_reward(
        grasp_success=True,
        grasp_distance=0.001,  # 1mm - perfect
        grasped_object=None,
        thrown=False,
        throw_distance=0.0,
        throw_velocity=0.0,
        objects_remaining=5,
        step_count=10,
    )
    print("Scenario 1 (Perfect grasp):")
    print(f"  Reward: {result['total']:.2f}")
    print(f"  Components: {result['components']}\n")

    # Scenario 2: Excellent throw
    result = shaper.calculate_reward(
        grasp_success=False,
        grasp_distance=0.0,
        grasped_object=None,
        thrown=True,
        throw_distance=0.8,
        throw_velocity=5.5,
        objects_remaining=4,
        step_count=50,
    )
    print("Scenario 2 (Excellent throw):")
    print(f"  Reward: {result['total']:.2f}")
    print(f"  Components: {result['components']}\n")

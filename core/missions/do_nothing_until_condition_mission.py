from core.missions.base_mission import Mission
from utils.types import DoNothing, Waypoint


class DoNothingUntilConditionMission(Mission):
    def __init__(self, condition_fn):
        self.condition_fn = condition_fn

    def finish_condition(self, t: float, waypoint_mes: Waypoint) -> bool:
        """La mission se termine lorsque la condition spécifiée est vérifiée."""
        return self.condition_fn(t, waypoint_mes)

    def get_waypoint_at_t(self, t):
        return DoNothing

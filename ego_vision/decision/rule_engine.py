"""Pedestrian actions; urgent/unknown evidence stops immediately."""
from dataclasses import dataclass
from ego_vision.config.settings import HYSTERESIS_N
from ego_vision.reasoning.risk import SEVERITY


@dataclass(frozen=True)
class Decision:
    action: str
    risk: str
    reason: str
    threat_id: int | None = None
    class_name: str | None = None
    motion_phrase: str = "ahead"
    speak: bool = True

    @property
    def key(self):
        return self.threat_id, self.class_name, self.action, self.risk

    def message(self):
        if self.action == "NONE" or not self.speak:
            return ""
        if self.risk == "UNKNOWN":
            return "Observations uncertain. Stop."
        name = (self.class_name or "Obstacle").capitalize()
        action = {"STOP": "Stop", "MOVE LEFT": "Move left", "MOVE RIGHT": "Move right"}[self.action]
        return f"{name} {self.motion_phrase}. {action}."


class RuleEngine:
    def __init__(self, hysteresis_n=HYSTERESIS_N):
        if hysteresis_n < 1:
            raise ValueError("Persistence must be positive")
        self.n = hysteresis_n
        self.current = Decision("NONE", "UNKNOWN", "starting", speak=False)
        self._pending = None
        self._count = 0

    def decide(self, scene):
        proposal = self._propose(scene)
        # Never preserve an obsolete side instruction during a switch or loss.
        urgent = proposal.risk in {"HIGH", "CRITICAL", "UNKNOWN"}
        if proposal.key == self._pending:
            self._count += 1
        else:
            self._pending, self._count = proposal.key, 1
        if proposal.action == "STOP" and urgent:
            self.current = proposal
        elif proposal.key == self.current.key:
            self.current = proposal
        elif self._count >= self.n:
            self.current = proposal
        elif proposal.action.startswith("MOVE"):
            self.current = Decision("STOP", proposal.risk, "confirming alternative direction",
                                    proposal.threat_id, proposal.class_name, proposal.motion_phrase)
        elif proposal.action == "NONE":
            # De-escalation may persist visually; obsolete threat is not narrated.
            if self.current.action != "NONE":
                self.current = Decision("STOP", self.current.risk, "confirming threat cleared", speak=False)
        else:
            self.current = Decision("NONE", proposal.risk, "confirming obstacle", speak=False)
        return self.current

    def _propose(self, scene):
        t = scene.threat
        if scene.risk == "LOW":
            return Decision("NONE", "LOW", scene.reason, speak=False)
        tid = t.detection.track_id if t else None
        name = t.detection.class_name if t else None
        phrase = "ahead"
        if t and t.motion.ready and t.motion.stable:
            if t.predicted_overlap and not t.current_overlap and t.motion.direction:
                phrase = "crossing from the right" if t.motion.direction == "left" else "crossing from the left"
            elif t.detection.ttc_valid:
                phrase = "approaching"
        action = "STOP"
        if scene.healthy and scene.risk not in {"CRITICAL", "UNKNOWN"} and t and tid is not None:
            if t.motion.ready and t.motion.stable:
                choices = [side for side, evidence in scene.sides.items() if evidence.usable]
                if choices:
                    previous = self.current.action.removeprefix("MOVE ")
                    # Stick to a still-supported side; changing sides requires persistence.
                    if previous in choices:
                        side = previous
                    else:
                        side = max(choices, key=lambda s: scene.sides[s].clearance or 0)
                    action = "MOVE " + side
        return Decision(action, scene.risk, scene.reason, tid, name, phrase)

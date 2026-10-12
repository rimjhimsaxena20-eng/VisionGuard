"""One speech worker, one replaceable pending event and evidence-bound expiry."""
from collections import deque
from dataclasses import dataclass
import sys
import threading
import time
from ego_vision.reasoning.risk import SEVERITY


@dataclass(frozen=True)
class WarningEvent:
    key: tuple
    text: str
    priority: int
    created: float
    expires: float


class WarningPolicy:
    """Pure state machine; caller holds the worker lock."""
    def __init__(self, cooldown=5.0):
        self.cooldown = cooldown
        self.current_key = None
        self.deadline = 0.0
        self.pending = None
        self.last_spoken = {}
        self.last_priority = -1
        self.last_started = float('-inf')
        self.active_key = None

    def offer(self, decision, now, ttl=1.0):
        key = decision.key if decision.message() else None
        if key != self.current_key:
            self.pending = None
        self.current_key, self.deadline = key, now + max(0., ttl)
        self.last_spoken = {k: t for k, t in self.last_spoken.items() if now - t < max(60., self.cooldown * 2)}
        if key is None or ttl <= 0 or self.active_key == key:
            return None
        if self.pending and self.pending.key == key:
            self.pending = WarningEvent(key, self.pending.text, self.pending.priority, self.pending.created, self.deadline)
            return None
        priority = SEVERITY[decision.risk]
        escalation = priority > self.last_priority
        if now - self.last_spoken.get(key, float('-inf')) < self.cooldown and not escalation:
            return None
        if now - self.last_started < 1.0 and not escalation and decision.action != "STOP":
            return None
        event = WarningEvent(key, decision.message(), priority, now, self.deadline)
        self.pending = event
        return event

    def take(self, now):
        event, self.pending = self.pending, None
        if event and event.key == self.current_key and now < min(event.expires, self.deadline):
            return event
        return None

    def started(self, event, now):
        self.last_spoken[event.key] = now
        self.last_priority, self.last_started = event.priority, now
        self.active_key = event.key

    def valid(self, event, now):
        return event.key == self.current_key and now < self.deadline

    def clear(self):
        self.current_key = self.pending = None
        self.deadline = 0.0


class WarningSpeaker:
    def __init__(self, enabled=True, cooldown=5.0, source_failed=lambda: False):
        self.policy = WarningPolicy(cooldown)
        self.status = "starting" if enabled else "muted"
        self.error = None
        self._source_failed = source_failed
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._events = deque(maxlen=128)
        self._thread = None
        if enabled:
            self._thread = threading.Thread(target=self._work, name="local-speech", daemon=True)
            self._thread.start()

    def publish(self, decision, ttl=1.0):
        if self.status in {"muted", "unavailable", "closed"}:
            return None
        with self._lock:
            return self.policy.offer(decision, time.monotonic(), ttl)

    def clear(self):
        with self._lock:
            self.policy.clear()

    def drain_events(self):
        with self._lock:
            events = list(self._events)
            self._events.clear()
        return events

    def _work(self):
        engine = None
        com = None
        loop_started = False
        active = None
        submitted = {}

        def on_started(name):
            with self._lock:
                event = submitted.pop(name, None)
                if event:
                    self._events.append(("speech_started", time.monotonic(), event))
        try:
            if sys.platform == "win32":
                import comtypes as com
                com.CoInitialize()
            import pyttsx3
            engine = pyttsx3.init(driverName="sapi5" if sys.platform == "win32" else None)
            engine.connect("started-utterance", on_started)
            engine.setProperty("rate", 185)
            engine.startLoop(False)
            loop_started = True
            self.status = "ready"
            while not self._stop.is_set():
                now = time.monotonic()
                with self._lock:
                    if self._source_failed():
                        self.policy.clear()
                    if active and not self.policy.valid(active, now):
                        engine.stop()
                        submitted.clear()
                        self._events.append(("speech_cancelled", now, active))
                        self.policy.active_key = None
                        active = None
                    if active and not engine.isBusy():
                        self.policy.active_key = None
                        active = None
                    if active is None:
                        event = self.policy.take(now)
                        if event:
                            name = str(now)
                            submitted[name] = event
                            engine.say(event.text, name=name)
                            self.policy.started(event, now)
                            self._events.append(("speech_submitted", now, event))
                            active = event
                engine.iterate()  # Pumps SAPI COM events in the owning thread.
                self._stop.wait(.02)
        except Exception as exc:
            self.status, self.error = "unavailable", str(exc)
            print(f"Audio unavailable; visual warnings continue: {exc}", file=sys.stderr)
        finally:
            if engine is not None:
                try:
                    engine.stop()
                    if loop_started:
                        engine.endLoop()
                except Exception:
                    pass
            # Drop engine/COM references in their owning thread before uninitializing.
            engine = None
            if com:
                com.CoUninitialize()

    def close(self):
        self.clear()
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2.0)
        if self.status != "unavailable":
            self.status = "closed"

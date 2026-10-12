"""Integration of real reasoning stages plus a controlled speech driver double.
These tests do not simulate or benchmark model inference or native Windows audio.
"""
import contextlib
import importlib
import io
import sys
import threading
import time
import types
import unittest
from unittest.mock import patch
import numpy as np
from ego_vision.perception.detector import Detection
from ego_vision.reasoning.distance import attach_depth
from ego_vision.reasoning.ttc import TtcEstimator
from ego_vision.reasoning.motion import MotionEstimator
from ego_vision.reasoning.ego_zone import Corridor
from ego_vision.reasoning.scene_state import assemble
from ego_vision.decision.rule_engine import RuleEngine, Decision
from ego_vision.audio.warnings import WarningSpeaker


class IntegrationTests(unittest.TestCase):
    def test_measurement_to_emergency_action_and_recession(self):
        ttc, motion, engine = TtcEstimator(), MotionEstimator(), RuleEngine(3)
        for t, distance in [(0., 3.), (.2, 2.6), (.4, 2.2)]:
            d = Detection('person', 'vru', (285, 120, 355, 430), .9, 1)
            depth = np.full((480, 640), 6.)
            depth[120:430, 285:355] = distance
            attach_depth([d], depth, t)
            ttc.update([d], t)
            motions = motion.update([d], t, (480, 640, 3))
            scene = assemble([d], (480, 640, 3), Corridor(), motions, depth)
            decision = engine.decide(scene)
        self.assertEqual(decision.action, 'STOP')
        self.assertEqual(decision.risk, 'CRITICAL')
        self.assertIn('approaching', decision.message())
        self.assertAlmostEqual(d.ttc_sec, 1.1)
        depth[120:430, 285:355] = 2.5
        attach_depth([d], depth, .6)
        ttc.update([d], .6)
        self.assertFalse(d.ttc_valid)
        self.assertIsNone(d.ttc_sec)

    def test_shifted_box_does_not_sample_cached_map(self):
        ttc = TtcEstimator()
        depth = np.full((480, 640), 9.)
        depth[120:430, 285:355] = 2.
        initial = Detection('person','vru',(285,120,355,430),.9,1)
        attach_depth([initial], depth, 0.)
        ttc.update([initial], 0.)
        shifted = Detection('person','vru',(400,120,470,430),.9,1)
        ttc.update([shifted], .2)
        self.assertEqual(shifted.distance_m, 2.)
        self.assertEqual(len(ttc._tracks[1].samples), 1)
        new_track = Detection('person','vru',(400,120,470,430),.9,2)
        ttc.update([new_track], .2)
        self.assertIsNone(new_track.distance_m)

    def test_unstable_image_motion_invalidates_ttc(self):
        ttc = TtcEstimator()
        d = Detection('person','vru',(285,120,355,430),.9,1)
        for t, value in [(0.,5.),(.2,4.6),(.4,4.2)]:
            attach_depth([d], np.full((480,640),value),t)
            ttc.update([d],t)
        self.assertTrue(d.ttc_valid)
        ttc.invalidate_motion(d)
        self.assertFalse(d.ttc_valid)
        self.assertNotIn(1,ttc._tracks)

    def test_core_modules_import_without_optional_model_libraries(self):
        for name in ['config.settings','ingest.video_reader','perception.detector',
                     'perception.depth_estimator','reasoning.distance','reasoning.ttc',
                     'reasoning.motion','reasoning.risk','reasoning.scene_state',
                     'decision.rule_engine','audio.warnings','pipeline']:
            importlib.import_module('ego_vision.'+name)


class FakeEngine:
    def __init__(self):
        self.started = threading.Event()
        self.stopped = threading.Event()
        self.busy = False
        self.owner = threading.get_ident()
        self.callback = None

    def connect(self, event, callback):
        self.callback = callback

    def setProperty(self, *args):
        pass

    def startLoop(self, value):
        pass

    def say(self, message, name):
        assert threading.get_ident() == self.owner
        self.busy = True
        self.callback(name)
        self.started.set()

    def isBusy(self):
        return self.busy

    def iterate(self):
        pass

    def stop(self):
        assert threading.get_ident() == self.owner
        self.busy = False
        self.stopped.set()

    def endLoop(self):
        pass


class SpeechWorkerTests(unittest.TestCase):
    def test_native_driver_failure_falls_back(self):
        with patch.dict(sys.modules, {'pyttsx3': None}), patch('sys.platform','linux'):
            with contextlib.redirect_stderr(io.StringIO()):
                speaker = WarningSpeaker()
                speaker._thread.join(timeout=1.)
                self.assertEqual(speaker.status, 'unavailable')
                speaker.close()

    def test_worker_owns_engine_and_cancels_on_source_failure(self):
        engines = []
        ready = threading.Event()
        failed = threading.Event()

        def init(**kwargs):
            engine = FakeEngine()
            engines.append(engine)
            ready.set()
            return engine

        with patch.dict(sys.modules, {'pyttsx3': types.SimpleNamespace(init=init)}), patch('sys.platform','linux'):
            speaker = WarningSpeaker(source_failed=failed.is_set)
            try:
                self.assertTrue(ready.wait(1.))
                speaker.publish(Decision('MOVE LEFT','HIGH','test',1,'person'), ttl=1.)
                self.assertTrue(engines[0].started.wait(1.))
                failed.set()
                self.assertTrue(engines[0].stopped.wait(1.))
                events = speaker.drain_events()
                self.assertIn('speech_started', [e[0] for e in events])
                self.assertIn('speech_cancelled', [e[0] for e in events])
            finally:
                speaker.close()
            self.assertFalse(speaker._thread.is_alive())

    def test_worker_cancels_without_new_inference_frame(self):
        engines = []
        ready = threading.Event()

        def init(**kwargs):
            engines.append(FakeEngine())
            ready.set()
            return engines[0]

        with patch.dict(sys.modules, {'pyttsx3': types.SimpleNamespace(init=init)}), patch('sys.platform','linux'):
            speaker = WarningSpeaker()
            try:
                self.assertTrue(ready.wait(1.))
                speaker.publish(Decision('MOVE LEFT','HIGH','test',1,'person'), ttl=.15)
                self.assertTrue(engines[0].started.wait(1.))
                self.assertTrue(engines[0].stopped.wait(1.))
            finally:
                speaker.close()


if __name__ == '__main__':
    unittest.main()

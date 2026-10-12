"""Synthetic regression tests: genuine reasoning code, no mocked model benchmarks."""
import contextlib
import csv
import io
import math
from pathlib import Path
import tempfile
import unittest
import numpy as np
from ego_vision.perception.detector import Detection
from ego_vision.reasoning.distance import attach_depth, sample_depth
from ego_vision.reasoning.ttc import TtcEstimator
from ego_vision.reasoning.ego_zone import Corridor
from ego_vision.reasoning.motion import MotionEstimator, Motion
from ego_vision.reasoning.risk import assess, SideEvidence, Assessment, Threat
from ego_vision.decision.rule_engine import RuleEngine, Decision
from ego_vision.audio.warnings import WarningPolicy
from ego_vision.ingest.video_reader import parse_source, FileClock, FramePacket
from ego_vision.pipeline import parse_args, EventLog

SHAPE = (480, 640, 3)
CLEAR = np.full(SHAPE[:2], 6., dtype=np.float32)


def det(distance=None, t=0., box=(285, 120, 355, 430), tid=1, name="person"):
    return Detection(name, "vru", box, .9, tid, distance_m=distance,
                     distance_timestamp=t, depth_fresh=distance is not None)


def distance_series(values, times=None, estimator=None):
    estimator = estimator or TtcEstimator()
    times = times if times is not None else [i * .2 for i in range(len(values))]
    for t, value in zip(times, values):
        d = det(value, t)
        estimator.update([d], t)
    return estimator, d


def scene_for(items, depth=CLEAR, motions=None, healthy=True):
    return assess(items, motions or {}, SHAPE, Corridor(), depth, healthy)


class TtcTests(unittest.TestCase):
    def test_approaching(self):
        _, d = distance_series([5, 4.6, 4.2, 3.8])
        self.assertTrue(d.ttc_valid)
        self.assertAlmostEqual(d.closing_mps, 2.)
        self.assertAlmostEqual(d.ttc_sec, 1.9)

    def test_receding_clears_reused_detection(self):
        estimator, d = distance_series([5, 4.6, 4.2])
        self.assertIsNotNone(d.ttc_sec)
        d.distance_m, d.distance_timestamp = 4.5, .6
        estimator.update([d], .6)
        self.assertIsNone(d.ttc_sec)
        self.assertFalse(d.ttc_valid)

    def test_stationary_noise(self):
        _, d = distance_series([3, 3.02, 2.98, 3.01, 3.0, 2.99])
        self.assertIsNone(d.ttc_sec)

    def test_cached_depth_does_not_create_observation(self):
        estimator, d = distance_series([5, 4.6, 4.2])
        count = len(estimator._tracks[1].samples)
        # Brand-new current box must inherit stored distance, without sampling old pixels.
        current = det(box=(20, 20, 50, 80))
        estimator.update([current], .5)
        self.assertEqual(len(estimator._tracks[1].samples), count)
        self.assertAlmostEqual(current.distance_m, 4.2)
        self.assertAlmostEqual(current.distance_age, .1)

    def test_irregular_elapsed_measurement_time(self):
        _, d = distance_series([6, 5.8, 4.6, 4.0], [0, .1, .7, 1.])
        self.assertAlmostEqual(d.closing_mps, 2.)
        self.assertAlmostEqual(d.ttc_sec, 2.)

    def test_duplicate_timestamp_cannot_mutate_history(self):
        estimator, d = distance_series([5, 4.6, 4.2])
        count = len(estimator._tracks[1].samples)
        duplicate = det(1., .4)
        estimator.update([duplicate], .4)
        self.assertEqual(len(estimator._tracks[1].samples), count)
        self.assertEqual(duplicate.distance_m, 4.2)

    def test_invalid_depth_clears_old_ttc(self):
        for bad in [float('nan'), float('inf'), -1, 0, 500]:
            estimator, _ = distance_series([5, 4.6, 4.2])
            d = det(bad, .6)
            estimator.update([d], .6)
            self.assertIsNone(d.ttc_sec)
            self.assertIsNone(d.distance_m)
            self.assertNotIn(1, estimator._tracks)

    def test_expiry_staleness_long_gap_and_jump(self):
        estimator, _ = distance_series([5, 4.6, 4.2])
        d = det()
        estimator.update([d], 1.6)
        self.assertIsNone(d.ttc_sec)
        self.assertIsNone(d.distance_m)
        estimator.update([], 3.)
        self.assertEqual(estimator._tracks, {})
        estimator, _ = distance_series([5, 4.6, 4.2])
        d = det(3, 2.)
        estimator.update([d], 2.)
        self.assertIsNone(d.ttc_sec)
        d = det(20, 2.01)
        estimator.update([d], 2.01)
        self.assertIsNone(d.ttc_sec)

    def test_clock_reversal_resets_history(self):
        estimator, _ = distance_series([5, 4.6, 4.2], [10, 10.2, 10.4])
        d = det(4., 0.)
        estimator.update([d], 0.)
        self.assertIsNone(d.ttc_sec)
        self.assertEqual(len(estimator._tracks[1].samples), 1)

    def test_new_untracked_object_cannot_inherit_ttc(self):
        estimator, _ = distance_series([5, 4.6, 4.2])
        d = det(1., .6, tid=None)
        estimator.update([d], .6)
        self.assertIsNone(d.ttc_sec)
        self.assertEqual(d.distance_m, 1.)


class DepthTests(unittest.TestCase):
    def test_inset_avoids_background_at_bottom(self):
        depth = np.full((100, 100), 40.)
        depth[20:79, 20:80] = 2.
        sample = sample_depth(depth, (20, 20, 80, 80))
        self.assertEqual(sample.distance, 2.)

    def test_invalid_pixels_and_clipped_boxes(self):
        bad = np.full((50, 50), np.nan)
        self.assertIsNone(sample_depth(bad, (0, 0, 40, 40)).distance)
        bad[:25] = -1
        self.assertIsNone(sample_depth(bad, (0, 0, 40, 40)).distance)
        self.assertIsNone(sample_depth(CLEAR, (700, 0, 800, 90)).distance)
        self.assertIsNone(sample_depth(CLEAR, (40, 30, 20, 70)).distance)
        self.assertEqual(sample_depth(CLEAR, (-50, 0, 60, 90), .5).distance, 3.)

    def test_matched_depth_timestamp_and_uncertainty(self):
        d = det()
        attach_depth([d], CLEAR, 2.5)
        self.assertTrue(d.depth_fresh)
        self.assertEqual(d.distance_timestamp, 2.5)
        self.assertEqual(d.distance_m, 6.)
        self.assertEqual(d.depth_uncertainty, 0.)


class GeometryRiskTests(unittest.TestCase):
    def test_wide_footprint_overlap(self):
        # Center is outside the corridor; width intersects it.
        d = det(2., box=(80, 100, 265, 440), name="bench")
        scene = scene_for([d])
        self.assertTrue(scene.threat.current_overlap)
        self.assertEqual(scene.risk, 'HIGH')

    def test_crossing_before_corridor_entry(self):
        estimator = MotionEstimator()
        for i, x in enumerate([520, 505, 490, 475]):
            d = det(4., i*.2, box=(x, 150, x+45, 425), name="bicycle")
            motions = estimator.update([d], i*.2, SHAPE)
        scene = scene_for([d], motions=motions)
        self.assertFalse(scene.threat.current_overlap)
        self.assertTrue(scene.threat.predicted_overlap)
        self.assertEqual(motions[1].direction, 'left')
        self.assertIn('crossing from the right', RuleEngine(1).decide(scene).message())

    def test_beside_path_suppressed(self):
        d = det(2., box=(10, 80, 75, 440))
        scene = scene_for([d])
        self.assertEqual(scene.risk, 'LOW')
        self.assertEqual(RuleEngine().decide(scene).action, 'NONE')

    def test_stationary_chair_in_path(self):
        d = det(1.8, name='chair')
        scene = scene_for([d])
        self.assertEqual(scene.risk, 'HIGH')
        self.assertEqual(RuleEngine().decide(scene).action, 'STOP')
        self.assertNotIn('approaching', RuleEngine().decide(scene).message())

    def test_upper_image_is_not_ground_obstacle(self):
        self.assertEqual(scene_for([det(3., box=(280, 0, 360, 150))]).risk, 'LOW')

    def test_immediate_proximity_beside_path(self):
        scene = scene_for([det(.5, box=(10, 80, 75, 440))])
        self.assertEqual(scene.risk, 'CRITICAL')
        self.assertEqual(RuleEngine().decide(scene).action, 'STOP')

    def test_resolution_invariance(self):
        a = det(2., box=(80, 100, 265, 440))
        b = det(2., box=tuple(v*2 for v in a.box_xyxy))
        self.assertEqual(scene_for([a]).risk, assess([b], {}, (960, 1280, 3), Corridor(), np.full((960,1280), 6.)).risk)

    def test_left_blocked_prevents_left(self):
        main = det(2.)
        left = det(2., tid=2, box=(20, 80, 190, 440), name='chair')
        m = Motion((.445, .555, .9), ready=True, observations=5)
        scene = scene_for([main, left], motions={1:m})
        self.assertFalse(scene.sides['LEFT'].usable)
        self.assertTrue(scene.sides['RIGHT'].usable)
        self.assertEqual(RuleEngine(1).decide(scene).action, 'MOVE RIGHT')

    def test_both_sides_blocked(self):
        items = [det(2.), det(2., tid=2, box=(20, 80, 190, 440)),
                 det(2., tid=3, box=(450, 80, 620, 440))]
        scene = scene_for(items, motions={1:Motion((.445,.555,.9), ready=True)})
        self.assertEqual(RuleEngine(1).decide(scene).action, 'STOP')

    def test_no_depth_is_unknown_stop(self):
        scene = scene_for([det(2.)], depth=None)
        self.assertEqual(scene.risk, 'UNKNOWN')
        self.assertEqual(RuleEngine(1).decide(scene).action, 'STOP')

    def test_undetected_but_near_side_depth_blocks(self):
        depth = CLEAR.copy()
        depth[278:442, 38:224] = 1.
        scene = scene_for([det(2.)], depth=depth)
        self.assertFalse(scene.sides['LEFT'].usable)
        self.assertTrue(scene.sides['RIGHT'].usable)

    def test_unstable_motion_forbids_direction(self):
        d = det(2.)
        m = Motion((.445,.555,.9), ready=True, stable=False)
        self.assertEqual(RuleEngine(1).decide(scene_for([d], motions={1:m})).action, 'STOP')


class DecisionTests(unittest.TestCase):
    def test_emergency_bypasses_persistence(self):
        engine = RuleEngine(5)
        engine.current = Decision('MOVE LEFT', 'CAUTION', 'test', 1, 'person')
        result = engine.decide(scene_for([det(.5)]))
        self.assertEqual((result.risk, result.action), ('CRITICAL', 'STOP'))

    def test_lost_observations_cancel_movement_immediately(self):
        engine = RuleEngine(5)
        engine.current = Decision('MOVE LEFT', 'HIGH', 'test', 1, 'person')
        result = engine.decide(scene_for([], healthy=False))
        self.assertEqual((result.risk, result.action), ('UNKNOWN', 'STOP'))

    def test_side_switch_waits_at_stop(self):
        engine = RuleEngine(3)
        engine.current = Decision('MOVE LEFT', 'HIGH', 'test', 1, 'person')
        m = Motion((.445,.555,.9), ready=True)
        scene = scene_for([det(2.)], motions={1:m})
        scene.sides['LEFT'] = SideEvidence(reason='blocked')
        self.assertEqual(engine.decide(scene).action, 'STOP')
        self.assertEqual(engine.decide(scene).action, 'STOP')
        self.assertEqual(engine.decide(scene).action, 'MOVE RIGHT')

    def test_deescalation_is_silent_and_persistent(self):
        engine = RuleEngine(3)
        engine.current = Decision('MOVE LEFT', 'HIGH', 'test', 1, 'person')
        scene = scene_for([])
        first = engine.decide(scene)
        self.assertEqual(first.action, 'STOP')
        self.assertEqual(first.message(), '')
        engine.decide(scene)
        self.assertEqual(engine.decide(scene).action, 'NONE')


class SpeechTests(unittest.TestCase):
    def test_cooldown_and_escalation(self):
        policy = WarningPolicy(5)
        d = Decision('STOP', 'CAUTION', 'test', 1, 'person')
        event = policy.offer(d, 0)
        self.assertIsNotNone(event)
        policy.started(policy.take(.1), .1)
        policy.active_key = None
        self.assertIsNone(policy.offer(d, 1))
        self.assertIsNotNone(policy.offer(Decision('STOP','CRITICAL','test',1,'person'), 1.1))

    def test_obsolete_speech_discarded(self):
        policy = WarningPolicy()
        policy.offer(Decision('MOVE LEFT','HIGH','test',1,'person'), 0)
        policy.offer(Decision('NONE','LOW','clear',speak=False), .1)
        self.assertIsNone(policy.take(.2))

    def test_expired_speech_discarded(self):
        policy = WarningPolicy()
        policy.offer(Decision('MOVE LEFT','HIGH','test',1,'person'), 0, .3)
        self.assertIsNone(policy.take(.31))

    def test_urgent_event_replaces_pending_and_active(self):
        policy = WarningPolicy()
        old = policy.offer(Decision('MOVE LEFT','CAUTION','test',1,'person'), 0)
        policy.started(policy.take(.1), .1)
        policy.offer(Decision('STOP','CRITICAL','test',2,'bicycle'), .2)
        self.assertFalse(policy.valid(old, .2))
        self.assertEqual(policy.take(.2).key[2], 'STOP')

    def test_active_warning_expires_without_inference_heartbeat(self):
        policy = WarningPolicy()
        event = policy.offer(Decision('MOVE LEFT','HIGH','test',1,'person'), 0, .5)
        self.assertFalse(policy.valid(event, .6))


class InputLogTests(unittest.TestCase):
    def test_numeric_camera_and_file_paths(self):
        self.assertEqual(parse_source('0'), 0)
        self.assertEqual(parse_source(' 2 '), 2)
        self.assertEqual(parse_source('clips/demo.mp4'), Path('clips/demo.mp4'))
        self.assertEqual(parse_source('./0'), Path('0'))
        with self.assertRaises(ValueError):
            parse_source('-1')

    def test_pts_then_fps_fallback_stays_monotonic(self):
        clock = FileClock(25.)
        self.assertEqual(clock.timestamp(0, 10.), 10.)
        self.assertEqual(clock.timestamp(1, 10.04), 10.04)
        self.assertAlmostEqual(clock.timestamp(2, 0.), 10.08)
        self.assertAlmostEqual(clock.timestamp(3, 1000.), 10.12)

    def test_invalid_cli_values_rejected(self):
        for args in [['--depth-every','0'], ['--resize-width','-1'], ['--source-fps','nan'],
                     ['--every','3'], ['--depth-scale','-1'], ['--camera-width','0'],
                     ['--source','x.mp4','--output','x.mp4']]:
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                parse_args(args)

    def test_csv_reports_unknown_as_empty(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'events.csv'
            log = EventLog(path)
            scene = scene_for([det()], depth=None)
            log.observations(FramePacket(0,None,0.,1.), scene, RuleEngine().decide(scene))
            log.close()
            with path.open() as f:
                row = next(csv.DictReader(f))
            self.assertEqual(row['distance_m'], '')
            self.assertEqual(row['ttc_s'], '')
            self.assertEqual(row['risk'], 'UNKNOWN')


if __name__ == '__main__':
    unittest.main()

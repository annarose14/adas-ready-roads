"""Segment building: distances, speeds (walking filter), splitting and verdict thresholds."""
import segments as s


def frame(i, status="painted", seq="A", step_m=10.0, dt=2.0):
    return {"image_id": f"{seq}{i}", "sequence": seq, "status": status,
            "lanes_painted": "1" if status == "painted" else "0", "area": "test",
            "_lat": -33.9 + i * step_m / 111_000, "_lon": 151.2, "_t": 1000 + i * dt}


def test_one_degree_of_latitude_is_about_111_km():
    assert 110_000 < s.dist_m(-33.0, 151.0, -34.0, 151.0) < 112_500


def test_capture_time_in_ms_or_s():
    assert s.to_seconds("1700000000000") == 1.7e9
    assert s.to_seconds("1700000000") == 1.7e9
    assert s.to_seconds("") == 0.0


def test_speed_of_a_driving_sequence():
    speed = s.sequence_speeds([frame(i) for i in range(5)])["A"]
    assert 17 < speed < 19          # 10 m every 2 s = 18 km/h


def test_walking_sequence_is_slower_than_the_vehicle_threshold():
    speed = s.sequence_speeds([frame(i, step_m=2.0) for i in range(5)])["A"]
    assert speed < s.MIN_VEHICLE_KMH


def test_a_gap_starts_a_new_segment():
    frames = [frame(i) for i in range(5)] + [frame(100)]
    assert len(s.build_segments(frames)) == 2


def test_verdict_thresholds():
    painted = [frame(i) for i in range(5)]
    none = [frame(i, "no_lanes") for i in range(5)]
    mixed = [frame(i) for i in range(2)] + [frame(i, "no_lanes") for i in range(2, 5)]
    assert s.summarise("x", "A", painted)["verdict"] == "adas_readable"
    assert s.summarise("x", "A", mixed)["verdict"] == "at_risk"
    assert s.summarise("x", "A", none)["verdict"] == "not_readable"
    assert s.summarise("x", "A", [frame(0)])["verdict"] == "insufficient_evidence"

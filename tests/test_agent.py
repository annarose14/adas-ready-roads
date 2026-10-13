"""The agent's decisions, with the VLM, Mapillary and image files replaced by test doubles."""
import agent

MAJOR = {"highway": "primary", "name": "Main Rd", "expected_marked": True,
         "lane_markings": "", "junction": False, "source": "osm"}
MINOR = {"highway": "residential", "name": "Side St", "expected_marked": False,
         "lane_markings": "", "junction": False, "source": "osm"}


def frames(n, status="painted", start=0):
    return [{"image_id": f"A{i}", "sequence": "A", "status": status, "lanes_painted": "1",
             "area": "test", "lat": str(-33.9 + i * 1e-4), "lon": "151.2",
             "_lat": -33.9 + i * 1e-4, "_lon": 151.2, "_t": float(i),
             "captured_at": str(i * 1000), "compass_angle": "0", "drivable_frac": "0.5"}
            for i in range(start, start + n)]


def setup(monkeypatch, answers, fetched=()):
    asked, fetches = [], []
    it = iter(answers)

    def ask(path, image_id):
        asked.append(image_id)
        return next(it)

    def fetch(lat, lon, **kw):
        fetches.append(kw)
        return [dict(f) for f in fetched], "ok (test)"

    monkeypatch.setattr(agent, "ask_lines_visible", ask)
    monkeypatch.setattr(agent, "find_image", lambda image_id: "photo.jpg")
    monkeypatch.setattr(agent, "fetch_frames", fetch)
    return asked, fetches


def run(fr, ctx, budget=None):
    budget = budget if budget is not None else {"left": 60}
    s, trace, _ = agent.run_location("L1", "A", {"A"}, fr, set(), budget, ctx, [[151.2, -33.9]], 1)
    return s, trace, budget


def test_stops_after_two_matching_answers(monkeypatch):
    asked, _ = setup(monkeypatch, ["YES"] * 6)
    s, _, _ = run(frames(5), MAJOR)
    assert len(asked) == 2
    assert s["final_verdict"] == "adas_readable"


def test_split_answers_go_to_a_person(monkeypatch):
    asked, _ = setup(monkeypatch, ["YES", "NO"] * 3)
    s, _, _ = run(frames(6), MAJOR)
    assert len(asked) == 6
    assert s["final_verdict"] == "human_review"


def test_no_lines_on_a_major_marked_road_is_high_priority(monkeypatch):
    setup(monkeypatch, ["NO"] * 6)
    s, _, _ = run(frames(5), MAJOR)
    assert s["final_verdict"] == "not_readable"
    assert s["priority"] == "high"


def test_no_lines_on_a_minor_road_is_unmarked_by_design(monkeypatch):
    setup(monkeypatch, ["NO"] * 6)
    s, _, _ = run(frames(5), MINOR)
    assert s["final_verdict"] == "unmarked_by_design"


def test_lines_where_osm_says_no_markings_is_a_map_conflict(monkeypatch):
    setup(monkeypatch, ["YES"] * 6)
    s, _, _ = run(frames(5), dict(MINOR, lane_markings="no"))
    assert s["final_verdict"] == "human_review"
    assert s["reason"].startswith("map conflict")


def test_thin_evidence_triggers_a_fetch(monkeypatch):
    _, fetches = setup(monkeypatch, ["YES"] * 6, fetched=frames(3, start=10))
    s, trace, budget = run(frames(2), MAJOR)
    assert len(fetches) == 1
    assert budget["left"] == 59
    assert "act_fetch" in [t["step"] for t in trace]
    assert s["n_road_frames"] == 5


def test_enough_evidence_means_no_fetch(monkeypatch):
    _, fetches = setup(monkeypatch, ["YES"] * 6)
    run(frames(5), MAJOR)
    assert fetches == []


def test_photos_opencv_rejected_are_never_sent_to_the_model(monkeypatch):
    road = frames(3)
    junk = [dict(f, image_id=f"J{i}", status="not_road") for i, f in enumerate(frames(3, start=20))]
    asked, _ = setup(monkeypatch, ["YES", "NO"] * 3)
    run(road + junk, MAJOR)
    assert asked and all(i.startswith("A") for i in asked)


def test_every_step_is_logged(monkeypatch):
    setup(monkeypatch, ["YES"] * 6)
    _, trace, _ = run(frames(5), MAJOR)
    steps = [t["step"] for t in trace]
    for step in ("perceive", "road_context", "ask_vlm", "decide"):
        assert step in steps


def test_heading_average_wraps_around_north():
    h = agent.mean_heading([{"compass_angle": "350"}, {"compass_angle": "10"}])
    assert min(h, 360 - h) < 1

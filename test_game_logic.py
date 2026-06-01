from game_parser import FutureBoardEvaluator, MovePathfinder


def test_diagonal_paths_are_valid():
    board = [
        ["muffin", "EMPTY", "EMPTY"],
        ["EMPTY", "muffin", "EMPTY"],
        ["EMPTY", "EMPTY", "muffin"],
    ]

    paths = MovePathfinder().find_paths(board)

    assert ((0, 0), (1, 1), (2, 2)) in paths or ((2, 2), (1, 1), (0, 0)) in paths


def test_completed_target_has_zero_item_score():
    board = [["biscuit", "biscuit", "biscuit"]]
    path = ((0, 0), (0, 1), (0, 2))

    score, reasons = MovePathfinder().score_path(board, {"biscuit": "27 / 27"}, "biscuit", path)

    assert score == 0.0
    assert "target biscuit completed: item score is zero" in reasons


def test_move_budget_rewards_on_pace_target_collection():
    board = [["muffin", "muffin", "muffin"], ["biscuit", "biscuit", "biscuit"]]
    muffin_path = ((0, 0), (0, 1), (0, 2))
    biscuit_path = ((1, 0), (1, 1), (1, 2))
    pathfinder = MovePathfinder()

    scored = pathfinder.score_paths(
        board,
        {"muffin": "0 / 9"},
        [biscuit_path, muffin_path],
        moves_left=2,
    )

    assert scored[0].path == muffin_path
    assert any(reason.startswith("move budget muffin") for reason in scored[0].reasons)


def test_analyze_state_parses_moves_left_for_solver_pressure():
    game_state = {
        "gameState": {"movesLeft": "2", "targets": {"muffin": "0 / 9"}},
        "board": [["muffin", "muffin", "muffin"], ["biscuit", "biscuit", "biscuit"]],
    }

    result = __import__("game_parser").analyze_state(game_state, top=2)

    assert result["analysis"]["movesLeftParsed"] == 2
    assert result["bestMove"]["item"] == "muffin"


def test_captured_frame_without_board_returns_none_instead_of_raising(tmp_path):
    import numpy as np
    from types import SimpleNamespace
    from game_parser import GameBoardParser, analyze_captured_frame

    args = SimpleNamespace(top=10, overlay=None, show_overlay_window=False, window_ms=0)
    frame = np.zeros((240, 108, 3), dtype=np.uint8)

    result = analyze_captured_frame(GameBoardParser(), frame, str(tmp_path / "state.json"), "test", args)

    assert result is None


def test_ice_target_scores_iced_base_tile():
    board = [["muffin_ice", "muffin", "muffin"]]
    path = ((0, 0), (0, 1), (0, 2))

    score, reasons = MovePathfinder().score_path(
        board,
        {"muffin": "0 / 26", "ice": "0 / 5"},
        "muffin",
        path,
    )

    assert score == 78.0
    assert any(reason.startswith("ice target") for reason in reasons)


def test_simulated_move_collapses_known_tiles_and_leaves_unknown_cells_empty():
    board = [
        ["red", "muffin", "biscuit"],
        ["red", "donut", "biscuit"],
        ["red", "muffin", "chocolate"],
    ]
    path = ((0, 0), (1, 0), (2, 0), (1, 1))

    collapsed = FutureBoardEvaluator().simulate_after_move(board, path)

    assert collapsed == [
        ["EMPTY", "EMPTY", "biscuit"],
        ["EMPTY", "muffin", "biscuit"],
        ["EMPTY", "muffin", "chocolate"],
    ]


def test_gravity_lets_pieces_pass_through_existing_empty_cells():
    board = [
        ["muffin"],
        ["EMPTY"],
        ["red"],
        ["EMPTY"],
        ["biscuit"],
    ]

    collapsed = FutureBoardEvaluator().simulate_after_move(board, ())

    assert collapsed == [
        ["EMPTY"],
        ["EMPTY"],
        ["muffin"],
        ["red"],
        ["biscuit"],
    ]


def test_future_ice_score_requires_collecting_the_iced_cell_itself():
    adjacent_only_board = [
        ["muffin", "muffin", "muffin"],
        ["muffin", "red_ice", "muffin"],
        ["muffin", "muffin", "muffin"],
    ]
    directly_collectable_board = [["red_ice", "red", "red"]]

    evaluator = FutureBoardEvaluator()
    adjacent_only = evaluator.evaluate(adjacent_only_board, {"ice": "0 / 1"})
    directly_collectable = evaluator.evaluate(directly_collectable_board, {"ice": "0 / 1"})

    assert adjacent_only.ice_target_score == 0.0
    assert directly_collectable.ice_target_score > adjacent_only.ice_target_score


def test_future_evaluator_rewards_clusters_over_orphans():
    clustered_board = [
        ["muffin", "muffin", "EMPTY"],
        ["muffin", "muffin", "EMPTY"],
        ["EMPTY", "EMPTY", "EMPTY"],
    ]
    orphan_board = [
        ["muffin", "EMPTY", "biscuit"],
        ["EMPTY", "red", "EMPTY"],
        ["donut", "EMPTY", "chocolate"],
    ]

    evaluator = FutureBoardEvaluator()
    clustered = evaluator.evaluate(clustered_board)
    orphaned = evaluator.evaluate(orphan_board)

    assert clustered.orphan_count == 0
    assert orphaned.orphan_count == 5
    assert clustered.score > orphaned.score


def test_best_move_uses_future_board_state_in_reasons():
    board = [["muffin", "muffin", "muffin"], ["red", "red", "red"]]

    move = MovePathfinder().score_paths(board, {}, [((0, 0), (0, 1), (0, 2))])[0]

    assert any(reason.startswith("utility") for reason in move.reasons)
    assert any(reason.startswith("future clusters") for reason in move.reasons)


def test_draw_move_overlay_marks_suggested_path(tmp_path):
    import cv2
    import numpy as np
    from game_parser import GameBoardParser

    parser = GameBoardParser()
    parser.center_xs = [20, 60, 100]
    parser.center_ys = [20, 60, 100]
    parser.cell_w = 40
    parser.cell_h = 40
    image = np.zeros((140, 140, 3), dtype=np.uint8)
    move = {
        "item": "muffin",
        "length": 3,
        "score": 42.0,
        "path": [{"row": 0, "col": 0}, {"row": 1, "col": 1}, {"row": 2, "col": 2}],
    }
    out_path = tmp_path / "overlay.jpg"

    overlay = parser.draw_move_overlay(image, move, str(out_path))

    assert out_path.exists()
    assert overlay.sum() > image.sum()
    assert cv2.imread(str(out_path)) is not None


def test_scrcpy_stream_source_numeric_string_becomes_camera_index():
    from game_parser import ScrcpyStreamReader

    assert ScrcpyStreamReader("2").source == 2
    assert ScrcpyStreamReader("/dev/video2").source == "/dev/video2"


def test_write_json_result_creates_parent_directory(tmp_path):
    import json
    from game_parser import write_json_result

    out_path = tmp_path / "nested" / "state.json"

    write_json_result({"bestMove": None}, str(out_path))

    assert json.loads(out_path.read_text()) == {"bestMove": None}


def test_explicit_adb_path_is_preserved():
    from game_parser import AdbScreenReader

    explicit = r"C:\Android\platform-tools\adb.exe"

    assert AdbScreenReader.resolve_adb_path(explicit) == explicit


def test_random_game_simulator_tracks_three_hit_ice_blocks(tmp_path):
    import json
    from game_parser import RandomGameSimulator, SimulationConfig

    etalon_dir = tmp_path / "etalon"
    etalon_dir.mkdir()
    (etalon_dir / "level.json").write_text(json.dumps({
        "gameState": {"movesLeft": "3", "level": "16", "targets": {"muffin": "0 / 3", "ice": "0 / 1"}},
        "board": [["muffin_ice"]],
    }), encoding="utf-8")
    simulator = RandomGameSimulator(str(etalon_dir), SimulationConfig(games=1, seed=1, rows=1, cols=3, ice_hits=3))
    board = [["muffin", "muffin", "muffin"]]
    ice_hp = [[3, 0, 0]]
    targets = {"muffin": 99, "ice": 1}

    simulator._apply_move(board, ice_hp, ((0, 0),), targets)
    simulator._refill_board(board, ice_hp)
    simulator._apply_move(board, ice_hp, ((0, 0),), targets)
    simulator._refill_board(board, ice_hp)
    assert targets["ice"] == 1
    assert ice_hp[0][0] == 1

    simulator._apply_move(board, ice_hp, ((0, 0),), targets)

    assert targets["ice"] == 0
    assert ice_hp[0][0] == 0


def test_simulation_report_contains_success_rate():
    from game_parser import RandomGameSimulator, SimulationConfig

    report = RandomGameSimulator("etalon_images", SimulationConfig(games=2, seed=7)).run_many(2)

    assert report["summary"]["games"] == 2
    assert "successRate" in report["summary"]
    assert report["config"]["iceHits"] == 3


def test_adb_play_move_converts_board_path_to_swipe_segments():
    from game_parser import AdbScreenReader, GameBoardParser

    calls = []

    class FakeReader(AdbScreenReader):
        def __init__(self):
            pass

        def ensure_device(self):
            return None

        def run_shell(self, *args, timeout=10):
            calls.append(args)

    parser = GameBoardParser()
    parser.center_xs = [10, 20, 30]
    parser.center_ys = [100, 200, 300]
    move = {"path": [{"row": 0, "col": 0}, {"row": 1, "col": 1}, {"row": 2, "col": 2}]}

    segments = FakeReader().play_move(parser, move, duration_ms=77)

    assert segments == 2
    assert calls[0] == ("input", "swipe", 10, 100, 20, 200, 77)
    assert calls[1] == ("input", "swipe", 20, 200, 30, 300, 77)

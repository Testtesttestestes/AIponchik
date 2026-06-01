from game_parser import MovePathfinder


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

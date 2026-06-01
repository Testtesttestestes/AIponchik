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

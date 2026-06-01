"""Game parser entry point that ranks moves with the trained neural policy.

This script intentionally mirrors ``game_parser.py`` CLI modes (image, ADB,
scrcpy stream, batch directory, overlays, GUI/watch), but defaults to the
trained neural policy instead of the heuristic-only move ranker.  The neural
policy receives 32 candidate moves per board state by default.
"""

from game_parser import DEFAULT_POLICY_MODEL, main


if __name__ == "__main__":
    main(default_policy_model=DEFAULT_POLICY_MODEL)

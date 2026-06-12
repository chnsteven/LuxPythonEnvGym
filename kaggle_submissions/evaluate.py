#!/usr/bin/env python3
"""
Run head-to-head Lux AI 2021 matches between two agents with random seeds.

Example (from kaggle_submissions/):
    python evaluate.py
    python evaluate.py --games 50 --maxtime 10000
    python evaluate.py --agent0 agent1/main.py --agent1 agent2/main.py --master-seed 0
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple


SCRIPT_DIR = Path(__file__).resolve().parent


@dataclass
class GameResult:
    game: int
    seed: int
    agent0: str
    agent1: str
    winner_agent_id: Optional[int]
    winner_agent: Optional[str]
    status: str
    replay_path: Optional[str]
    duration_s: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate two Lux AI agents over multiple random seeds."
    )
    parser.add_argument("--agent0", default="agent1/main.py", help="Player 0 agent path")
    parser.add_argument("--agent1", default="agent2/main.py", help="Player 1 agent path")
    parser.add_argument("--games", type=int, default=50, help="Number of matches")
    parser.add_argument(
        "--maxtime", type=int, default=10000, help="Max ms per turn (lux-ai-2021 --maxtime)"
    )
    parser.add_argument(
        "--master-seed",
        type=int,
        default=None,
        help="Fix RNG for seed generation (reproducible seed list)",
    )
    parser.add_argument(
        "--seed-min",
        type=int,
        default=1,
        help="Minimum random seed value (inclusive)",
    )
    parser.add_argument(
        "--seed-max",
        type=int,
        default=2_000_000_000,
        help="Maximum random seed value (exclusive)",
    )
    parser.add_argument(
        "--swap-sides",
        action="store_true",
        help="Alternate agent order each game (odd games swap player 0/1)",
    )
    parser.add_argument(
        "--lux-cli",
        default="lux-ai-2021",
        help="lux-ai-2021 executable name or path",
    )
    parser.add_argument(
        "--replay-dir",
        default="replays/eval",
        help="Directory for match replays",
    )
    parser.add_argument(
        "--results-dir",
        default="eval_results",
        help="Directory for summary JSON/CSV",
    )
    parser.add_argument(
        "--store-logs",
        action="store_true",
        help="Keep per-match error logs (lux-ai-2021 --storeLogs)",
    )
    parser.add_argument(
        "--loglevel",
        type=int,
        default=1,
        help="lux-ai-2021 log level (0-4); use 1 to reduce log volume",
    )
    parser.add_argument(
        "--match-timeout",
        type=int,
        default=600,
        help="Seconds before giving up on a single match (0 = no limit)",
    )
    return parser.parse_args()


def resolve_lux_cli(lux_cli: str) -> str:
    """Resolve lux-ai-2021 to an absolute path (needed for Windows .cmd shims)."""
    if Path(lux_cli).exists():
        return str(Path(lux_cli).resolve())
    resolved = shutil.which(lux_cli)
    if resolved is None:
        raise FileNotFoundError(
            f"Cannot find '{lux_cli}'. Install with: npm install -g @lux-ai/2021-challenge"
        )
    return resolved


def make_seeds(games: int, master_seed: Optional[int], seed_min: int, seed_max: int) -> List[int]:
    rng = random.Random(master_seed)
    return [rng.randrange(seed_min, seed_max) for _ in range(games)]


def parse_winner(replay_path: Path) -> Tuple[Optional[int], str]:
    with replay_path.open(encoding="utf-8") as f:
        data = json.load(f)

    ranks = data.get("results", {}).get("ranks")
    if not ranks:
        return None, "no_results"

    winner_id = None
    for entry in ranks:
        if entry.get("rank") == 1:
            winner_id = entry.get("agentID")
            break

    if winner_id is None:
        return None, "no_winner"

    return int(winner_id), "ok"


def run_match(
    lux_cli: str,
    agent0: str,
    agent1: str,
    seed: int,
    maxtime: int,
    replay_path: Path,
    store_logs: bool,
    loglevel: int,
    match_timeout: int,
) -> Tuple[int, str]:
    replay_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        lux_cli,
        agent0,
        agent1,
        "--maxtime",
        str(maxtime),
        "--seed",
        str(seed),
        "--out",
        str(replay_path),
        "--loglevel",
        str(loglevel),
        "--storeReplay",
        "true",
        "--storeLogs",
        "true" if store_logs else "false",
    ]
    # Do not capture stdout/stderr: lux-ai-2021 is verbose and can deadlock
    # when the pipe buffer fills while subprocess.run() is waiting.
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(SCRIPT_DIR),
            timeout=match_timeout if match_timeout > 0 else None,
        )
        return proc.returncode, ""
    except subprocess.TimeoutExpired as exc:
        tail = ""
        if exc.stderr:
            tail = exc.stderr.decode(errors="replace")[-500:]
        return -1, f"match timeout after {match_timeout}s\n{tail}"


def winner_agent_path(
    winner_id: Optional[int], agent0: str, agent1: str, swapped: bool
) -> Optional[str]:
    if winner_id is None:
        return None
    if swapped:
        return agent1 if winner_id == 0 else agent0
    return agent0 if winner_id == 0 else agent1


def write_summary(
    results: List[GameResult],
    args: argparse.Namespace,
    run_id: str,
) -> Path:
    out_dir = SCRIPT_DIR / args.results_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    wins0 = sum(1 for r in results if r.status == "ok" and r.winner_agent == args.agent0)
    wins1 = sum(1 for r in results if r.status == "ok" and r.winner_agent == args.agent1)
    errors = sum(1 for r in results if r.status != "ok")
    completed = len(results) - errors

    summary = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "agent0": args.agent0,
        "agent1": args.agent1,
        "games": completed,
        "failed": errors,
        "wins_agent0": wins0,
        "wins_agent1": wins1,
        "win_rate_agent0": round(wins0 / completed, 4) if completed else 0.0,
        "win_rate_agent1": round(wins1 / completed, 4) if completed else 0.0,
    }

    json_path = out_dir / f"summary_{run_id}.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return json_path


def main() -> int:
    args = parse_args()
    try:
        lux_cli = resolve_lux_cli(args.lux_cli)
    except FileNotFoundError as exc:
        print(exc, file=sys.stderr)
        return 1

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    seeds = make_seeds(args.games, args.master_seed, args.seed_min, args.seed_max)
    replay_dir = SCRIPT_DIR / args.replay_dir / run_id

    print(f"Lux AI evaluation run: {run_id}")
    print(f"  agent0 : {args.agent0}")
    print(f"  agent1 : {args.agent1}")
    print(f"  games  : {args.games}")
    print(f"  maxtime: {args.maxtime} ms/turn")
    if args.master_seed is not None:
        print(f"  master seed: {args.master_seed}")
    print(f"  replays: {replay_dir}")
    print()

    results: List[GameResult] = []

    for i, seed in enumerate(seeds, start=1):
        swapped = args.swap_sides and i % 2 == 0
        if swapped:
            agent0, agent1 = args.agent1, args.agent0
            slot0_name, slot1_name = args.agent1, args.agent0
        else:
            agent0, agent1 = args.agent0, args.agent1
            slot0_name, slot1_name = args.agent0, args.agent1

        replay_path = replay_dir / f"game{i:03d}_seed{seed}.json"
        print(f"[{i:>2}/{args.games}] seed={seed} ... ", end="", flush=True)

        t0 = time.time()
        returncode, output = run_match(
            lux_cli,
            agent0,
            agent1,
            seed,
            args.maxtime,
            replay_path,
            args.store_logs,
            args.loglevel,
            args.match_timeout,
        )
        duration = time.time() - t0

        winner_id: Optional[int] = None
        status = "ok"

        if returncode != 0:
            status = f"cli_error_{returncode}"
        elif not replay_path.exists():
            status = "missing_replay"
        else:
            winner_id, parse_status = parse_winner(replay_path)
            if parse_status != "ok":
                status = parse_status

        winner = winner_agent_path(winner_id, args.agent0, args.agent1, swapped)
        results.append(
            GameResult(
                game=i,
                seed=seed,
                agent0=slot0_name,
                agent1=slot1_name,
                winner_agent_id=winner_id,
                winner_agent=winner,
                status=status,
                replay_path=str(replay_path.relative_to(SCRIPT_DIR)) if replay_path.exists() else None,
                duration_s=round(duration, 1),
            )
        )

        if status == "ok":
            print(f"{winner} wins ({duration:.0f}s)")
        else:
            print(f"FAILED ({status}, {duration:.0f}s)")
            if output.strip():
                tail = output.strip().splitlines()[-3:]
                for line in tail:
                    print(f"    {line}")

    wins0 = sum(1 for r in results if r.status == "ok" and r.winner_agent == args.agent0)
    wins1 = sum(1 for r in results if r.status == "ok" and r.winner_agent == args.agent1)
    ok_games = sum(1 for r in results if r.status == "ok")
    failed = len(results) - ok_games

    print()
    print("=" * 60)
    print(f"Completed: {ok_games}/{args.games}  |  Failed: {failed}")
    print(f"{args.agent0}: {wins0} wins ({100 * wins0 / ok_games:.1f}%)" if ok_games else f"{args.agent0}: 0 wins")
    print(f"{args.agent1}: {wins1} wins ({100 * wins1 / ok_games:.1f}%)" if ok_games else f"{args.agent1}: 0 wins")
    print("=" * 60)

    summary_path = write_summary(results, args, run_id)
    print(f"Saved: {summary_path}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Clean training launcher for EfficientZero A/B experiment.

Usage:
  # Baseline (no SupCon)
  python scripts/train.py --env PongNoFrameskip-v4 --mode baseline

  # SupCon experiment
  python scripts/train.py --env PongNoFrameskip-v4 --mode supcon --supcon_coeff 0.5

  # Full A/B comparison across multiple games
  python scripts/train.py --mode ab --games Pong Breakout Qbert --seeds 0 1 2

  # Monitor an existing run's tensorboard logs
  python scripts/train.py --monitor results/supcon/
"""

import argparse
import math
import os
import subprocess
import sys
import time
from pathlib import Path

# Rich imports
from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn, SpinnerColumn
from rich.table import Table
from rich.text import Text

console = Console()

# ── Game presets ──────────────────────────────────────────────────────

ATARI_100K_GAMES = [
    "PongNoFrameskip-v4",
    "BreakoutNoFrameskip-v4",
    "QbertNoFrameskip-v4",
    "SeaquestNoFrameskip-v4",
    "MsPacmanNoFrameskip-v4",
]

GAME_ALIASES = {
    "Pong": "PongNoFrameskip-v4",
    "Breakout": "BreakoutNoFrameskip-v4",
    "Qbert": "QbertNoFrameskip-v4",
    "Seaquest": "SeaquestNoFrameskip-v4",
    "MsPacman": "MsPacmanNoFrameskip-v4",
    "Alien": "AlienNoFrameskip-v4",
    "BankHeist": "BankHeistNoFrameskip-v4",
    "Boxing": "BoxingNoFrameskip-v4",
    "Freeway": "FreewayNoFrameskip-v4",
    "Hero": "HeroNoFrameskip-v4",
}


def resolve_game(name):
    if name in GAME_ALIASES:
        return GAME_ALIASES[name]
    if "NoFrameskip" in name:
        return name
    return f"{name}NoFrameskip-v4"


# ── Display helpers ───────────────────────────────────────────────────

def make_header(mode, games, seeds, supcon_coeff):
    t = Table(show_header=False, box=None, padding=(0, 2))
    t.add_column(style="bold cyan")
    t.add_column()
    t.add_row("Mode", mode.upper())
    t.add_row("Games", ", ".join(g.replace("NoFrameskip-v4", "") for g in games))
    t.add_row("Seeds", ", ".join(str(s) for s in seeds))
    if mode == "supcon" or mode == "ab":
        t.add_row("SupCon coeff", str(supcon_coeff))
        t.add_row("Schedule", "cosine ramp-up over first 50% of training")
        t.add_row("SupCon target", "encoder only (representation network)")
    return Panel(t, title="[bold bright_cyan]EfficientZero + SupCon Experiment[/]", border_style="cyan")


def make_schedule_table(total_steps, warmup_frac, supcon_coeff):
    """Show the SupCon ramp-up schedule at key checkpoints."""
    t = Table(title="SupCon Coefficient Schedule", border_style="dim")
    t.add_column("Step", style="dim")
    t.add_column("Progress", style="dim")
    t.add_column("SupCon coeff", style="bold green")
    t.add_column("", style="dim")

    checkpoints = [0, 0.1, 0.25, 0.5, 0.75, 1.0]
    for frac in checkpoints:
        step = int(frac * total_steps)
        if warmup_frac > 0 and frac < warmup_frac:
            progress = frac / warmup_frac
            ramp = 0.5 * (1 - math.cos(math.pi * progress))
            eff = supcon_coeff * ramp
        else:
            eff = supcon_coeff
        bar = "+" * int(eff / supcon_coeff * 20) if supcon_coeff > 0 else ""
        t.add_row(f"{step:,}", f"{frac:.0%}", f"{eff:.4f}", bar)

    return t


def make_run_status(runs):
    """Build a table showing status of all runs."""
    t = Table(title="Run Status", border_style="blue")
    t.add_column("Game", style="cyan")
    t.add_column("Seed")
    t.add_column("Mode", style="bold")
    t.add_column("Status")
    t.add_column("Time")

    for r in runs:
        if r["status"] == "running":
            status = Text("RUNNING", style="bold yellow")
        elif r["status"] == "done":
            status = Text("DONE", style="bold green")
        elif r["status"] == "failed":
            status = Text("FAILED", style="bold red")
        else:
            status = Text("PENDING", style="dim")

        elapsed = ""
        if r.get("start_time"):
            e = (r.get("end_time") or time.time()) - r["start_time"]
            elapsed = f"{e/60:.1f}m"

        game_short = r["game"].replace("NoFrameskip-v4", "")
        t.add_row(game_short, str(r["seed"]), r["mode"], status, elapsed)

    return t


# ── Run logic ─────────────────────────────────────────────────────────

def build_cmd(args, game, seed, mode, supcon_coeff):
    """Build the main.py command for a single run."""
    root = Path(__file__).parent.parent
    cmd = [
        sys.executable, str(root / "main.py"),
        "--env", game,
        "--case", "atari",
        "--opr", "train",
        "--amp_type", args.amp_type,
        "--num_gpus", str(args.num_gpus),
        "--num_cpus", str(args.num_cpus),
        "--gpu_actor", str(args.gpu_actor),
        "--cpu_actor", str(args.cpu_actor),
        "--seed", str(seed),
        "--use_priority",
        "--use_max_priority",
        "--use_augmentation",
        "--info", f"{mode}_s{seed}",
        "--result_dir", str(root / "results" / mode),
    ]

    if mode == "supcon":
        cmd += [
            "--supcon_coeff", str(supcon_coeff),
            "--supcon_num_bins", str(args.supcon_num_bins),
            "--supcon_min_bin_size", str(args.supcon_min_bin_size),
            "--supcon_warmup_fraction", str(args.supcon_warmup_fraction),
        ]
    else:
        cmd += ["--supcon_coeff", "0.0"]

    return cmd


def run_single(args, game, seed, mode, supcon_coeff):
    """Run a single training and stream output."""
    cmd = build_cmd(args, game, seed, mode, supcon_coeff)
    game_short = game.replace("NoFrameskip-v4", "")

    console.print()
    console.rule(f"[bold]{mode.upper()}: {game_short} (seed={seed})[/]")
    console.print(f"[dim]$ {' '.join(cmd)}[/]")
    console.print()

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=env,
        cwd=str(Path(__file__).parent.parent),
    )

    try:
        for line in proc.stdout:
            line = line.rstrip()
            # Highlight loss lines
            if "Total Loss" in line:
                console.print(f"  [green]{line}[/]")
            elif "Test Mean Score" in line:
                console.print(f"  [bold bright_cyan]{line}[/]")
            elif "Warning" in line:
                console.print(f"  [yellow]{line}[/]")
            elif "Error" in line or "error" in line:
                console.print(f"  [red]{line}[/]")
            else:
                console.print(f"  [dim]{line}[/]")
    except KeyboardInterrupt:
        proc.terminate()
        console.print("[bold red]Interrupted.[/]")
        sys.exit(1)

    proc.wait()
    return proc.returncode


def run_ab(args):
    """Run full A/B comparison."""
    games = [resolve_game(g) for g in args.games]
    seeds = args.seeds
    supcon_coeff = args.supcon_coeff

    # Build run list
    runs = []
    for game in games:
        for seed in seeds:
            runs.append({"game": game, "seed": seed, "mode": "baseline", "status": "pending"})
            runs.append({"game": game, "seed": seed, "mode": "supcon", "status": "pending"})

    # Header
    console.print(make_header("ab", games, seeds, supcon_coeff))
    console.print()
    console.print(make_schedule_table(100000 + 20000, args.supcon_warmup_fraction, supcon_coeff))
    console.print()
    console.print(f"[bold]Total runs: {len(runs)}[/] ({len(games)} games x {len(seeds)} seeds x 2 modes)")
    console.print()

    # Run sequentially
    for i, run in enumerate(runs):
        run["status"] = "running"
        run["start_time"] = time.time()

        console.print(make_run_status(runs))

        rc = run_single(args, run["game"], run["seed"], run["mode"],
                        supcon_coeff if run["mode"] == "supcon" else 0.0)

        run["end_time"] = time.time()
        run["status"] = "done" if rc == 0 else "failed"

    # Final summary
    console.print()
    console.print(make_run_status(runs))
    console.print()

    done = sum(1 for r in runs if r["status"] == "done")
    failed = sum(1 for r in runs if r["status"] == "failed")

    if failed == 0:
        console.print("[bold green]All runs complete![/]")
    else:
        console.print(f"[bold yellow]{done} succeeded, {failed} failed.[/]")

    console.print()
    console.print("[bold]Compare results:[/]")
    console.print(f"  tensorboard --logdir results/")
    console.print()
    console.print("[dim]Baseline logs: results/baseline/")
    console.print(f"[dim]SupCon logs:   results/supcon/")


# ── CLI ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="EfficientZero + SupCon training launcher",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single baseline run
  python scripts/train.py --env Pong --mode baseline

  # Single SupCon run
  python scripts/train.py --env Pong --mode supcon --supcon_coeff 0.5

  # Full A/B across 5 games, 3 seeds
  python scripts/train.py --mode ab --games Pong Breakout Qbert Seaquest MsPacman --seeds 0 1 2
        """,
    )

    parser.add_argument("--env", type=str, default="PongNoFrameskip-v4",
                        help="Atari game (for single runs)")
    parser.add_argument("--mode", choices=["baseline", "supcon", "ab"], default="supcon",
                        help="baseline / supcon / ab (A/B comparison)")
    parser.add_argument("--games", nargs="+", default=["Pong", "Breakout", "Qbert", "Seaquest", "MsPacman"],
                        help="Games for A/B mode")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2],
                        help="Random seeds")

    # SupCon params
    parser.add_argument("--supcon_coeff", type=float, default=0.5,
                        help="SupCon loss coefficient (default: 0.5)")
    parser.add_argument("--supcon_num_bins", type=int, default=32,
                        help="Number of quantile bins (default: 32)")
    parser.add_argument("--supcon_min_bin_size", type=int, default=2,
                        help="Min samples per bin (default: 2)")
    parser.add_argument("--supcon_warmup_fraction", type=float, default=0.5,
                        help="Fraction of training for SupCon cosine ramp-up (default: 0.5)")

    # Infra params
    parser.add_argument("--amp_type", default="torch_amp", choices=["torch_amp", "none"])
    parser.add_argument("--num_gpus", type=int, default=1)
    parser.add_argument("--num_cpus", type=int, default=16)
    parser.add_argument("--gpu_actor", type=int, default=2)
    parser.add_argument("--cpu_actor", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0, help="Seed for single runs")

    args = parser.parse_args()

    if args.mode == "ab":
        run_ab(args)
    else:
        game = resolve_game(args.env)
        supcon_coeff = args.supcon_coeff if args.mode == "supcon" else 0.0

        console.print(make_header(args.mode, [game], [args.seed], supcon_coeff))
        if args.mode == "supcon":
            console.print()
            console.print(make_schedule_table(120000, args.supcon_warmup_fraction, supcon_coeff))

        rc = run_single(args, game, args.seed, args.mode, supcon_coeff)

        if rc == 0:
            console.print()
            console.print("[bold green]Training complete![/]")
            console.print(f"  tensorboard --logdir results/{args.mode}/")
        else:
            console.print()
            console.print(f"[bold red]Training failed (exit code {rc})[/]")
            sys.exit(rc)


if __name__ == "__main__":
    main()

"""
Long-running skill curriculum for the quadruped.

This script automates the debugging loop:

1. Train stable flat-ground running.
2. Continue from the running model to train a low jump/platform skill.
3. Continue from the jump model to train low stair climbing.
4. Record videos and write a markdown log after every stage.

Example:
    python3 examples/run_quadruped_skill_curriculum.py --hours 10 --num_envs 8
"""

import argparse
import ast
import datetime as dt
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
TRAINER = REPO_ROOT / "examples" / "quadruped_ppo_residual_stairs.py"


def now_stamp():
    return dt.datetime.now().strftime("%Y%m%d_%H%M%S")


def run_command(cmd, log_path):
    start = time.time()
    with log_path.open("w", encoding="utf-8") as log:
        log.write("$ " + " ".join(str(x) for x in cmd) + "\n\n")
        log.flush()
        proc = subprocess.Popen(
            cmd,
            cwd=REPO_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        captured = []
        for line in proc.stdout:
            print(line, end="")
            log.write(line)
            captured.append(line)
        code = proc.wait()
        log.write(f"\n[exit_code] {code}\n[elapsed_s] {time.time() - start:.1f}\n")
    if code != 0:
        raise RuntimeError(f"Command failed with code {code}: {' '.join(map(str, cmd))}")
    return "".join(captured)


def parse_episode_info(output):
    for line in reversed(output.splitlines()):
        if line.startswith("episode ended:"):
            payload = line.split("episode ended:", 1)[1].strip()
            try:
                return ast.literal_eval(payload)
            except (SyntaxError, ValueError):
                return {}
    return {}


def metric_for(task, info):
    if task == "run":
        up = float(info.get("body_up_z", 0.0))
        x = float(info.get("forward_dist", 0.0))
        status_bonus = 2.0 if info.get("status") == "timeout" else 0.0
        return x + status_bonus + max(up - 0.88, 0.0)
    if task == "jump":
        return (
            3.0 * float(info.get("climb", 0.0))
            + float(info.get("forward_dist", 0.0))
            + 0.5 * int(info.get("stair_toe_contacts", 0))
        )
    return (
        4.0 * float(info.get("climb", 0.0))
        + float(info.get("forward_dist", 0.0))
        + 2.0 * (1.0 if info.get("status") == "success" else 0.0)
    )


def append_summary(summary_path, text):
    with summary_path.open("a", encoding="utf-8") as f:
        f.write(text.rstrip() + "\n\n")


def train_stage(args, run_dir, cycle, task, timesteps, load_model, model_path):
    log_path = run_dir / "logs" / f"{cycle:02d}_{task}_train.log"
    cmd = [
        sys.executable,
        str(TRAINER),
        "train",
        "--task",
        task,
        "--timesteps",
        str(timesteps),
        "--num_envs",
        str(args.num_envs),
        "--batch_size",
        str(args.batch_size),
        "--curriculum",
        "--model",
        str(model_path),
    ]
    if load_model and Path(load_model).exists():
        cmd.extend(["--load_model", str(load_model)])
    run_command(cmd, log_path)


def demo_stage(args, run_dir, cycle, task, model_path):
    video_path = run_dir / "videos" / f"{cycle:02d}_{task}.mp4"
    log_path = run_dir / "logs" / f"{cycle:02d}_{task}_demo.log"
    cmd = [
        sys.executable,
        str(TRAINER),
        "demo",
        "--task",
        task,
        "--model",
        str(model_path),
        "--steps",
        str(args.demo_steps),
        "--record",
        str(video_path),
    ]
    if task == "jump":
        cmd.extend(["--stair_steps", "1", "--step_height", "0.035", "--init_x", "0.20"])
    elif task == "stairs":
        cmd.extend(["--stair_steps", "4", "--step_height", "0.03", "--init_x", "0.00"])
    output = run_command(cmd, log_path)
    return video_path, parse_episode_info(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hours", type=float, default=10.0)
    parser.add_argument("--num_envs", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=2048)
    parser.add_argument("--demo_steps", type=int, default=500)
    parser.add_argument("--run_timesteps", type=int, default=300_000)
    parser.add_argument("--jump_timesteps", type=int, default=400_000)
    parser.add_argument("--stairs_timesteps", type=int, default=500_000)
    parser.add_argument("--out_dir", default="")
    parser.add_argument("--seed_run_model", default=str(REPO_ROOT / "examples" / "ppo_run_flat.zip"))
    args = parser.parse_args()

    root = Path(args.out_dir) if args.out_dir else REPO_ROOT / "examples" / "rl_runs" / now_stamp()
    run_dir = root.resolve()
    for sub in ("models", "videos", "logs", "important_videos"):
        (run_dir / sub).mkdir(parents=True, exist_ok=True)

    summary = run_dir / "SUMMARY.md"
    append_summary(
        summary,
        "# Quadruped Skill Curriculum Run\n"
        f"- Started: {dt.datetime.now().isoformat(timespec='seconds')}\n"
        f"- Time budget: {args.hours} hours\n"
        f"- Output directory: `{run_dir}`\n",
    )

    deadline = time.time() + args.hours * 3600.0
    best = {"run": (-1e9, None), "jump": (-1e9, None), "stairs": (-1e9, None)}
    previous = {
        "run": Path(args.seed_run_model) if args.seed_run_model else None,
        "jump": None,
        "stairs": None,
    }
    cycle = 1

    while time.time() < deadline:
        append_summary(summary, f"## Cycle {cycle}\n")

        stages = [
            ("run", args.run_timesteps, previous["run"]),
            ("jump", args.jump_timesteps, previous["run"]),
            ("stairs", args.stairs_timesteps, previous["jump"]),
        ]

        for task, timesteps, load_model in stages:
            if time.time() >= deadline:
                break
            model_path = run_dir / "models" / f"{cycle:02d}_{task}.zip"
            train_stage(args, run_dir, cycle, task, timesteps, load_model, model_path)
            video_path, info = demo_stage(args, run_dir, cycle, task, model_path)
            score = metric_for(task, info)
            previous[task] = model_path
            append_summary(
                summary,
                f"### {task}\n"
                f"- Model: `{model_path}`\n"
                f"- Video: `{video_path}`\n"
                f"- Info: `{info}`\n"
                f"- Metric: `{score:.3f}`\n",
            )
            if score > best[task][0]:
                best[task] = (score, video_path)
                important = run_dir / "important_videos" / f"best_{task}_{cycle:02d}.mp4"
                shutil.copy2(video_path, important)
                append_summary(summary, f"- New best {task} video: `{important}`\n")

        cycle += 1

    append_summary(
        summary,
        "## Final Best Videos\n"
        + "\n".join(
            f"- {task}: score={score:.3f}, video=`{video}`"
            for task, (score, video) in best.items()
        )
        + f"\n\nFinished: {dt.datetime.now().isoformat(timespec='seconds')}\n",
    )
    print(f"Long run complete. Summary: {summary}")


if __name__ == "__main__":
    main()

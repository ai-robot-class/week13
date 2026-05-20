# Week 13: Quadruped Reinforcement Learning

This repository contains the runnable code for Week 13 of the AI Robot class.
The main course website references this repository as a Git submodule.

## Files

```text
.
├── quadruped_ppo_residual_stairs.py      # PPO + residual controller trainer/demo
├── run_quadruped_skill_curriculum.py     # Long-running run -> jump -> stairs loop
├── quadruped_training_debug_notes.md     # Notes from the 7h22m training run
├── ppo_run_flat.zip                      # Pretrained flat-ground running model
├── ppo_residual_stairs.zip               # Pretrained low-stair model
├── quadruped_rl_stairs.py                # Older CMA-ES stair-climbing baseline
├── quadruped_rl_slope.py                 # Older CMA-ES slope baseline
├── train_log.json                        # Example training log
└── train_log_stairs.json                 # Example stair training log
```

## Install

```bash
pip install pybullet numpy gymnasium stable-baselines3 torch opencv-python
```

## Use the Pretrained Models

Flat-ground running:

```bash
python3 quadruped_ppo_residual_stairs.py demo \
    --task run \
    --model ppo_run_flat.zip \
    --steps 500 \
    --gui
```

Low-stair climbing progress:

```bash
python3 quadruped_ppo_residual_stairs.py demo \
    --task stairs \
    --model ppo_residual_stairs.zip \
    --stair_steps 4 \
    --step_height 0.03 \
    --init_x 0.00 \
    --steps 500 \
    --gui
```

Record a video instead of opening the GUI:

```bash
python3 quadruped_ppo_residual_stairs.py demo \
    --task stairs \
    --model ppo_residual_stairs.zip \
    --stair_steps 4 \
    --step_height 0.03 \
    --init_x 0.00 \
    --steps 500 \
    --record student_stairs_demo.mp4
```

## Train From Scratch

Train flat-ground running first:

```bash
python3 quadruped_ppo_residual_stairs.py train \
    --task run \
    --timesteps 300000 \
    --num_envs 8 \
    --batch_size 2048 \
    --curriculum \
    --model ppo_run_flat_student.zip
```

Continue into low-platform jumping:

```bash
python3 quadruped_ppo_residual_stairs.py train \
    --task jump \
    --load_model ppo_run_flat_student.zip \
    --timesteps 400000 \
    --num_envs 8 \
    --batch_size 2048 \
    --curriculum \
    --model ppo_jump_low_student.zip
```

Continue into low stairs:

```bash
python3 quadruped_ppo_residual_stairs.py train \
    --task stairs \
    --load_model ppo_jump_low_student.zip \
    --timesteps 500000 \
    --num_envs 8 \
    --batch_size 2048 \
    --curriculum \
    --model ppo_stairs_low_student.zip
```

## Continue From the Provided Stair Model

```bash
python3 quadruped_ppo_residual_stairs.py train \
    --task stairs \
    --load_model ppo_residual_stairs.zip \
    --timesteps 300000 \
    --num_envs 8 \
    --batch_size 2048 \
    --curriculum \
    --model student_stairs_continue.zip
```

## Long-Running Curriculum

```bash
python3 run_quadruped_skill_curriculum.py --hours 10 --num_envs 8
```

The long-running script creates an `rl_runs/<timestamp>/` directory with models,
logs, demo videos, and selected important videos.

## Important Note

`ppo_residual_stairs.zip` is not a perfect final stair-climbing policy. It is
the representative model from the class debugging session: the quadruped can
visibly climb about three low steps, but it does not yet satisfy the strict
"stand upright on the final stair with stable toe contacts" success condition.

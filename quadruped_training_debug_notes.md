# Quadruped RL Debug Notes

## Current Diagnosis

The stair task failed because the base gait was not stable enough on flat
ground. A zero-residual high-lift gait flipped in about one second, so PPO was
trying to solve balance, locomotion, jumping, and stair climbing at the same
time.

## Changes Made

- Split training into three tasks: `run`, `jump`, and `stairs`.
- Added `--load_model` so each stage can continue from the previous skill.
- Added stricter flat-ground running rewards:
  - stable forward distance is rewarded only when the body stays upright;
  - lateral drift reduces reward;
  - low body-up posture is penalized;
  - `body_up_z < 0.65` is treated as falling for the run task.
- Added task-specific gait:
  - `run` uses a lower, slower, more stable base gait;
  - `jump` and `stairs` keep the higher-lift gait.
- Lowered stair height to 3 cm for early stair training.
- Added near-start curriculum for jumping and stair approaches.

## Recommended Training Order

1. Train flat running until the demo shows upright forward motion.
2. Load the running policy and train low-platform jumping.
3. Load the jumping policy and train low stair climbing.
4. Only then increase step height, number of steps, and initial distance.

## Useful Commands

Train running:

```bash
python3 examples/quadruped_ppo_residual_stairs.py train \
  --task run --timesteps 300000 --num_envs 8 --batch_size 2048 \
  --curriculum --model examples/ppo_run_flat.zip
```

Train jumping from running:

```bash
python3 examples/quadruped_ppo_residual_stairs.py train \
  --task jump --load_model examples/ppo_run_flat.zip \
  --timesteps 400000 --num_envs 8 --batch_size 2048 \
  --curriculum --model examples/ppo_jump_low.zip
```

Train stairs from jumping:

```bash
python3 examples/quadruped_ppo_residual_stairs.py train \
  --task stairs --load_model examples/ppo_jump_low.zip \
  --timesteps 500000 --num_envs 8 --batch_size 2048 \
  --curriculum --model examples/ppo_stairs_low.zip
```

Run the long automatic curriculum:

```bash
python3 examples/run_quadruped_skill_curriculum.py --hours 10 --num_envs 8
```

## Video Judgement Rules

Do not call a clip successful unless:

- running: the body remains upright and moves forward for the full episode;
- jumping: feet contact the platform and the body does not collapse onto it;
- stairs: the robot reaches the final tread with toe contacts, upright posture,
  and holds the final position.


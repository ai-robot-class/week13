"""
PPO + residual controller for quadruped stair climbing.

- A hand-written trot generator provides a stable base gait.
- PPO learns a small residual correction on top of the gait targets.
- Success requires real toe contacts on the final stair, upright posture, and
  holding the final tread for a short time.

Dependencies:
    pip install pybullet numpy gymnasium stable-baselines3 torch opencv-python

Train:
    python3 quadruped_ppo_residual_stairs.py train --timesteps 1000000 --num_envs 8

Demo:
    python3 quadruped_ppo_residual_stairs.py demo --model ppo_residual_stairs.zip \
        --record ppo_residual_success.mp4
"""

import argparse
import math
import os
import time
from dataclasses import dataclass

import numpy as np
import pybullet as pb
import pybullet_data

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError as exc:  # pragma: no cover - user-facing dependency message
    raise SystemExit(
        "Missing gymnasium. Install with: pip install gymnasium stable-baselines3"
    ) from exc


SIM_DT = 1.0 / 240.0
CTRL_DT = 1.0 / 50.0
EPISODE_T = 8.0

STAIR_STEP_HEIGHT = 0.03
STAIR_STEP_DEPTH = 0.40
STAIR_NUM_STEPS = 4
STAIR_X_START = 0.60
STAIR_WIDTH = 3.0

INIT_HEIGHT = 0.42
INIT_X = 0.0
NOMINAL_THIGH = 0.65
NOMINAL_CALF = -1.20

TOE_LINK_IDS = [3, 7, 11, 15]
LEG_JOINTS = {
    "RF": [0, 1, 2],
    "LF": [4, 5, 6],
    "RH": [8, 9, 10],
    "LH": [12, 13, 14],
}
LEG_ORDER = ["RF", "LF", "RH", "LH"]
ACTION_SCALE = np.array([0.14, 0.30, 0.38] * 4, dtype=np.float32)

SUCCESS_HOLD_T = 0.45
SUCCESS_X_MARGIN = 0.12
SUCCESS_HEIGHT_TOL = 0.03


@dataclass
class GaitConfig:
    freq: float = 1.10
    lift: float = 0.34
    duty: float = 0.58
    stance_thigh: float = 0.70
    stance_calf: float = -1.25
    forward_bias: float = 0.12
    stride: float = 0.30
    calf_lift_gain: float = 2.20
    body_pitch_bias: float = -0.06
    hip_sway: float = 0.04


def smoothstep(s):
    s = float(np.clip(s, 0.0, 1.0))
    return s * s * (3.0 - 2.0 * s)


def build_stairs(
    client_id,
    num_steps=STAIR_NUM_STEPS,
    step_height=STAIR_STEP_HEIGHT,
    width=STAIR_WIDTH,
):
    palette = [
        [0.95, 0.40, 0.30, 1.0],
        [0.98, 0.75, 0.20, 1.0],
        [0.55, 0.85, 0.35, 1.0],
        [0.30, 0.65, 0.90, 1.0],
    ]
    stairs = []
    for i in range(num_steps):
        half = [
            STAIR_STEP_DEPTH / 2,
            width / 2,
            step_height * (i + 1) / 2,
        ]
        col = pb.createCollisionShape(
            pb.GEOM_BOX, halfExtents=half, physicsClientId=client_id
        )
        vis = pb.createVisualShape(
            pb.GEOM_BOX,
            halfExtents=half,
            rgbaColor=palette[i % len(palette)],
            physicsClientId=client_id,
        )
        pos = [
            STAIR_X_START + STAIR_STEP_DEPTH / 2 + i * STAIR_STEP_DEPTH,
            0,
            step_height * (i + 1) / 2,
        ]
        body = pb.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=col,
            baseVisualShapeIndex=vis,
            basePosition=pos,
            physicsClientId=client_id,
        )
        pb.changeDynamics(
            body,
            -1,
            lateralFriction=1.35,
            spinningFriction=0.03,
            rollingFriction=0.01,
            physicsClientId=client_id,
        )
        stairs.append(body)
    return stairs


def gait_targets(t, leg_name, cfg):
    phases = {"LF": 0.0, "RH": 0.0, "RF": 0.5, "LH": 0.5}
    phi = (t * cfg.freq + phases[leg_name]) % 1.0

    if phi < cfg.duty:
        s = smoothstep(phi / max(cfg.duty, 1e-3))
        thigh_sway = cfg.stride * (s - 0.5)
        foot_lift = 0.0
    else:
        s = smoothstep((phi - cfg.duty) / max(1.0 - cfg.duty, 1e-3))
        foot_lift = cfg.lift * math.sin(math.pi * s)
        thigh_sway = cfg.stride * (0.5 - s)

    is_left = 1.0 if leg_name in ("LF", "LH") else -1.0
    is_front = 1.0 if leg_name in ("LF", "RF") else -1.0
    hip = is_left * cfg.hip_sway * math.sin(2.0 * math.pi * phi)
    thigh = (
        cfg.stance_thigh
        + thigh_sway
        + cfg.forward_bias
        + cfg.body_pitch_bias * is_front
        + 0.80 * foot_lift
    )
    calf = cfg.stance_calf - cfg.calf_lift_gain * foot_lift
    return np.array([hip, thigh, calf], dtype=np.float32)


class ResidualStairsEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    def __init__(
        self,
        gui=False,
        record=False,
        stair_steps=STAIR_NUM_STEPS,
        step_height=STAIR_STEP_HEIGHT,
        init_x=INIT_X,
        task="stairs",
    ):
        super().__init__()
        self.gui = gui
        self.record = record
        self.client_id = pb.connect(pb.GUI if gui else pb.DIRECT)
        pb.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=self.client_id)
        pb.setGravity(0, 0, -9.81, physicsClientId=self.client_id)
        pb.setTimeStep(SIM_DT, physicsClientId=self.client_id)

        self.gait = GaitConfig()
        self.stair_steps = int(stair_steps)
        self.step_height = float(step_height)
        self.init_x = float(init_x)
        self.task = task
        self.pending_curriculum = None
        self.robot = None
        self.stairs = []
        self.final_stair = None
        self.t = 0.0
        self.prev_action = np.zeros(12, dtype=np.float32)
        self.prev_x = 0.0
        self.best_climb = 0.0
        self.success_timer = 0.0

        self.action_space = spaces.Box(-1.0, 1.0, shape=(12,), dtype=np.float32)
        # Compact proprioceptive observation. This stays deliberately simple for teaching.
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(54,), dtype=np.float32)
        self._build_world()

    @property
    def stair_top_x(self):
        return STAIR_X_START + self.stair_steps * STAIR_STEP_DEPTH

    @property
    def stair_top_height(self):
        return self.stair_steps * self.step_height

    def _build_world(self):
        if self.robot is not None:
            pb.removeBody(self.robot, physicsClientId=self.client_id)
            self.robot = None
        for stair in self.stairs:
            pb.removeBody(stair, physicsClientId=self.client_id)
        self.stairs = []

        plane = pb.loadURDF("plane.urdf", physicsClientId=self.client_id)
        pb.changeDynamics(
            plane,
            -1,
            lateralFriction=1.2,
            spinningFriction=0.02,
            rollingFriction=0.01,
            physicsClientId=self.client_id,
        )
        self.stairs = build_stairs(
            self.client_id,
            num_steps=self.stair_steps,
            step_height=self.step_height,
        ) if self.task != "run" else []
        self.final_stair = self.stairs[-1] if self.stairs else None
        self._spawn_robot()

    def set_curriculum_stage(self, stair_steps, step_height, init_x, task):
        self.pending_curriculum = (
            int(stair_steps),
            float(step_height),
            float(init_x),
            task,
        )

    def _apply_pending_curriculum(self):
        if self.pending_curriculum is None:
            return
        stair_steps, step_height, init_x, task = self.pending_curriculum
        self.pending_curriculum = None
        if (
            stair_steps == self.stair_steps
            and abs(step_height - self.step_height) < 1e-6
            and abs(init_x - self.init_x) < 1e-6
            and task == self.task
        ):
            return
        self.stair_steps = stair_steps
        self.step_height = step_height
        self.init_x = init_x
        self.task = task
        pb.resetSimulation(physicsClientId=self.client_id)
        pb.setGravity(0, 0, -9.81, physicsClientId=self.client_id)
        pb.setTimeStep(SIM_DT, physicsClientId=self.client_id)
        self.robot = None
        self.stairs = []
        self.final_stair = None
        self._build_world()

    def _spawn_robot(self):
        if self.robot is not None:
            pb.removeBody(self.robot, physicsClientId=self.client_id)
        orn = pb.getQuaternionFromEuler([math.pi / 2, 0, math.pi / 2])
        self.robot = pb.loadURDF(
            "laikago/laikago_toes.urdf",
            [self.init_x, 0, INIT_HEIGHT],
            orn,
            physicsClientId=self.client_id,
        )
        for link_id in range(-1, pb.getNumJoints(self.robot, physicsClientId=self.client_id)):
            pb.changeDynamics(
                self.robot,
                link_id,
                lateralFriction=1.15,
                spinningFriction=0.03,
                rollingFriction=0.01,
                linearDamping=0.02,
                angularDamping=0.02,
                physicsClientId=self.client_id,
            )
        for joint_ids in LEG_JOINTS.values():
            for joint_id, target in zip(joint_ids, [0.0, NOMINAL_THIGH, NOMINAL_CALF]):
                pb.resetJointState(
                    self.robot, joint_id, target, physicsClientId=self.client_id
                )

        for _ in range(120):
            self._apply_targets(np.zeros(12, dtype=np.float32))
            pb.stepSimulation(physicsClientId=self.client_id)

        pos, _ = pb.getBasePositionAndOrientation(self.robot, physicsClientId=self.client_id)
        self.prev_x = pos[0]
        self.best_climb = 0.0
        self.success_timer = 0.0
        self.t = 0.0
        self.prev_action[:] = 0.0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._apply_pending_curriculum()
        self._spawn_robot()
        return self._get_obs(), {}

    def step(self, action):
        action = np.asarray(action, dtype=np.float32)
        action = np.clip(action, -1.0, 1.0)
        sub_steps = int(CTRL_DT / SIM_DT)
        for _ in range(sub_steps):
            self.t += SIM_DT
            self._apply_targets(action)
            pb.stepSimulation(physicsClientId=self.client_id)
            if self.gui:
                time.sleep(SIM_DT)

        obs = self._get_obs()
        reward, terminated, truncated, info = self._reward_done(action)
        self.prev_action = action.copy()
        return obs, reward, terminated, truncated, info

    def _active_gait(self):
        if self.task == "run":
            return GaitConfig(
                freq=0.85,
                lift=0.10,
                duty=0.70,
                stance_thigh=0.63,
                stance_calf=-1.10,
                forward_bias=0.06,
                stride=-0.12,
                calf_lift_gain=1.00,
                body_pitch_bias=0.0,
                hip_sway=0.015,
            )
        return self.gait

    def _apply_targets(self, action):
        residual = action * ACTION_SCALE
        idx = 0
        gait = self._active_gait()
        for leg in LEG_ORDER:
            base = gait_targets(self.t, leg, gait)
            targets = base + residual[idx : idx + 3]
            targets[0] = np.clip(targets[0], -0.24, 0.24)
            targets[1] = np.clip(targets[1], NOMINAL_THIGH - 0.60, NOMINAL_THIGH + 0.75)
            targets[2] = np.clip(targets[2], NOMINAL_CALF - 0.85, NOMINAL_CALF + 0.55)
            for joint_id, target in zip(LEG_JOINTS[leg], targets):
                pb.setJointMotorControl2(
                    self.robot,
                    joint_id,
                    pb.POSITION_CONTROL,
                    targetPosition=float(target),
                    force=180,
                    positionGain=0.95,
                    velocityGain=0.55,
                    physicsClientId=self.client_id,
                )
            idx += 3

    def _get_obs(self):
        pos, orn = pb.getBasePositionAndOrientation(self.robot, physicsClientId=self.client_id)
        lin_vel, ang_vel = pb.getBaseVelocity(self.robot, physicsClientId=self.client_id)
        rot = pb.getMatrixFromQuaternion(orn)
        body_up_z = rot[7]
        forward_dist = pos[0]
        climb = max(0.0, pos[2] - INIT_HEIGHT)
        phase = (self.t * self.gait.freq) % 1.0

        joint_pos = []
        joint_vel = []
        for leg in LEG_ORDER:
            for joint_id in LEG_JOINTS[leg]:
                js = pb.getJointState(self.robot, joint_id, physicsClientId=self.client_id)
                joint_pos.append(js[0])
                joint_vel.append(0.1 * js[1])

        toe_bits = []
        for toe in TOE_LINK_IDS:
            contacts = pb.getContactPoints(
                bodyA=self.robot,
                linkIndexA=toe,
                physicsClientId=self.client_id,
            )
            toe_bits.append(1.0 if contacts else 0.0)

        obs = np.array(
            [
                forward_dist,
                climb,
                pos[1],
                body_up_z,
                *lin_vel,
                *ang_vel,
                math.sin(2 * math.pi * phase),
                math.cos(2 * math.pi * phase),
                self.stair_top_x - pos[0],
                self.stair_top_height - climb,
                *joint_pos,
                *joint_vel,
                *toe_bits,
                *self.prev_action,
            ],
            dtype=np.float32,
        )
        return obs

    def _reward_done(self, action):
        pos, orn = pb.getBasePositionAndOrientation(self.robot, physicsClientId=self.client_id)
        lin_vel, ang_vel = pb.getBaseVelocity(self.robot, physicsClientId=self.client_id)
        climb = max(0.0, pos[2] - INIT_HEIGHT)
        delta_x = pos[0] - self.prev_x
        climb_gain = max(0.0, climb - self.best_climb)
        self.prev_x = pos[0]
        self.best_climb = max(self.best_climb, climb)

        rot = pb.getMatrixFromQuaternion(orn)
        body_up_z = rot[7]
        tilt = 1.0 - body_up_z
        toe_contacts = self._count_final_tread_toe_contacts()
        stair_toe_contacts = self._count_stair_toe_contacts()
        bad_body_contact = self._has_non_toe_support_contact()
        speed = math.sqrt(sum(v * v for v in lin_vel))
        spin = math.sqrt(sum(v * v for v in ang_vel))
        first_step_x = STAIR_X_START + STAIR_STEP_DEPTH / 2
        approach_progress = min(max(pos[0] / max(first_step_x, 1e-6), 0.0), 1.0)

        if self.task == "run":
            forward_speed = max(0.0, lin_vel[0])
            backward_speed = max(0.0, -lin_vel[0])
            progress = pos[0] - self.init_x
            lateral_drift = abs(pos[1])
            upright = body_up_z > 0.88
            centered = lateral_drift < 0.35
            stable_forward = max(progress, 0.0) if upright and centered else 0.0
            reward = 0.05
            reward += 20.0 * max(delta_x, -0.03)
            reward += 4.0 * min(forward_speed, 1.2) if upright else -2.0 * forward_speed
            reward += 8.0 * stable_forward
            reward += 2.0 * min(stable_forward, 1.0) ** 2
            reward += 5.0 * body_up_z
            reward -= 5.0 * backward_speed
            reward -= 2.0 * max(-progress, 0.0)
            reward -= 4.0 * tilt
            reward -= 8.0 * max(0.0, 0.82 - body_up_z)
            reward -= 1.0 * lateral_drift
            if progress > 0.5 and upright and centered:
                reward += 1.0
            if progress > 1.0 and upright and centered:
                reward += 2.0
            if progress > 1.5 and upright and centered:
                reward += 3.0
            reward -= 0.03 * spin
            reward -= 0.01 * float(np.sum(np.square(action)))
            reward -= 0.01 * float(np.sum(np.square(action - self.prev_action)))

            terminated = False
            truncated = False
            status = "running"
            if pos[2] < 0.18 or body_up_z < 0.65 or abs(pos[1]) > 1.3:
                terminated = True
                status = "fell"
                reward -= 400.0
            elif self.t >= EPISODE_T:
                truncated = True
                status = "timeout"
                reward += 30.0 + 120.0 * stable_forward

            info = {
                "status": status,
                "forward_dist": pos[0],
                "climb": climb,
                "toe_contacts": 0,
                "stair_toe_contacts": 0,
                "body_up_z": body_up_z,
                "success_timer": 0.0,
            }
            return reward, terminated, truncated, info

        reward = 0.02
        reward += 7.0 * max(delta_x, -0.02)
        reward += 45.0 * climb_gain
        reward += 4.0 * min(climb, self.stair_top_height)
        reward += 1.2 * approach_progress
        reward += 0.8 * min(stair_toe_contacts, 4)
        reward -= 1.0 * tilt
        reward -= 0.01 * float(np.sum(np.square(action)))
        reward -= 0.01 * float(np.sum(np.square(action - self.prev_action)))

        stable_on_top = (
            self.stair_top_x - SUCCESS_X_MARGIN <= pos[0] <= self.stair_top_x + 0.08
            and climb >= self.stair_top_height - SUCCESS_HEIGHT_TOL
            and body_up_z > 0.88
            and toe_contacts >= 2
            and not bad_body_contact
            and abs(pos[1]) < 0.35
            and speed < 0.55
            and spin < 1.2
        )
        if stable_on_top:
            self.success_timer += CTRL_DT
            reward += 10.0 + 25.0 * self.success_timer
        else:
            self.success_timer = 0.0

        terminated = False
        truncated = False
        status = "running"
        if self.success_timer >= SUCCESS_HOLD_T:
            terminated = True
            status = "success"
            reward += 300.0
        elif pos[2] < 0.18 or body_up_z < 0.35 or abs(pos[1]) > 1.3:
            terminated = True
            status = "fell"
            reward -= 400.0
        elif self.t >= EPISODE_T:
            truncated = True
            status = "timeout"

        info = {
            "status": status,
            "forward_dist": pos[0],
            "climb": climb,
            "toe_contacts": toe_contacts,
            "stair_toe_contacts": stair_toe_contacts,
            "body_up_z": body_up_z,
            "success_timer": self.success_timer,
        }
        return reward, terminated, truncated, info

    def _count_stair_toe_contacts(self):
        toe_links = set()
        for stair in self.stairs:
            contacts = pb.getContactPoints(
                bodyA=self.robot, bodyB=stair, physicsClientId=self.client_id
            )
            for contact in contacts:
                link_a = contact[3]
                link_b = contact[4]
                normal_force = contact[9]
                if normal_force <= 0.5:
                    continue
                if link_a in TOE_LINK_IDS or link_b in TOE_LINK_IDS:
                    toe_links.add(link_a if link_a in TOE_LINK_IDS else link_b)
        return len(toe_links)

    def _count_final_tread_toe_contacts(self):
        if self.final_stair is None:
            return 0
        contacts = pb.getContactPoints(
            bodyA=self.robot, bodyB=self.final_stair, physicsClientId=self.client_id
        )
        toe_links = set()
        for contact in contacts:
            link_a = contact[3]
            link_b = contact[4]
            normal_force = contact[9]
            if normal_force <= 0.5:
                continue
            if link_a in TOE_LINK_IDS or link_b in TOE_LINK_IDS:
                toe_links.add(link_a if link_a in TOE_LINK_IDS else link_b)
        return len(toe_links)

    def _has_non_toe_support_contact(self):
        contacts = pb.getContactPoints(bodyA=self.robot, physicsClientId=self.client_id)
        for contact in contacts:
            other_body = contact[2]
            link_a = contact[3]
            normal_force = contact[9]
            if other_body == self.robot or normal_force <= 1.0:
                continue
            if link_a not in TOE_LINK_IDS:
                return True
        return False

    def render(self):
        view = pb.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=[1.35, 0, 0.30],
            distance=3.5,
            yaw=40,
            pitch=-25,
            roll=0,
            upAxisIndex=2,
            physicsClientId=self.client_id,
        )
        proj = pb.computeProjectionMatrixFOV(
            45, 960 / 540, 0.1, 20, physicsClientId=self.client_id
        )
        _, _, rgb, _, _ = pb.getCameraImage(
            960,
            540,
            view,
            proj,
            renderer=pb.ER_TINY_RENDERER,
            flags=pb.ER_NO_SEGMENTATION_MASK,
            physicsClientId=self.client_id,
        )
        return np.array(rgb, dtype=np.uint8).reshape(540, 960, 4)[:, :, :3]

    def close(self):
        if pb.isConnected(self.client_id):
            pb.disconnect(self.client_id)


def make_env(
    rank,
    gui=False,
    stair_steps=STAIR_NUM_STEPS,
    step_height=STAIR_STEP_HEIGHT,
    init_x=INIT_X,
    task="stairs",
):
    def _init():
        env = ResidualStairsEnv(
            gui=gui and rank == 0,
            stair_steps=stair_steps,
            step_height=step_height,
            init_x=init_x,
            task=task,
        )
        return env

    return _init


def curriculum_schedule(total_timesteps, task):
    stage_count = 5
    if task == "run":
        stages = [(0, 0.0, 0.0, "run")]
    elif task == "jump":
        stages = [
            (1, 0.015, 0.45, "jump"),
            (1, 0.020, 0.38, "jump"),
            (1, 0.025, 0.32, "jump"),
            (1, 0.030, 0.26, "jump"),
            (1, 0.035, 0.20, "jump"),
        ]
    else:
        stages = [
            (1, 0.020, 0.42, "jump"),
            (1, 0.030, 0.30, "jump"),
            (2, 0.030, 0.20, "stairs"),
            (3, 0.030, 0.10, "stairs"),
            (4, 0.030, 0.00, "stairs"),
        ]
    timesteps_per_stage = max(1, total_timesteps // stage_count)
    return [(i * timesteps_per_stage, *stage) for i, stage in enumerate(stages)]


def train(args):
    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
        from stable_baselines3.common.monitor import Monitor
        from stable_baselines3.common.callbacks import BaseCallback
    except ImportError as exc:  # pragma: no cover - user-facing dependency message
        raise SystemExit(
            "Missing stable-baselines3. Install with: "
            "pip install stable-baselines3 torch gymnasium"
        ) from exc

    stages = curriculum_schedule(args.timesteps, args.task) if args.curriculum else []
    initial_steps = stages[0][1] if stages else args.stair_steps
    initial_height = stages[0][2] if stages else args.step_height
    initial_x = stages[0][3] if stages else args.init_x
    initial_task = stages[0][4] if stages else args.task

    class StairCurriculumCallback(BaseCallback):
        def __init__(self, schedule):
            super().__init__()
            self.schedule = schedule
            self.active_index = -1

        def _on_step(self):
            next_index = self.active_index + 1
            if next_index >= len(self.schedule):
                return True
            start_t, stair_steps, step_height, init_x, task = self.schedule[next_index]
            if self.num_timesteps < start_t:
                return True
            self.training_env.env_method(
                "set_curriculum_stage", stair_steps, step_height, init_x, task
            )
            self.active_index = next_index
            print(
                f"Curriculum stage {next_index + 1}/{len(self.schedule)}: "
                f"task={task}, {stair_steps} steps, "
                f"height={step_height:.3f} m, init_x={init_x:.2f}"
            )
            return True

    def monitored_env(rank):
        def _init():
            return Monitor(
                make_env(
                    rank,
                    stair_steps=initial_steps,
                    step_height=initial_height,
                    init_x=initial_x,
                    task=initial_task,
                )()
            )

        return _init

    vec_cls = SubprocVecEnv if args.num_envs > 1 else DummyVecEnv
    env = vec_cls([monitored_env(i) for i in range(args.num_envs)])
    if args.load_model:
        model = PPO.load(args.load_model, env=env, device=args.device)
        model.verbose = 1
    else:
        model = PPO(
            "MlpPolicy",
            env,
            learning_rate=args.lr,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=args.ent_coef,
            policy_kwargs={"log_std_init": args.log_std_init, "net_arch": [128, 128]},
            device=args.device,
            verbose=1,
            tensorboard_log=args.tensorboard_log or None,
        )
    callback = StairCurriculumCallback(stages) if stages else None
    model.learn(
        total_timesteps=args.timesteps,
        progress_bar=args.progress_bar,
        callback=callback,
    )
    model.save(args.model)
    env.close()
    print(f"Saved PPO residual policy: {args.model}")


def demo(args):
    try:
        from stable_baselines3 import PPO
    except ImportError as exc:  # pragma: no cover
        raise SystemExit("Missing stable-baselines3. Install it before demo.") from exc

    model = PPO.load(args.model)
    env = ResidualStairsEnv(
        gui=args.gui,
        stair_steps=args.stair_steps,
        step_height=args.step_height,
        init_x=args.init_x,
        task=args.task,
    )
    obs, _ = env.reset()
    frames = []
    try:
        for step in range(args.steps):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            if args.record:
                frames.append(env.render())
            if step % 25 == 0:
                print(
                    f"step={step:4d} status={info['status']} "
                    f"x={info['forward_dist']:.2f} climb={info['climb']:.2f} "
                    f"toes={info['toe_contacts']}"
                )
            if terminated or truncated:
                print("episode ended:", info)
                break
    finally:
        env.close()

    if args.record and frames:
        import cv2

        h, w = frames[0].shape[:2]
        writer = cv2.VideoWriter(
            args.record, cv2.VideoWriter_fourcc(*"mp4v"), 25, (w, h)
        )
        for frame in frames:
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        writer.release()
        print(f"Saved demo video: {args.record}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)

    train_p = sub.add_parser("train")
    train_p.add_argument("--timesteps", type=int, default=1_000_000)
    train_p.add_argument("--num_envs", type=int, default=8)
    train_p.add_argument("--model", default="ppo_residual_stairs.zip")
    train_p.add_argument("--load_model", default="")
    train_p.add_argument("--task", choices=["run", "jump", "stairs"], default="stairs")
    train_p.add_argument("--lr", type=float, default=3e-4)
    train_p.add_argument("--n_steps", type=int, default=1024)
    train_p.add_argument("--batch_size", type=int, default=2048)
    train_p.add_argument("--tensorboard_log", default="")
    train_p.add_argument("--progress_bar", action="store_true")
    train_p.add_argument("--device", default="cpu", choices=["cpu", "cuda", "auto"])
    train_p.add_argument("--log_std_init", type=float, default=-2.0)
    train_p.add_argument("--ent_coef", type=float, default=0.0)
    train_p.add_argument("--stair_steps", type=int, default=STAIR_NUM_STEPS)
    train_p.add_argument("--step_height", type=float, default=STAIR_STEP_HEIGHT)
    train_p.add_argument("--init_x", type=float, default=INIT_X)
    train_p.add_argument(
        "--curriculum",
        action="store_true",
        help="Train from easier stairs first: 1 low step -> full 4-step stairs.",
    )

    demo_p = sub.add_parser("demo")
    demo_p.add_argument("--model", default="ppo_residual_stairs.zip")
    demo_p.add_argument("--task", choices=["run", "jump", "stairs"], default="stairs")
    demo_p.add_argument("--stair_steps", type=int, default=STAIR_NUM_STEPS)
    demo_p.add_argument("--step_height", type=float, default=STAIR_STEP_HEIGHT)
    demo_p.add_argument("--init_x", type=float, default=INIT_X)
    demo_p.add_argument("--steps", type=int, default=500)
    demo_p.add_argument("--gui", action="store_true")
    demo_p.add_argument("--record", default=None)

    args = parser.parse_args()
    if args.mode == "train":
        train(args)
    elif args.mode == "demo":
        demo(args)


if __name__ == "__main__":
    main()

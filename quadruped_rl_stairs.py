"""
四足机器人爬楼梯强化学习示例（多机器人并排训练 + CMA-ES）
======================================================

教学版：用进化策略 (CMA-ES) 优化参数化步态，
让一群四足机器人**同时**学习爬上楼梯。

特点：
- ✅ 完全 CPU 可跑（不需要 GPU）
- ✅ 一群机器狗并排显示，视觉震撼
- ✅ 每个机器人独立 PyBullet 仿真，物理隔离
- ✅ 多进程并行（--num_envs N）加速训练
- ✅ 训练全过程录 GIF，呈现"从摔倒到爬上"的演化
- ✅ 训练 50 代 ~ 1-3 分钟

依赖：
    pip install pybullet numpy cma matplotlib imageio

用法：
    # 单进程训练
    python quadruped_rl_stairs.py train --generations 50

    # 6 个并行 worker
    python quadruped_rl_stairs.py train --generations 50 --num_envs 6

    # 同时录训练过程 GIF（一群机器狗）
    python quadruped_rl_stairs.py train --generations 50 --num_envs 6 \\
        --record_progress training.gif

    # 用训练好的参数演示一群机器狗爬楼梯
    python quadruped_rl_stairs.py demo --num_robots 9 --record swarm_demo.gif

关于 GPU：
    PyBullet 物理引擎不支持 GPU。要用 GPU 加速大规模并行，
    请切换到 NVIDIA Isaac Lab / MuJoCo MJX / Google Brax。
"""

import os
import math
import time
import json
import argparse
import multiprocessing as mp
import numpy as np
import pybullet as p
import pybullet_data


# ============================================================
# 1. 楼梯场景
# ============================================================

STAIR_STEP_HEIGHT = 0.05    # 每级 5 cm
STAIR_STEP_DEPTH = 0.40     # 每级深 40 cm
STAIR_NUM_STEPS = 4         # 共 4 级
STAIR_X_START = 0.6         # 楼梯起点距机器人 60cm（近一些）
STAIR_WIDTH = 5.0           # 楼梯左右宽 5 m（容纳 12+ 机器人）
STAIR_TOP_X = STAIR_X_START + STAIR_NUM_STEPS * STAIR_STEP_DEPTH
STAIR_TOP_HEIGHT = STAIR_NUM_STEPS * STAIR_STEP_HEIGHT
SUCCESS_HOLD_T = 0.40       # 必须稳定站在最后一级至少 0.4 秒
SUCCESS_X_MARGIN = 0.12     # base 到最后一级靠后区域，才算完整爬完
SUCCESS_HEIGHT_TOL = 0.03   # 身体爬升高度允许 3cm 误差
TOE_LINK_IDS = [3, 7, 11, 15]


GAIT_PARAM_NAMES = [
    'freq', 'lift', 'duty', 'stance_thigh', 'stance_calf',
    'phase_lf', 'phase_rf', 'phase_lh', 'phase_rh',
    'forward_bias', 'stride', 'calf_lift_gain', 'body_pitch_bias',
    'hip_sway',
]
DEFAULT_GAIT_PARAMS = np.array([
    1.40, 0.24, 0.65, 0.67, -1.18,
    0.00, 0.50, 0.50, 0.00,
    0.08, 0.24, 2.00, -0.03, 0.035,
], dtype=float)


def build_stairs(client_id, num_steps=STAIR_NUM_STEPS,
                 step_h=STAIR_STEP_HEIGHT, step_d=STAIR_STEP_DEPTH,
                 width=STAIR_WIDTH, x_start=STAIR_X_START):
    """用一堆 box 拼成楼梯（彩虹渐变色，对比鲜明）"""
    # 渐变色：从橙红 → 黄 → 绿 → 蓝 → 紫
    palette = [
        [0.95, 0.40, 0.30, 1.0],  # 橙红
        [0.98, 0.75, 0.20, 1.0],  # 金黄
        [0.55, 0.85, 0.35, 1.0],  # 草绿
        [0.30, 0.65, 0.90, 1.0],  # 天蓝
        [0.65, 0.45, 0.90, 1.0],  # 淡紫
        [0.95, 0.55, 0.75, 1.0],  # 樱粉
        [0.45, 0.85, 0.85, 1.0],  # 青色
    ]
    stairs = []
    for i in range(num_steps):
        half_size = [step_d / 2, width / 2, step_h * (i + 1) / 2]
        col = p.createCollisionShape(p.GEOM_BOX, halfExtents=half_size,
                                     physicsClientId=client_id)
        color = palette[i % len(palette)]
        vis = p.createVisualShape(p.GEOM_BOX, halfExtents=half_size,
                                  rgbaColor=color,
                                  physicsClientId=client_id)
        pos = [x_start + step_d / 2 + i * step_d, 0, step_h * (i + 1) / 2]
        body = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=col,
            baseVisualShapeIndex=vis,
            basePosition=pos,
            physicsClientId=client_id,
        )
        p.changeDynamics(body, -1, lateralFriction=1.3, spinningFriction=0.03,
                         rollingFriction=0.01, physicsClientId=client_id)
        stairs.append(body)

    # 起点标线（深绿）
    start_line_half = [0.02, width / 2, 0.005]
    sl_col = p.createCollisionShape(p.GEOM_BOX, halfExtents=start_line_half,
                                     physicsClientId=client_id)
    sl_vis = p.createVisualShape(p.GEOM_BOX, halfExtents=start_line_half,
                                  rgbaColor=[0.1, 0.5, 0.1, 1.0],
                                  physicsClientId=client_id)
    p.createMultiBody(baseMass=0, baseCollisionShapeIndex=sl_col,
                      baseVisualShapeIndex=sl_vis,
                      basePosition=[0, 0, 0.005],
                      physicsClientId=client_id)
    return stairs


def _smoothstep(s):
    """0..1 平滑插值，减少腿部目标角突变。"""
    s = min(1.0, max(0.0, s))
    return s * s * (3.0 - 2.0 * s)


def decode_gait_params(params):
    """兼容旧 10 维日志，同时支持新的 14 维优雅爬楼参数。"""
    decoded = DEFAULT_GAIT_PARAMS.copy()
    params = np.asarray(params, dtype=float)
    decoded[:min(len(params), len(decoded))] = params[:len(decoded)]
    return decoded


def gait_targets(params, t, leg_name, nominal_thigh, nominal_calf):
    """根据参数化 trot 生成单条腿的 hip/thigh/calf 目标角。"""
    (
        freq, lift, duty, stance_thigh, stance_calf,
        phase_lf, phase_rf, phase_lh, phase_rh,
        forward_bias, stride, calf_lift_gain, body_pitch_bias, hip_sway,
    ) = decode_gait_params(params)

    leg_phases = {'LF': phase_lf, 'RF': phase_rf, 'LH': phase_lh, 'RH': phase_rh}
    phase = leg_phases[leg_name]
    phi = (t * freq + phase) % 1.0

    if phi < duty:
        s = _smoothstep(phi / max(duty, 1e-3))
        # 支撑相：Laikago 此姿态下 thigh 增大时会产生向前推进。
        thigh_sway = stride * (s - 0.5)
        foot_lift = 0.0
    else:
        s = _smoothstep((phi - duty) / max(1.0 - duty, 1e-3))
        # 摆动相：先收腿抬高，再向前迈到下一阶台阶。
        foot_lift = lift * math.sin(math.pi * s)
        thigh_sway = stride * (0.5 - s)

    is_left = 1.0 if leg_name in ('LF', 'LH') else -1.0
    is_front = 1.0 if leg_name in ('LF', 'RF') else -1.0
    hip = is_left * hip_sway * math.sin(2.0 * math.pi * phi)
    thigh = (
        stance_thigh + thigh_sway + forward_bias + body_pitch_bias * is_front
        + 0.85 * foot_lift
    )
    calf = stance_calf - calf_lift_gain * foot_lift

    thigh = float(np.clip(thigh, nominal_thigh - 0.45, nominal_thigh + 0.55))
    calf = float(np.clip(calf, nominal_calf - 0.45, nominal_calf + 0.35))
    hip = float(np.clip(hip, -0.18, 0.18))
    return hip, thigh, calf


# ============================================================
# 2. 单个机器人环境（用于训练时的物理仿真）
# ============================================================

class StairsEnv:
    """单个机器人在楼梯场景上"""

    SIM_DT = 1.0 / 240.0
    CTRL_DT = 1.0 / 50.0
    EPISODE_T = 6.0
    # Laikago nominal 站立高度约 0.40m（thigh=0.65, calf=-1.20）
    # 略高一点点让机器人轻轻落地，避免穿模
    INIT_HEIGHT = 0.42
    NOMINAL_THIGH = 0.65
    NOMINAL_CALF = -1.20

    def __init__(self, gui=False):
        self.gui = gui
        self.client_id = p.connect(p.GUI if gui else p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath(),
                                  physicsClientId=self.client_id)
        p.setGravity(0, 0, -9.81, physicsClientId=self.client_id)
        p.setTimeStep(self.SIM_DT, physicsClientId=self.client_id)
        plane = p.loadURDF('plane.urdf', physicsClientId=self.client_id)
        p.changeDynamics(plane, -1, lateralFriction=1.2, spinningFriction=0.02,
                         rollingFriction=0.01, physicsClientId=self.client_id)
        self.stairs = build_stairs(self.client_id)
        self.final_stair = self.stairs[-1]
        self.robot = None
        self.leg_joints = {
            'RF': [0, 1, 2], 'LF': [4, 5, 6],
            'RH': [8, 9, 10], 'LH': [12, 13, 14],
        }
        self._spawn_robot()

    def _spawn_robot(self, y_offset=0.0):
        if self.robot is not None:
            p.removeBody(self.robot, physicsClientId=self.client_id)
        start_pos = [0, y_offset, self.INIT_HEIGHT]
        start_orn = p.getQuaternionFromEuler([math.pi / 2, 0, math.pi / 2])
        self.robot = p.loadURDF('laikago/laikago_toes.urdf',
                                start_pos, start_orn,
                                physicsClientId=self.client_id)
        for link_id in range(-1, p.getNumJoints(self.robot, physicsClientId=self.client_id)):
            p.changeDynamics(self.robot, link_id, lateralFriction=1.1,
                             spinningFriction=0.03, rollingFriction=0.01,
                             linearDamping=0.02, angularDamping=0.02,
                             physicsClientId=self.client_id)
        for joint_ids in self.leg_joints.values():
            for joint_id, target in zip(joint_ids,
                                        [0.0, self.NOMINAL_THIGH, self.NOMINAL_CALF]):
                p.resetJointState(self.robot, joint_id, target,
                                  physicsClientId=self.client_id)
        # 稳定姿态
        for _ in range(100):
            for joint_ids in self.leg_joints.values():
                for joint_id, target in zip(joint_ids,
                                            [0.0, self.NOMINAL_THIGH, self.NOMINAL_CALF]):
                    p.setJointMotorControl2(
                        self.robot, joint_id, p.POSITION_CONTROL,
                        targetPosition=target, force=120,
                        positionGain=1.0, velocityGain=0.5,
                        physicsClientId=self.client_id,
                    )
            p.stepSimulation(physicsClientId=self.client_id)
        pos, _ = p.getBasePositionAndOrientation(self.robot,
                                                  physicsClientId=self.client_id)
        self.start_x = pos[0]
        self.start_z = pos[2]
        self.prev_x = pos[0]
        self.prev_z = pos[2]
        self.best_x = pos[0]
        self.best_climb = 0.0
        self.best_step = 0
        self.success_timer = 0.0
        self.ever_on_final_tread = False
        self.t = 0.0

    def reset(self):
        self._spawn_robot()

    def step(self, params):
        sub_steps = int(self.CTRL_DT / self.SIM_DT)
        for _ in range(sub_steps):
            self.t += self.SIM_DT
            for leg_name, joint_ids in self.leg_joints.items():
                hip, thigh, calf = gait_targets(
                    params, self.t, leg_name, self.NOMINAL_THIGH, self.NOMINAL_CALF
                )
                for joint_id, target in zip(joint_ids, [hip, thigh, calf]):
                    p.setJointMotorControl2(
                        self.robot, joint_id, p.POSITION_CONTROL,
                        targetPosition=target, force=170,
                        positionGain=0.95, velocityGain=0.55,
                        physicsClientId=self.client_id,
                    )
            p.stepSimulation(physicsClientId=self.client_id)
            if self.gui:
                time.sleep(self.SIM_DT)
        reward, done, info = self._compute_reward()
        return reward, done, info

    def _compute_reward(self):
        pos, orn = p.getBasePositionAndOrientation(self.robot,
                                                    physicsClientId=self.client_id)
        forward_dist = pos[0] - self.start_x
        climbed_height = max(0, pos[2] - self.start_z)
        delta_x = pos[0] - self.prev_x
        climb_gain = max(0.0, climbed_height - self.best_climb)
        self.best_climb = max(self.best_climb, climbed_height)
        self.prev_x = pos[0]
        self.prev_z = pos[2]
        self.best_x = max(self.best_x, pos[0])

        spatial_step = int(np.clip(
            math.floor((pos[0] - STAIR_X_START) / STAIR_STEP_DEPTH) + 1,
            0, STAIR_NUM_STEPS,
        ))
        expected_climb = spatial_step * STAIR_STEP_HEIGHT
        if pos[0] < STAIR_X_START:
            expected_climb = 0.0
        stable_step = spatial_step
        while stable_step > 0 and climbed_height < stable_step * STAIR_STEP_HEIGHT - 0.035:
            stable_step -= 1

        # 用 base 的"上方向"判断翻车（更准确，不依赖 Euler）
        rot_mat = p.getMatrixFromQuaternion(orn)
        # rot_mat 是 3x3 旋转矩阵的 flatten，base 局部 +Y 在世界的方向就是机身上方
        # （因为初始 orn = [π/2, 0, π/2]，机器人 y 轴朝上）
        body_up_world_z = rot_mat[7]  # 第 7 个元素 = Y-axis 在世界 Z 上的分量
        tilt = 1.0 - body_up_world_z  # 0 = 直立，2 = 完全倒立
        lateral_drift = abs(pos[1])
        backward_slip = max(0.0, self.best_x - pos[0] - 0.04)
        toe_contacts = self._count_final_tread_toe_contacts()
        bad_body_contact = self._has_non_toe_support_contact()
        lin_vel, ang_vel = p.getBaseVelocity(self.robot, physicsClientId=self.client_id)
        speed = math.sqrt(sum(v * v for v in lin_vel))
        spin = math.sqrt(sum(v * v for v in ang_vel))

        # 奖励采用“增量进度 + 阶梯课程”的形式，减少靠累计距离刷分的情况。
        reward = 0.03                         # 活着且保持控制的微小奖励
        reward += 8.0 * max(delta_x, -0.02)    # 稳定向前
        reward += 45.0 * climb_gain            # 只奖励新的最高爬升，避免原地蹦跳刷分
        reward += 2.2 * stable_step            # 必须身体高度跟上，才算踏上该阶
        reward += 4.0 * min(climbed_height, STAIR_TOP_HEIGHT)
        reward -= 2.5 * tilt
        reward -= 0.9 * lateral_drift
        reward -= 3.0 * backward_slip

        # 到台阶附近但身体高度不够，说明在撞台阶/拖脚，降低得分。
        climb_deficit = max(0.0, expected_climb - climbed_height - 0.03)
        reward -= 12.0 * climb_deficit

        done = False
        on_final_tread = (
            stable_step == STAIR_NUM_STEPS
            and STAIR_TOP_X - SUCCESS_X_MARGIN <= pos[0] <= STAIR_TOP_X + 0.08
            and climbed_height >= STAIR_TOP_HEIGHT - SUCCESS_HEIGHT_TOL
            and body_up_world_z > 0.88
            and toe_contacts >= 2
            and not bad_body_contact
            and lateral_drift < 0.35
            and speed < 0.55
            and spin < 1.2
        )
        if on_final_tread:
            self.ever_on_final_tread = True
            self.success_timer += self.CTRL_DT
            reward += 8.0 + 20.0 * self.success_timer
        else:
            self.success_timer = 0.0

        info = {'forward_dist': forward_dist, 'climbed_height': climbed_height,
                'height': pos[2], 'tilt': tilt, 'step_idx': stable_step,
                'spatial_step': spatial_step,
                'body_up_world_z': body_up_world_z,
                'toe_contacts': toe_contacts,
                'bad_body_contact': bad_body_contact,
                'on_final_tread': on_final_tread,
                'success_timer': self.success_timer}
        success = self.success_timer >= SUCCESS_HOLD_T
        if success:
            done = True
            reward += 250.0
            info['terminated'] = 'success'
        # 翻车 = 上方向偏离 > 60°（cos60°=0.5，tilt=0.5）
        if not done and (pos[2] < 0.20 or tilt > 0.7 or abs(pos[1]) > 1.2):
            done = True
            reward -= 3000.0 + 80.0 * tilt
            if self.ever_on_final_tread:
                reward -= 2000.0
            if pos[0] > STAIR_TOP_X + 0.08:
                reward -= 1000.0
            info['terminated'] = 'fell'
        if not done and self.t >= self.EPISODE_T:
            done = True
            if self.ever_on_final_tread and self.success_timer < SUCCESS_HOLD_T:
                reward -= 1200.0
            info['terminated'] = 'timeout'
        return reward, done, info

    def _count_final_tread_toe_contacts(self):
        """成功必须是脚趾真实踩在最后一级，而不是身体或腿翻上去。"""
        contacts = p.getContactPoints(
            bodyA=self.robot, bodyB=self.final_stair,
            physicsClientId=self.client_id,
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
        """最终判定时禁止身体/大腿/小腿把机器人撑在台阶或地面上。"""
        contacts = p.getContactPoints(bodyA=self.robot, physicsClientId=self.client_id)
        for contact in contacts:
            other_body = contact[2]
            link_a = contact[3]
            normal_force = contact[9]
            if other_body == self.robot or normal_force <= 1.0:
                continue
            if link_a not in TOE_LINK_IDS:
                return True
        return False

    def close(self):
        if p.isConnected(self.client_id):
            p.disconnect(self.client_id)


# ============================================================
# 3. Swarm 渲染环境：一群机器人在同一仿真中（仅用于演示和录 GIF）
# ============================================================

class SwarmStairsEnv:
    """多个机器人并排在同一物理仿真中爬楼梯（用于可视化）"""

    SIM_DT = 1.0 / 240.0
    CTRL_DT = 1.0 / 50.0
    EPISODE_T = 6.0
    NOMINAL_THIGH = 0.65
    NOMINAL_CALF = -1.20

    def __init__(self, num_robots=5, y_spacing=0.6):
        self.client_id = p.connect(p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath(),
                                  physicsClientId=self.client_id)
        p.setGravity(0, 0, -9.81, physicsClientId=self.client_id)
        p.setTimeStep(self.SIM_DT, physicsClientId=self.client_id)
        plane = p.loadURDF('plane.urdf', physicsClientId=self.client_id)
        p.changeDynamics(plane, -1, lateralFriction=1.2, spinningFriction=0.02,
                         rollingFriction=0.01, physicsClientId=self.client_id)
        # 楼梯加宽以容纳多个机器人
        build_stairs(self.client_id, width=max(2.0, num_robots * y_spacing + 1.0))

        self.num_robots = num_robots
        self.y_spacing = y_spacing
        self.robots = []
        self.leg_joints = {
            'RF': [0, 1, 2], 'LF': [4, 5, 6],
            'RH': [8, 9, 10], 'LH': [12, 13, 14],
        }
        # 机器人左右排开（用 nominal 高度 0.42 让机器狗稳稳着地）
        y0 = -(num_robots - 1) * y_spacing / 2
        for i in range(num_robots):
            y = y0 + i * y_spacing
            r = p.loadURDF('laikago/laikago_toes.urdf',
                           [0, y, 0.42],
                           p.getQuaternionFromEuler([math.pi / 2, 0, math.pi / 2]),
                           physicsClientId=self.client_id)
            self.robots.append(r)
            for link_id in range(-1, p.getNumJoints(r, physicsClientId=self.client_id)):
                p.changeDynamics(r, link_id, lateralFriction=1.1,
                                 spinningFriction=0.03, rollingFriction=0.01,
                                 linearDamping=0.02, angularDamping=0.02,
                                 physicsClientId=self.client_id)
            # 初始关节
            for joint_ids in self.leg_joints.values():
                for jid, target in zip(joint_ids,
                                       [0.0, self.NOMINAL_THIGH, self.NOMINAL_CALF]):
                    p.resetJointState(r, jid, target,
                                      physicsClientId=self.client_id)

        # 稳定
        for _ in range(150):
            for r in self.robots:
                for joint_ids in self.leg_joints.values():
                    for jid, target in zip(joint_ids,
                                           [0.0, self.NOMINAL_THIGH, self.NOMINAL_CALF]):
                        p.setJointMotorControl2(
                            r, jid, p.POSITION_CONTROL,
                            targetPosition=target, force=120,
                            positionGain=1.0, velocityGain=0.5,
                            physicsClientId=self.client_id,
                        )
            p.stepSimulation(physicsClientId=self.client_id)
        self.t = 0.0

    def respawn(self):
        """把所有机器人重置回起点，重新稳定"""
        y0 = -(self.num_robots - 1) * self.y_spacing / 2
        for i, r in enumerate(self.robots):
            y = y0 + i * self.y_spacing
            p.resetBasePositionAndOrientation(
                r, [0, y, 0.42],   # nominal 站立高度
                p.getQuaternionFromEuler([math.pi / 2, 0, math.pi / 2]),
                physicsClientId=self.client_id,
            )
            p.resetBaseVelocity(r, [0, 0, 0], [0, 0, 0],
                                physicsClientId=self.client_id)
            for joint_ids in self.leg_joints.values():
                for jid, target in zip(joint_ids,
                                       [0.0, self.NOMINAL_THIGH, self.NOMINAL_CALF]):
                    p.resetJointState(r, jid, target,
                                      physicsClientId=self.client_id)
        # 充分稳定，让机器狗轻轻坐到地上
        for _ in range(200):
            for r in self.robots:
                for joint_ids in self.leg_joints.values():
                    for jid, target in zip(joint_ids,
                                           [0.0, self.NOMINAL_THIGH, self.NOMINAL_CALF]):
                        p.setJointMotorControl2(
                            r, jid, p.POSITION_CONTROL,
                            targetPosition=target, force=120,
                            positionGain=1.0, velocityGain=0.5,
                            physicsClientId=self.client_id,
                        )
            p.stepSimulation(physicsClientId=self.client_id)
        self.t = 0.0

    def step_all(self, params_list):
        """每个机器人用各自的参数走一步"""
        sub_steps = int(self.CTRL_DT / self.SIM_DT)
        for _ in range(sub_steps):
            self.t += self.SIM_DT
            for r_idx, (robot, params) in enumerate(zip(self.robots, params_list)):
                for leg_name, joint_ids in self.leg_joints.items():
                    hip, thigh, calf = gait_targets(
                        params, self.t, leg_name, self.NOMINAL_THIGH, self.NOMINAL_CALF
                    )
                    for joint_id, target in zip(joint_ids, [hip, thigh, calf]):
                        p.setJointMotorControl2(
                            robot, joint_id, p.POSITION_CONTROL,
                            targetPosition=target, force=170,
                            positionGain=0.95, velocityGain=0.55,
                            physicsClientId=self.client_id,
                        )
            p.stepSimulation(physicsClientId=self.client_id)

    def render(self, width=720, height=400):
        """渲染当前 swarm 状态（电影感视角）"""
        p.configureDebugVisualizer(
            p.COV_ENABLE_SHADOWS, 1, physicsClientId=self.client_id
        )

        # 视角：从机器人左后方斜俯视，能看清完整楼梯和到顶动作
        view = p.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=[1.35, 0, 0.30],
            distance=3.5,
            yaw=40,                                # 左后方角度
            pitch=-25,                             # 偏俯视
            roll=0,
            upAxisIndex=2,
            physicsClientId=self.client_id,
        )
        proj = p.computeProjectionMatrixFOV(
            45, width / height, 0.1, 20,
            physicsClientId=self.client_id,
        )

        _, _, rgb, _, _ = p.getCameraImage(
            width, height, view, proj,
            lightDirection=[1.5, 1.0, 4.0],        # 斜上方光源
            lightColor=[1.0, 0.95, 0.88],          # 暖色调
            lightAmbientCoeff=0.55,
            lightDiffuseCoeff=0.85,
            lightSpecularCoeff=0.4,
            renderer=p.ER_TINY_RENDERER,
            flags=p.ER_NO_SEGMENTATION_MASK,
            physicsClientId=self.client_id,
        )
        return np.array(rgb, dtype=np.uint8).reshape(height, width, 4)[:, :, :3]

    def get_climbed(self, robot_idx):
        """获取某机器人爬升的高度"""
        pos, _ = p.getBasePositionAndOrientation(self.robots[robot_idx],
                                                  physicsClientId=self.client_id)
        return pos[2], pos[0]

    def close(self):
        if p.isConnected(self.client_id):
            p.disconnect(self.client_id)


# ============================================================
# 4. 策略评估
# ============================================================

def evaluate_policy(env, params, max_steps=300):
    env.reset()
    total_reward = 0.0
    last_dist = 0.0
    last_height = 0.0
    last_step = 0
    terminated = 'timeout'
    for _ in range(max_steps):
        reward, done, info = env.step(params)
        total_reward += reward
        last_dist = info.get('forward_dist', last_dist)
        last_height = info.get('climbed_height', last_height)
        last_step = info.get('step_idx', last_step)
        terminated = info.get('terminated', terminated)
        if done:
            break
    return total_reward, last_dist, last_height, last_step, terminated


_worker_env = None


def _worker_init():
    global _worker_env
    _worker_env = StairsEnv(gui=False)


def _worker_evaluate(params):
    global _worker_env
    return evaluate_policy(_worker_env, params)


def parallel_evaluate(pool, params_list):
    if pool is None:
        global _worker_env
        if _worker_env is None:
            _worker_env = StairsEnv(gui=False)
        return [evaluate_policy(_worker_env, p) for p in params_list]
    return list(pool.map(_worker_evaluate, params_list))


# ============================================================
# 5. CMA-ES 训练
# ============================================================

def train(generations=50, sigma=0.3, popsize=12, num_envs=1,
          log_file='train_log_stairs.json', record_progress=None):
    try:
        import cma
    except ImportError:
        print("❌ 缺少 cma: pip install cma")
        return None, []

    # 参数空间：保守但给足台阶抬脚余量，搜索对象见 GAIT_PARAM_NAMES。
    bounds_low = np.array([
        0.85, 0.12, 0.52, 0.55, -1.35,
        0.0, 0.0, 0.0, 0.0,
        -0.04, 0.16, 1.20, -0.12, 0.00,
    ])
    bounds_high = np.array([
        1.80, 0.42, 0.72, 0.82, -0.95,
        1.0, 1.0, 1.0, 1.0,
        0.22, 0.38, 2.20, 0.06, 0.08,
    ])

    def unnormalize(z):
        z = np.clip(z, -1, 1)
        return bounds_low + (z + 1) / 2 * (bounds_high - bounds_low)

    # 初始：稳定 diagonal trot。新参数让机器人更像“抬脚上台阶”，而不是撞台阶。
    init_params = DEFAULT_GAIT_PARAMS.copy()
    z0 = (init_params - bounds_low) / (bounds_high - bounds_low) * 2 - 1

    es = cma.CMAEvolutionStrategy(
        z0.tolist(), sigma,
        {'popsize': popsize, 'verbose': -9,
         'bounds': [[-1] * len(init_params), [1] * len(init_params)]}
    )

    pool = None
    if num_envs > 1:
        ctx = mp.get_context('spawn')
        pool = ctx.Pool(num_envs, initializer=_worker_init)
        print(f"🚀 {num_envs} 个 worker 并行训练")

    # 录制 GIF 的多机器人环境（与 popsize 同样数量 → 整代机器狗一起跑）
    rec_env = None
    rec_frames = []
    if record_progress:
        rec_env = SwarmStairsEnv(num_robots=popsize, y_spacing=0.5)
        print(f"📹 录制训练过程（{popsize} 只机器狗同时上）: {record_progress}")

    history = []
    best_params = None
    best_reward = -np.inf
    elite_z = z0.copy()

    t_start = time.time()
    record_every = max(1, generations // 25)

    try:
        for gen in range(generations):
            solutions = es.ask()
            # 保留当前精英候选，避免一代采样全是摔倒动作时丢掉已知可行步态。
            solutions[0] = elite_z.tolist()
            params_list = [unnormalize(np.array(z)) for z in solutions]
            results = parallel_evaluate(pool, params_list)
            rewards = [-r for r, _, _, _, _ in results]
            dists = [d for _, d, _, _, _ in results]
            heights = [h for _, _, h, _, _ in results]
            steps = [s for _, _, _, s, _ in results]
            terminations = [term for _, _, _, _, term in results]
            successes = [term == 'success' for term in terminations]
            es.tell(solutions, rewards)

            best_idx = int(np.argmin(rewards))
            worst_idx = int(np.argmax(rewards))
            gen_best_reward = -rewards[best_idx]
            gen_best_dist = dists[best_idx]
            gen_best_height = heights[best_idx]
            gen_best_step = steps[best_idx]
            gen_worst_reward = -rewards[worst_idx]
            if gen_best_reward > best_reward:
                best_reward = gen_best_reward
                best_params = params_list[best_idx]
                elite_z = np.array(solutions[best_idx])

            history.append({
                'gen': gen,
                'best_reward': float(gen_best_reward),
                'best_dist': float(gen_best_dist),
                'best_height': float(gen_best_height),
                'best_step': int(gen_best_step),
                'best_terminated': terminations[best_idx],
                'best_params': params_list[best_idx].tolist(),
                'worst_reward': float(gen_worst_reward),
                'worst_dist': float(dists[worst_idx]),
                'worst_height': float(heights[worst_idx]),
                'worst_step': int(steps[worst_idx]),
                'worst_terminated': terminations[worst_idx],
                'worst_params': params_list[worst_idx].tolist(),
                'success_rate': float(np.mean(successes)),
                'mean_reward': float(-np.mean(rewards)),
                'mean_dist': float(np.mean(dists)),
            })
            elapsed = time.time() - t_start
            print(f"Gen {gen+1:3d}/{generations} | "
                  f"best dist = {gen_best_dist:5.2f}m  "
                  f"best climbed = {gen_best_height:.2f}m  "
                  f"step = {gen_best_step}/{STAIR_NUM_STEPS}  "
                  f"success = {np.mean(successes):.0%}  "
                  f"reward = {gen_best_reward:6.2f}  "
                  f"t = {elapsed:.0f}s")

            # 录 GIF：每代用全部 popsize 个候选让所有机器狗同时跑
            # （即使训练失败的也保留 → 学生能看到一群里有的摔、有的爬、有的进步）
            if rec_env is not None and (gen % record_every == 0 or gen == generations - 1):
                # 重新生成一群机器狗（重置位置和姿态）
                rec_env.respawn()
                # 跑 80 步（约 1.6 秒物理时间）让动作充分展开
                num_capture = 4  # 每代采 4 帧形成小动画
                steps_per_capture = 20
                for capture_step in range(num_capture):
                    for _ in range(steps_per_capture):
                        rec_env.step_all(params_list)
                    frame = rec_env.render()
                    rec_frames.append(_add_label(frame, gen + 1, gen_best_height,
                                                  gen_best_dist, popsize))
    finally:
        if pool is not None:
            pool.close()
            pool.join()
        if rec_env is not None:
            for _ in range(15):
                rec_frames.append(rec_frames[-1])
            if rec_frames:
                import imageio.v2 as imageio
                imageio.mimsave(record_progress, rec_frames, fps=8, loop=0)
                print(f"📹 训练 GIF 保存: {record_progress} ({len(rec_frames)} 帧)")
            rec_env.close()

    with open(log_file, 'w') as f:
        json.dump({
            'history': history,
            'best_params': best_params.tolist() if best_params is not None else None,
            'best_reward': float(best_reward),
            'config': {'generations': generations, 'popsize': popsize,
                       'num_envs': num_envs, 'sigma': sigma,
                       'param_names': GAIT_PARAM_NAMES},
        }, f, indent=2)
    print(f"💾 日志: {log_file}")

    return best_params, history


def _add_label(frame, gen, climbed, dist, popsize=None):
    from PIL import Image, ImageDraw, ImageFont
    img = Image.fromarray(frame)
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', 22)
        font_m = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', 14)
        font_s = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 12)
    except OSError:
        font = font_m = font_s = ImageFont.load_default()
    h, w = frame.shape[:2]
    # 顶部半透明白底
    draw.rectangle([(0, 0), (w, 38)], fill=(255, 255, 255, 235))
    draw.text((12, 7), f"Gen {gen}", fill=(20, 30, 80), font=font)
    if popsize:
        draw.text((105, 13), f"({popsize} robots evolving)",
                  fill=(100, 100, 110), font=font_s)
    # 右侧绩效
    text = f"best climb: {climbed:.2f}m   forward: {dist:.2f}m"
    draw.text((w - 320, 12), text, fill=(60, 25, 25), font=font_m)
    return np.array(img)


def _add_highlight_label(frame, title, subtitle):
    from PIL import Image, ImageDraw, ImageFont
    img = Image.fromarray(frame)
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', 22)
        font_s = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 14)
    except OSError:
        font = font_s = ImageFont.load_default()
    h, w = frame.shape[:2]
    draw.rectangle([(0, 0), (w, 58)], fill=(255, 255, 255, 238))
    draw.text((12, 7), title, fill=(20, 30, 80), font=font)
    draw.text((12, 35), subtitle, fill=(70, 70, 80), font=font_s)
    return np.array(img)


def _pick_highlight_entries(history, max_entries=10):
    """从训练历史里挑约 10 个失败、过渡和成功片段，尽量去重。"""
    picks = []
    used = set()

    def add(kind, item, title):
        if item is None or len(picks) >= max_entries:
            return
        key = (kind, item['gen'])
        if key in used:
            return
        used.add(key)
        picks.append((kind, item, title))

    early = history[:min(20, len(history))]
    if early:
        failure = min(
            early,
            key=lambda h: (
                h.get('worst_terminated') != 'fell',
                h.get('worst_step', STAIR_NUM_STEPS),
                h.get('worst_reward', 0.0),
            ),
        )
        add('worst', failure, 'Obvious mistake')

    late_failures = [
        h for h in history[:min(80, len(history))]
        if h.get('worst_terminated') == 'fell'
    ]
    if late_failures:
        add('worst', min(late_failures, key=lambda h: h.get('worst_reward', 0.0)),
            'Another failed candidate')

    for target_step in (1, 2, 3, 4):
        candidates = [
            h for h in history
            if h.get('best_step', 0) >= target_step
            and h.get('best_terminated') != 'success'
        ]
        if candidates:
            add('best', candidates[0], f'Learning progress: step {target_step}')

    successes = [h for h in history if h.get('best_terminated') == 'success']
    if successes:
        add('best', successes[0], 'Seed policy success')
        trained = next((h for h in successes if h['gen'] >= 20), None)
        add('best', trained, 'Trained policy reaches the top')

    if history:
        best = max(history, key=lambda h: h.get('best_reward', -np.inf))
        add('best', best, 'Best policy after training')
        add('best', history[-1], 'Final generation: complete climb')

    high_success = sorted(
        history, key=lambda h: (h.get('success_rate', 0.0), h.get('best_reward', 0.0)),
        reverse=True,
    )
    for item in high_success:
        if item.get('success_rate', 0.0) > 0:
            add('best', item, 'Higher population success rate')

    if history and len(picks) < max_entries:
        spread = np.linspace(0, len(history) - 1, max_entries, dtype=int)
        for idx in spread:
            add('best', history[int(idx)], 'Training timeline sample')

    return picks


def render_highlight_video(log_file, output_path, width=720, height=400,
                           seconds_per_clip=6.0, fps=25, max_clips=10):
    """根据训练日志挑选失败和成功片段，重放并保存为 MP4/GIF。"""
    with open(log_file) as f:
        data = json.load(f)
    history = data.get('history', [])
    picks = _pick_highlight_entries(history, max_entries=max_clips)
    if not picks:
        print("⚠️  没有可用于剪辑的训练历史")
        return []

    import imageio.v2 as imageio
    frames = []
    steps_per_clip = int(seconds_per_clip / SwarmStairsEnv.CTRL_DT)
    capture_every = max(1, int(1.0 / (fps * SwarmStairsEnv.CTRL_DT)))

    env = SwarmStairsEnv(num_robots=1)
    try:
        for clip_idx, (kind, item, title) in enumerate(picks, start=1):
            params = np.array(item[f'{kind}_params'])
            env.respawn()
            for step in range(steps_per_clip):
                env.step_all([params])
                if step % capture_every == 0:
                    climbed = item.get(f'{kind}_height', 0.0)
                    dist = item.get(f'{kind}_dist', 0.0)
                    terminated = item.get(f'{kind}_terminated', 'unknown')
                    subtitle = (
                        f"Gen {item['gen'] + 1} | step {item.get(f'{kind}_step', 0)}/"
                        f"{STAIR_NUM_STEPS} | climb {climbed:.2f}m | "
                        f"forward {dist:.2f}m | {terminated}"
                    )
                    frames.append(_add_highlight_label(
                        env.render(width=width, height=height),
                        f"Case {clip_idx}/{len(picks)}: {title}", subtitle
                    ))
            for _ in range(fps // 2):
                frames.append(frames[-1])
    finally:
        env.close()

    if output_path.lower().endswith('.gif'):
        imageio.mimsave(output_path, frames, fps=fps, loop=0)
    else:
        try:
            imageio.mimsave(output_path, frames, fps=fps, quality=8)
        except ValueError:
            try:
                import cv2
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                h, w = frames[0].shape[:2]
                writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
                if not writer.isOpened():
                    raise RuntimeError('cv2 VideoWriter failed to open')
                for frame in frames:
                    writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                writer.release()
            except Exception:
                fallback = os.path.splitext(output_path)[0] + '.gif'
                imageio.mimsave(fallback, frames, fps=fps, loop=0)
                output_path = fallback
    print(f"🎞️  高光视频保存: {output_path} ({len(frames)} 帧)")
    return picks


# ============================================================
# 6. Demo（一群机器狗）
# ============================================================

def demo_swarm(params, num_robots=9, record_path=None):
    """让 N 个机器人用同一组参数（或加点噪声）爬楼梯，可录 GIF"""
    env = SwarmStairsEnv(num_robots=num_robots)
    params = decode_gait_params(params)
    # 给每个机器人加点小噪声让动作不完全同步（更有趣）
    rng = np.random.default_rng(42)
    params_list = []
    for i in range(num_robots):
        noise = rng.normal(0, 0.035, size=len(DEFAULT_GAIT_PARAMS))
        noise[5:9] = rng.normal(0, 0.1, size=4)  # 相位噪声大点
        noise[10:] *= 0.5
        p_i = params + noise
        params_list.append(p_i)

    frames = []
    print(f"🎬 演示 {num_robots} 只机器狗爬楼梯...")
    for step in range(200):
        env.step_all(params_list)
        if record_path:
            frames.append(env.render())
        if step % 50 == 0:
            heights = [env.get_climbed(i)[0] for i in range(num_robots)]
            print(f"  step {step}: 机器狗高度 {np.mean(heights):.2f}m (max {max(heights):.2f}m)")

    if record_path and frames:
        import imageio.v2 as imageio
        imageio.mimsave(record_path, frames, fps=25, loop=0)
        print(f"📹 演示 GIF: {record_path}")
    env.close()


# ============================================================
# 7. CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__)
    parser.add_argument('mode', choices=['train', 'demo'])
    parser.add_argument('--generations', type=int, default=50)
    parser.add_argument('--popsize', type=int, default=12)
    parser.add_argument('--sigma', type=float, default=0.25,
                        help='CMA-ES 初始探索幅度')
    parser.add_argument('--num_envs', type=int, default=1,
                        help='并行 worker 数（建议 = CPU 核数）')
    parser.add_argument('--num_robots', type=int, default=9,
                        help='演示时的机器狗数量')
    parser.add_argument('--gpu', action='store_true',
                        help='GPU 加速（PyBullet 不支持，提示替代方案）')
    parser.add_argument('--params_file', default='train_log_stairs.json')
    parser.add_argument('--record', help='演示 GIF 输出路径')
    parser.add_argument('--record_progress', help='训练过程 GIF 路径')
    parser.add_argument('--record_highlights',
                        help='训练后自动剪辑失败/进步/成功片段视频，如 highlights.mp4')
    args = parser.parse_args()

    if args.gpu:
        print("⚠️  PyBullet 不支持 GPU。要真正用 GPU 大规模并行：")
        print("    - NVIDIA Isaac Lab / Isaac Gym (需 RTX 显卡)")
        print("    - MuJoCo MJX 或 Brax (JAX, 支持 GPU/TPU)")
        print("    本示例用 CPU 多进程 --num_envs 已足够\n")

    if args.mode == 'train':
        best_params, history = train(
            generations=args.generations,
            popsize=args.popsize,
            num_envs=args.num_envs,
            sigma=args.sigma,
            log_file=args.params_file,
            record_progress=args.record_progress,
        )
        if best_params is not None:
            print(f"\n✅ 训练完成！climbed {history[-1]['best_height']:.2f}m, "
                  f"forward {history[-1]['best_dist']:.2f}m")
            if args.record_highlights:
                render_highlight_video(args.params_file, args.record_highlights)

    elif args.mode == 'demo':
        if not os.path.exists(args.params_file):
            print(f"❌ 参数文件不存在: {args.params_file}")
            print(f"   先训练: python {__file__} train")
            return
        with open(args.params_file) as f:
            data = json.load(f)
        if not data.get('best_params'):
            print("❌ 没有 best_params")
            return
        params = np.array(data['best_params'])
        demo_swarm(params, num_robots=args.num_robots, record_path=args.record)


if __name__ == '__main__':
    main()

"""
四足机器人跨斜坡强化学习示例（CMA-ES + 多 worker 并行加速）
==========================================================

教学版本：用进化策略 (CMA-ES) 优化参数化步态，
让四足机器人爬上 15° 斜坡而不摔倒。

特点：
- ✅ 完全 CPU 可跑（不需要 GPU）
- ✅ 支持多进程并行（--num_envs N）加速训练
- ✅ 可录制训练全过程 GIF 给学生看"进化"
- ✅ 训练 50 代 ~ 1-3 分钟（取决于核数）
- ✅ 不需要 PyTorch / stable-baselines3

依赖：
    pip install pybullet numpy cma matplotlib imageio

用法：
    # 单进程训练（最慢，但调试方便）
    python quadruped_rl_slope.py train --generations 50

    # 多进程并行训练（4 个 worker，约 4 倍加速）
    python quadruped_rl_slope.py train --generations 50 --num_envs 4

    # 录制训练全过程 GIF（演示用，每 5 代一帧）
    python quadruped_rl_slope.py train --generations 50 --num_envs 4 \\
        --record_progress training.gif

    # 用训练好的参数演示（GUI 实时）
    python quadruped_rl_slope.py demo

    # 演示并录 GIF
    python quadruped_rl_slope.py demo --record demo.gif

关于 GPU 加速：
    PyBullet 本身**不支持 GPU 物理引擎**。要真正使用 GPU 加速大规模并行
    （如同时跑 4096 个机器人），需要切换到：
      • NVIDIA Isaac Lab / Isaac Gym (需 RTX 显卡 + Linux)
      • MuJoCo MJX (Google, JAX 实现, 支持 GPU/TPU)
      • Brax (Google, JAX 实现)
    本教学示例用 multiprocessing 并行 CPU worker，已经够用。
"""

import os
import math
import time
import json
import argparse
import multiprocessing as mp
from functools import partial
import numpy as np
import pybullet as p
import pybullet_data


# ============================================================
# 1. 环境：斜坡 + 四足机器人
# ============================================================

class QuadrupedSlopeEnv:
    """四足机器人爬斜坡环境（每个进程一个独立的 PyBullet 客户端）"""

    SIM_DT = 1.0 / 240.0
    CTRL_DT = 1.0 / 50.0
    EPISODE_T = 6.0
    SLOPE_ANGLE_DEG = 15
    INIT_HEIGHT = 0.48

    def __init__(self, gui=False):
        self.gui = gui
        self.client_id = p.connect(p.GUI if gui else p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=self.client_id)
        p.setGravity(0, 0, -9.81, physicsClientId=self.client_id)
        p.setTimeStep(self.SIM_DT, physicsClientId=self.client_id)
        self.robot = None
        self.leg_joints = {
            'RF': [0, 1, 2], 'LF': [4, 5, 6],
            'RH': [8, 9, 10], 'LH': [12, 13, 14],
        }
        self.t = 0.0
        self.start_x = 0.0
        self._build_world()

    def _build_world(self):
        p.loadURDF('plane.urdf', physicsClientId=self.client_id)
        slope_orn = p.getQuaternionFromEuler([0, -math.radians(self.SLOPE_ANGLE_DEG), 0])
        self.slope = p.loadURDF(
            'cube.urdf', [1.5, 0, 0.25], slope_orn,
            globalScaling=2.0, useFixedBase=True,
            physicsClientId=self.client_id,
        )
        self._spawn_robot()

    def _spawn_robot(self):
        if self.robot is not None:
            p.removeBody(self.robot, physicsClientId=self.client_id)
        start_pos = [0, 0, self.INIT_HEIGHT]
        start_orn = p.getQuaternionFromEuler([math.pi / 2, 0, math.pi / 2])
        self.robot = p.loadURDF('laikago/laikago_toes.urdf',
                                start_pos, start_orn,
                                physicsClientId=self.client_id)

        for joint_ids in self.leg_joints.values():
            for joint_id, target in zip(joint_ids, [0.0, 0.65, -1.20]):
                p.resetJointState(self.robot, joint_id, target,
                                  physicsClientId=self.client_id)

        for _ in range(120):
            for joint_ids in self.leg_joints.values():
                for joint_id, target in zip(joint_ids, [0.0, 0.65, -1.20]):
                    p.setJointMotorControl2(
                        self.robot, joint_id, p.POSITION_CONTROL,
                        targetPosition=target, force=120,
                        positionGain=1.0, velocityGain=0.5,
                        physicsClientId=self.client_id,
                    )
            p.stepSimulation(physicsClientId=self.client_id)

        pos, _ = p.getBasePositionAndOrientation(self.robot, physicsClientId=self.client_id)
        self.start_x = pos[0]
        self.t = 0.0

    def reset(self):
        self._spawn_robot()

    def step(self, params):
        freq, lift, duty, stance_thigh, stance_calf = params[:5]
        phase_lf, phase_rf, phase_lh, phase_rh = params[5:9]
        forward_bias = params[9]
        leg_phases = {
            'LF': phase_lf, 'RF': phase_rf,
            'LH': phase_lh, 'RH': phase_rh,
        }

        sub_steps = int(self.CTRL_DT / self.SIM_DT)
        for _ in range(sub_steps):
            self.t += self.SIM_DT
            for leg_name, joint_ids in self.leg_joints.items():
                phase = leg_phases[leg_name]
                phi = (self.t * freq + phase) % 1.0

                # 步幅幅度（增大让机器人能真正前进）
                stride = 0.35  # thigh 摆动范围 ±0.35 rad

                if phi < duty:
                    # 支撑相：thigh 从前向后扫，推动机身前进
                    s = phi / duty
                    thigh_sway = stride * (0.5 - s)  # +stride/2 → -stride/2
                    z_lift = 0.0
                else:
                    # 摆动相：抬腿+向前迈
                    s = (phi - duty) / max(1.0 - duty, 1e-3)
                    z_lift = lift * math.sin(math.pi * s)
                    thigh_sway = stride * (s - 0.5)  # -stride/2 → +stride/2

                hip = 0.0
                # 前腿和后腿摆动方向相同（都从前往后推 → 都让 thigh 减小推动前进）
                thigh = stance_thigh + thigh_sway - z_lift * 0.5 + forward_bias
                calf = stance_calf - z_lift * 1.5

                for joint_id, target in zip(joint_ids, [hip, thigh, calf]):
                    p.setJointMotorControl2(
                        self.robot, joint_id, p.POSITION_CONTROL,
                        targetPosition=target, force=180,
                        positionGain=1.0, velocityGain=0.6,
                        physicsClientId=self.client_id,
                    )
            p.stepSimulation(physicsClientId=self.client_id)
            if self.gui:
                time.sleep(self.SIM_DT)

        reward, done, info = self._compute_reward()
        return reward, done, info

    def _compute_reward(self):
        pos, orn = p.getBasePositionAndOrientation(self.robot, physicsClientId=self.client_id)
        rpy = p.getEulerFromQuaternion(orn)
        forward_dist = pos[0] - self.start_x
        height = pos[2]
        roll, pitch, _ = rpy

        reward = forward_dist * 1.0
        reward += max(0, height - 0.2) * 0.5

        done = False
        info = {'forward_dist': forward_dist, 'height': height}
        if height < 0.15 or abs(roll) > 1.0 or abs(pitch) > 1.0:
            done = True
            reward -= 3.0
            info['terminated'] = 'fell'
        if self.t >= self.EPISODE_T:
            done = True
            info['terminated'] = 'timeout'
        return reward, done, info

    def render_frame(self, width=480, height=320):
        """渲染当前一帧（用于录制 GIF）"""
        view = p.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=[1.5, 0, 0.3],
            distance=2.5, yaw=70, pitch=-15, roll=0, upAxisIndex=2,
            physicsClientId=self.client_id,
        )
        proj = p.computeProjectionMatrixFOV(
            50, width / height, 0.1, 10,
            physicsClientId=self.client_id,
        )
        _, _, rgb, _, _ = p.getCameraImage(
            width, height, view, proj,
            renderer=p.ER_TINY_RENDERER,
            flags=p.ER_NO_SEGMENTATION_MASK,
            physicsClientId=self.client_id,
        )
        return np.array(rgb, dtype=np.uint8).reshape(height, width, 4)[:, :, :3]

    def close(self):
        if p.isConnected(self.client_id):
            p.disconnect(self.client_id)


# ============================================================
# 2. 策略评估
# ============================================================

def evaluate_policy(env, params, max_steps=300):
    """跑一个完整 episode，返回累计 reward 和距离"""
    env.reset()
    total_reward = 0.0
    last_dist = 0.0
    for _ in range(max_steps):
        reward, done, info = env.step(params)
        total_reward += reward
        last_dist = info.get('forward_dist', last_dist)
        if done:
            break
    return total_reward, last_dist


# ============================================================
# 3. 并行评估：每个 worker 进程独立的 PyBullet 客户端
# ============================================================

# 全局变量：每个 worker 进程有自己的 env 实例
_worker_env = None


def _worker_init():
    """每个 worker 进程启动时调用，创建独立的 PyBullet 客户端"""
    global _worker_env
    _worker_env = QuadrupedSlopeEnv(gui=False)


def _worker_evaluate(params):
    """worker 进程评估一组参数"""
    global _worker_env
    return evaluate_policy(_worker_env, params)


def parallel_evaluate(pool, params_list):
    """用进程池并行评估多组参数"""
    if pool is None:
        # 单进程
        global _worker_env
        if _worker_env is None:
            _worker_env = QuadrupedSlopeEnv(gui=False)
        return [evaluate_policy(_worker_env, p) for p in params_list]
    return list(pool.map(_worker_evaluate, params_list))


# ============================================================
# 4. CMA-ES 训练（支持并行 + 录制 progress GIF）
# ============================================================

def train(generations=50, sigma=0.3, popsize=12, num_envs=1,
          log_file='train_log.json', record_progress=None):
    """
    CMA-ES 训练。

    num_envs: 并行 worker 数（≥2 时启用 multiprocessing）
    record_progress: 录训练过程 GIF 的输出路径
    """
    try:
        import cma
    except ImportError:
        print("❌ 缺少 cma 库，请运行: pip install cma")
        return None

    bounds_low = np.array([0.5, 0.05, 0.4, 0.4, -1.5, 0.0, 0.0, 0.0, 0.0, -0.3])
    bounds_high = np.array([3.0, 0.25, 0.8, 1.0, -0.8, 1.0, 1.0, 1.0, 1.0, 0.3])

    def unnormalize(z):
        z = np.clip(z, -1, 1)
        return bounds_low + (z + 1) / 2 * (bounds_high - bounds_low)

    init_params = np.array([1.6, 0.15, 0.55, 0.65, -1.20, 0.0, 0.5, 0.5, 0.0, 0.1])
    z0 = (init_params - bounds_low) / (bounds_high - bounds_low) * 2 - 1

    es = cma.CMAEvolutionStrategy(
        z0.tolist(), sigma,
        {'popsize': popsize, 'verbose': -9, 'bounds': [[-1] * 10, [1] * 10]}
    )

    # 并行 pool
    pool = None
    if num_envs > 1:
        ctx = mp.get_context('spawn')
        pool = ctx.Pool(num_envs, initializer=_worker_init)
        print(f"🚀 开启 {num_envs} 个 worker 并行训练")

    # 录制 GIF 用的环境（主进程内单独一个，每 record_every 代采样一帧）
    rec_env = None
    rec_frames = []
    if record_progress:
        rec_env = QuadrupedSlopeEnv(gui=False)
        print(f"📹 同步录制训练过程到 {record_progress}")

    history = []
    best_params = None
    best_reward = -np.inf

    t_start = time.time()
    record_every = max(1, generations // 30)  # 总共 ~30 帧 GIF

    try:
        for gen in range(generations):
            solutions = es.ask()
            params_list = [unnormalize(np.array(z)) for z in solutions]
            results = parallel_evaluate(pool, params_list)
            rewards = [-r for r, _ in results]
            dists = [d for _, d in results]
            es.tell(solutions, rewards)

            best_idx = int(np.argmin(rewards))
            gen_best_reward = -rewards[best_idx]
            gen_best_dist = dists[best_idx]
            if gen_best_reward > best_reward:
                best_reward = gen_best_reward
                best_params = params_list[best_idx]

            history.append({
                'gen': gen,
                'best_reward': float(gen_best_reward),
                'best_dist': float(gen_best_dist),
                'mean_reward': float(-np.mean(rewards)),
                'mean_dist': float(np.mean(dists)),
            })
            elapsed = time.time() - t_start
            print(f"Gen {gen+1:3d}/{generations} | "
                  f"best dist = {gen_best_dist:5.2f}m  "
                  f"best reward = {gen_best_reward:6.2f}  "
                  f"mean dist = {np.mean(dists):.2f}m  "
                  f"elapsed = {elapsed:.1f}s")

            # 录制 GIF：每 record_every 代用当前最优参数跑一遍，添加几帧
            if rec_env is not None and (gen % record_every == 0 or gen == generations - 1):
                rec_env.reset()
                for _ in range(80):
                    _, done, _ = rec_env.step(best_params if best_params is not None else params_list[best_idx])
                    if done:
                        break
                frame = rec_env.render_frame()
                # 加文字：当前代数 / 距离
                rec_frames.append(_add_label(frame, gen + 1, gen_best_dist))

    finally:
        if pool is not None:
            pool.close()
            pool.join()
        if rec_env is not None:
            # 末尾停留几帧
            for _ in range(15):
                rec_frames.append(rec_frames[-1])
            if rec_frames:
                import imageio.v2 as imageio
                imageio.mimsave(record_progress, rec_frames, fps=6, loop=0)
                print(f"📹 训练过程 GIF 已保存: {record_progress} ({len(rec_frames)} 帧)")
            rec_env.close()

    # 保存最优参数
    with open(log_file, 'w') as f:
        json.dump({
            'history': history,
            'best_params': best_params.tolist() if best_params is not None else None,
            'best_reward': float(best_reward),
            'config': {
                'generations': generations,
                'popsize': popsize,
                'num_envs': num_envs,
                'sigma': sigma,
            }
        }, f, indent=2)
    print(f"💾 训练日志保存: {log_file}")

    return best_params, history


def _add_label(frame, gen, dist):
    """在 GIF 帧上加标签"""
    from PIL import Image, ImageDraw, ImageFont
    img = Image.fromarray(frame)
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', 18)
        font_s = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 12)
    except OSError:
        font = font_s = ImageFont.load_default()
    draw.rectangle([(0, 0), (frame.shape[1], 32)], fill=(255, 255, 255, 230))
    draw.text((10, 5), f"Gen {gen}", fill=(20, 30, 80), font=font)
    draw.text((130, 9), f"Best dist: {dist:.2f}m", fill=(80, 30, 30), font=font_s)
    return np.array(img)


# ============================================================
# 5. Demo（用训练好的参数演示）
# ============================================================

def demo(params, gui=True, record_path=None):
    env = QuadrupedSlopeEnv(gui=gui and record_path is None)
    try:
        env.reset()
        images = []
        print("🎬 开始演示...")
        for step in range(300):
            _, done, info = env.step(params)
            if record_path:
                images.append(env.render_frame())
            if done:
                print(f"  Step {step}: 前进 {info.get('forward_dist', 0):.2f}m, "
                      f"原因: {info.get('terminated', 'unknown')}")
                break

        if record_path and images:
            import imageio.v2 as imageio
            imageio.mimsave(record_path, images, fps=25, loop=0)
            print(f"📹 演示 GIF 已保存: {record_path}")
    finally:
        env.close()


# ============================================================
# 6. CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
                                     description=__doc__)
    parser.add_argument('mode', choices=['train', 'demo'],
                        help='train 训练 / demo 演示')
    parser.add_argument('--generations', type=int, default=50, help='CMA-ES 代数')
    parser.add_argument('--popsize', type=int, default=12, help='种群大小')
    parser.add_argument('--num_envs', type=int, default=1,
                        help='并行 worker 数（实验台数）。默认 1，建议 = CPU 核数')
    parser.add_argument('--gpu', action='store_true',
                        help='GPU 加速（PyBullet 不支持；提示切换到 Isaac/MJX/Brax）')
    parser.add_argument('--params_file', default='train_log.json',
                        help='参数文件路径')
    parser.add_argument('--record', help='演示 GIF 输出路径')
    parser.add_argument('--record_progress',
                        help='训练过程 GIF 输出路径（每几代采样一帧）')
    args = parser.parse_args()

    if args.gpu:
        print("⚠️  PyBullet 物理引擎不支持 GPU 加速。要想使用 GPU 大规模并行：")
        print("    - 切换到 NVIDIA Isaac Lab / Isaac Gym (需 RTX 显卡)")
        print("    - 或 MuJoCo MJX / Google Brax (JAX, 支持 GPU/TPU)")
        print("    本示例使用 CPU 多进程 --num_envs 已经够快了\n")

    if args.mode == 'train':
        best_params, history = train(
            generations=args.generations,
            popsize=args.popsize,
            num_envs=args.num_envs,
            log_file=args.params_file,
            record_progress=args.record_progress,
        )
        if best_params is not None:
            print(f"\n✅ 训练完成！最优奖励 {history[-1]['best_reward']:.2f}，"
                  f"最优前进距离 {history[-1]['best_dist']:.2f}m")
            print(f"   下一步：python {__file__} demo --record demo.gif")

    elif args.mode == 'demo':
        if not os.path.exists(args.params_file):
            print(f"❌ 参数文件不存在: {args.params_file}")
            print(f"   请先训练: python {__file__} train")
            return
        with open(args.params_file) as f:
            data = json.load(f)
        if not data.get('best_params'):
            print(f"❌ 参数文件没有 best_params 字段")
            return
        params = np.array(data['best_params'])
        print(f"📂 加载参数 (best_reward = {data['best_reward']:.2f})")
        demo(params, gui=args.record is None, record_path=args.record)


if __name__ == '__main__':
    main()

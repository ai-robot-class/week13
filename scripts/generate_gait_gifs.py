"""
用 PyBullet 仿真生成 Week 13 步态动画（v2：固定悬浮版）

策略：让机器人固定悬浮在空中，4 条腿做出步态动作。
这样：
✅ 机器人绝不会摔（固定 base）
✅ 学生能清楚看到 4 条腿的相位关系
✅ 视觉上像在"半空踏步"，符合教学目的

输出：
- pybullet_walk.gif
- pybullet_trot.gif
- pybullet_bound.gif
- pybullet_compare.gif（三步态横向对比）
"""
import os
import math
import numpy as np
import pybullet as p
import pybullet_data
import imageio.v2 as imageio
from PIL import Image, ImageDraw, ImageFont

OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'assets', 'gaits')
os.makedirs(OUT_DIR, exist_ok=True)

WIDTH = 480
HEIGHT = 320
FPS = 20
SECONDS = 4


# 各步态参数：phase = 相位偏移（[0,1)），duty = 支撑相占比
GAITS = {
    'walk': {
        'phase': {'LF': 0.0, 'RF': 0.5, 'LH': 0.25, 'RH': 0.75},
        'duty': 0.75, 'freq': 1.0, 'lift': 0.18, 'sway': 0.06,
        'desc': 'Walk (slow, ultra stable)',
    },
    'trot': {
        'phase': {'LF': 0.0, 'RF': 0.5, 'LH': 0.5, 'RH': 0.0},
        'duty': 0.55, 'freq': 1.6, 'lift': 0.22, 'sway': 0.08,
        'desc': 'Trot (diagonal pairs, daily gait)',
    },
    'bound': {
        'phase': {'LF': 0.0, 'RF': 0.0, 'LH': 0.5, 'RH': 0.5},
        'duty': 0.50, 'freq': 2.0, 'lift': 0.24, 'sway': 0.10,
        'desc': 'Bound (front pair + hind pair, rabbit hop)',
    },
}


def setup_robot():
    """加载机器人，固定悬浮在地面上方"""
    if p.isConnected():
        p.disconnect()
    p.connect(p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)
    p.setTimeStep(1.0 / 240.0)

    # 地面（视觉参考）
    p.loadURDF('plane.urdf')

    # 机器人 base 固定在合适高度（足端刚好踩地）
    start_pos = [0, 0, 0.48]
    start_orn = p.getQuaternionFromEuler([math.pi / 2, 0, math.pi / 2])
    robot = p.loadURDF('laikago/laikago_toes.urdf',
                       start_pos, start_orn,
                       useFixedBase=True)  # ⭐ 固定 base 不会摔倒

    return robot


def get_leg_joints():
    """关节映射：FR/FL/RR/RL → RF/LF/RH/LH"""
    return {
        'RF': (0, 1, 2),     # Front-Right
        'LF': (4, 5, 6),     # Front-Left
        'RH': (8, 9, 10),    # Rear-Right
        'LH': (12, 13, 14),  # Rear-Left
    }


def gait_target(phase, duty, lift, sway, leg_name):
    """
    步态足端轨迹 → 关节角度

    nominal pose（站立）：
    - hip = 0
    - thigh ≈ 0.65 (腿向下弯)
    - calf ≈ -1.20 (小腿弯曲)

    步态:
    - 支撑相 (phi < duty): 假装"踩地"，腿向下伸直（少量）
    - 摆动相 (phi >= duty): 抬腿，thigh 减小，calf 弯曲更紧
    """
    base_hip = 0.0
    base_thigh = 0.65
    base_calf = -1.20

    if phase < duty:
        # 支撑相：腿稍微向下推（接触地面感）
        s = phase / duty  # 0 → 1
        # 在支撑相内做轻微的前后摆（推动感）
        x_sway = sway * (0.5 - s) * 0.8
        z_lift = -0.02  # 稍微向下伸
    else:
        # 摆动相：抬腿（半正弦）
        s = (phase - duty) / max(1.0 - duty, 1e-3)
        x_sway = sway * (s - 0.5) * 0.8
        z_lift = lift * math.sin(math.pi * s)

    # 直接调整关节角实现足端轨迹
    # z_lift 增大 → thigh 减小, calf 收紧
    direction = 1.0 if leg_name in ('LF', 'RF') else -1.0

    hip = base_hip
    thigh = base_thigh - z_lift * 1.8 + x_sway * direction * 0.6
    calf = base_calf + z_lift * 2.5

    return hip, thigh, calf


def settle_robot(robot, leg_joints, n_steps=300):
    """让机器人在 nominal 站立姿态下稳定"""
    for leg_name, joint_ids in leg_joints.items():
        # nominal 站立角度
        for joint_id, target in zip(joint_ids, [0.0, 0.65, -1.20]):
            p.setJointMotorControl2(
                robot, joint_id, p.POSITION_CONTROL,
                targetPosition=target, force=200,
                positionGain=1.0, velocityGain=0.5,
            )
    for _ in range(n_steps):
        p.stepSimulation()


def run_gait(gait_name, n_frames):
    """运行一段步态，返回每帧渲染图"""
    g = GAITS[gait_name]
    robot = setup_robot()
    leg_joints = get_leg_joints()
    settle_robot(robot, leg_joints)

    images = []
    t = 0.0
    dt = 1.0 / FPS
    sub_steps = max(1, int(240 / FPS))

    for frame in range(n_frames):
        cycle_t = (t * g['freq']) % 1.0

        for leg_name, joint_ids in leg_joints.items():
            phase = g['phase'][leg_name]
            phi = (t * g['freq'] + phase) % 1.0
            hip, thigh, calf = gait_target(phi, g['duty'], g['lift'], g['sway'], leg_name)
            for joint_id, target in zip(joint_ids, [hip, thigh, calf]):
                p.setJointMotorControl2(
                    robot, joint_id, p.POSITION_CONTROL,
                    targetPosition=target, force=200,
                    positionGain=1.0, velocityGain=0.5,
                )

        for _ in range(sub_steps):
            p.stepSimulation()

        img = render_frame(robot, gait_name, cycle_t, g['desc'], leg_joints, g)
        images.append(img)
        t += dt

    return images


def render_frame(robot, gait_name, phase, label_text, leg_joints, g):
    """渲染当前仿真状态为 numpy 图像"""
    cam_target = [0, 0, 0.4]
    view = p.computeViewMatrixFromYawPitchRoll(
        cameraTargetPosition=cam_target,
        distance=1.4, yaw=45, pitch=-15, roll=0, upAxisIndex=2,
    )
    proj = p.computeProjectionMatrixFOV(fov=50, aspect=WIDTH / HEIGHT,
                                        nearVal=0.1, farVal=10.0)

    _, _, rgb, _, _ = p.getCameraImage(
        WIDTH, HEIGHT, view, proj,
        renderer=p.ER_TINY_RENDERER,
        flags=p.ER_NO_SEGMENTATION_MASK,
    )
    rgb = np.array(rgb, dtype=np.uint8).reshape(HEIGHT, WIDTH, 4)[:, :, :3]

    img = Image.fromarray(rgb)
    draw = ImageDraw.Draw(img)
    try:
        font_big = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', 18)
        font_mid = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', 14)
        font_small = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 11)
    except OSError:
        font_big = font_mid = font_small = ImageFont.load_default()

    # 顶部标题
    draw.rectangle([(0, 0), (WIDTH, 30)], fill=(255, 255, 255, 230))
    draw.text((10, 5), label_text, fill=(20, 30, 80), font=font_mid)

    # 右上：cycle 进度
    cycle_text = f"Cycle: {phase:.2f}"
    draw.text((WIDTH - 90, 5), cycle_text, fill=(80, 30, 30), font=font_small)

    # 底部：4 条腿状态指示器
    bar_y = HEIGHT - 35
    draw.rectangle([(0, bar_y), (WIDTH, HEIGHT)], fill=(255, 255, 255, 230))

    leg_order = ['LF', 'RF', 'LH', 'RH']
    leg_x_start = 15
    leg_x_step = 110

    for i, leg_name in enumerate(leg_order):
        leg_phase = g['phase'][leg_name]
        phi = (phase + leg_phase) % 1.0
        is_stance = phi < g['duty']

        x = leg_x_start + i * leg_x_step
        # 标签
        draw.text((x, bar_y + 3), leg_name, fill=(60, 60, 60), font=font_small)

        # 状态指示器（小圆）
        color = (50, 150, 50) if is_stance else (220, 130, 50)
        symbol = "STANCE" if is_stance else "SWING"
        draw.ellipse([(x + 25, bar_y + 5), (x + 39, bar_y + 19)],
                     fill=color, outline=(50, 50, 50))
        draw.text((x + 45, bar_y + 6), symbol, fill=color, font=font_small)

    return np.array(img)


def save_gif(images, filename, fps=FPS):
    out_path = os.path.join(OUT_DIR, filename)
    imageio.mimsave(out_path, images, fps=fps, loop=0)
    size_kb = os.path.getsize(out_path) / 1024
    print(f"  ✅ {filename}: {size_kb:.1f} KB ({len(images)} 帧)")
    return out_path


def main():
    print(f"📁 输出: {OUT_DIR}\n")
    n_frames = SECONDS * FPS

    print(f"🎬 生成稳定步态动画（机器人固定悬浮）每段 {SECONDS}s, {n_frames} 帧\n")

    all_frames = {}
    for gait_name in ['walk', 'trot', 'bound']:
        print(f"▶ {gait_name.upper()}...")
        frames = run_gait(gait_name, n_frames)
        all_frames[gait_name] = frames
        save_gif(frames, f'pybullet_{gait_name}.gif')

    # 三步态横向拼接
    print("\n▶ 三步态对比拼接...")
    h, w = HEIGHT, WIDTH
    combined = []
    for i in range(n_frames):
        canvas = Image.new('RGB', (w * 3 + 30, h + 20), color=(245, 245, 250))
        for j, gait_name in enumerate(['walk', 'trot', 'bound']):
            frame = Image.fromarray(all_frames[gait_name][i])
            canvas.paste(frame, (j * (w + 10) + 10, 10))
        combined.append(np.array(canvas))
    save_gif(combined, 'pybullet_compare.gif', fps=FPS)

    p.disconnect()
    print("\n✨ 完成！")


if __name__ == '__main__':
    main()

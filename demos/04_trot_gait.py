"""Week 13 demo 04: clearer fixed-base trot gait visualization."""

import math
import time

import pybullet as p
import pybullet_data


LEG_JOINTS = {
    "RF": [0, 1, 2],
    "LF": [4, 5, 6],
    "RH": [8, 9, 10],
    "LH": [12, 13, 14],
}


def smoothstep(s):
    s = max(0.0, min(1.0, s))
    return s * s * (3.0 - 2.0 * s)


def trot_targets(t, leg_name, frequency=1.2):
    phases = {"LF": 0.0, "RH": 0.0, "RF": 0.5, "LH": 0.5}
    duty = 0.55
    phi = (frequency * t + phases[leg_name]) % 1.0

    if phi < duty:
        s = smoothstep(phi / duty)
        foot_lift = 0.0
        stride = 0.12 * (s - 0.5)
    else:
        s = smoothstep((phi - duty) / (1.0 - duty))
        foot_lift = 0.18 * math.sin(math.pi * s)
        stride = 0.12 * (0.5 - s)

    hip = 0.0
    thigh = 0.65 + stride - 1.3 * foot_lift
    calf = -1.20 + 2.0 * foot_lift
    return hip, thigh, calf


def main():
    p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)
    p.loadURDF("plane.urdf")

    orn = p.getQuaternionFromEuler([math.pi / 2, 0, math.pi / 2])
    robot = p.loadURDF("laikago/laikago_toes.urdf", [0, 0, 0.52], orn, useFixedBase=True)

    t = 0.0
    dt = 1.0 / 240.0
    while p.isConnected():
        for leg_name, joints in LEG_JOINTS.items():
            targets = trot_targets(t, leg_name)
            for joint_id, target in zip(joints, targets):
                p.setJointMotorControl2(
                    robot,
                    joint_id,
                    p.POSITION_CONTROL,
                    targetPosition=target,
                    force=180,
                    positionGain=1.0,
                    velocityGain=0.5,
                )
        p.stepSimulation()
        time.sleep(dt)
        t += dt


if __name__ == "__main__":
    main()

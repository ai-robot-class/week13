"""Week 13 demo 03: simple sinusoidal trot-like joint targets."""

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


def simple_gait(t, leg_name, frequency=0.8):
    phases = {"LF": 0.0, "RH": 0.0, "RF": 0.5, "LH": 0.5}
    phase = 2.0 * math.pi * (frequency * t + phases[leg_name])
    lift = max(0.0, math.sin(phase))
    hip = 0.0
    thigh = 0.65 - 0.28 * lift
    calf = -1.20 + 0.45 * lift
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
            targets = simple_gait(t, leg_name)
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

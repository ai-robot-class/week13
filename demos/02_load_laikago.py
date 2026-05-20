"""Week 13 demo 02: load the PyBullet Laikago quadruped model."""

import math
import time

import pybullet as p
import pybullet_data


def main():
    p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)
    p.loadURDF("plane.urdf")

    orn = p.getQuaternionFromEuler([math.pi / 2, 0, math.pi / 2])
    p.loadURDF("laikago/laikago_toes.urdf", [0, 0, 0.5], orn)

    for _ in range(1200):
        p.stepSimulation()
        time.sleep(1.0 / 240.0)

    p.disconnect()


if __name__ == "__main__":
    main()

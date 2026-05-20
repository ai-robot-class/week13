"""Week 13 demo 01: minimal PyBullet scene with a plane and a falling box."""

import time

import pybullet as p
import pybullet_data


def main():
    p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)
    p.loadURDF("plane.urdf")
    p.loadURDF("cube.urdf", [0, 0, 1.0])

    for _ in range(600):
        p.stepSimulation()
        time.sleep(1.0 / 240.0)

    p.disconnect()


if __name__ == "__main__":
    main()

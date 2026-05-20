# 第 13 周：四足机器人强化学习爬楼梯

本仓库提供 AI Robot 课程第 13 周的最终版强化学习代码与预训练模型。主课程网站通过 Git Submodule 引用本仓库。

实验方法为 **PPO + residual controller**：基础步态生成器提供四足机器人的周期性运动，PPO 策略在基础步态上输出关节残差动作，用于改进平地跑步与低台阶爬楼梯表现。

## 快速复现实验效果

安装依赖：

```bash
pip install pybullet numpy gymnasium stable-baselines3 torch opencv-python
```

在本仓库根目录运行以下命令，可打开 PyBullet GUI，并加载预训练模型演示低台阶爬楼梯：

```bash
python3 quadruped_ppo_residual_stairs.py demo --task stairs --model ppo_residual_stairs.zip --stair_steps 4 --step_height 0.03 --init_x 0.00 --steps 500 --gui
```

预期现象：四足机器人能够明显向前爬上约三阶低台阶，但尚未满足“在最终台阶上稳定站住”的严格成功标准。因此，该模型是课程训练得到的阶段性成果，而不是完整解决楼梯任务的最终策略。

若在作业仓库中通过 Git Submodule 使用本仓库，可参考下一节。

## 在学生作业仓库中使用 Submodule

学生通常应在自己的作业仓库中引用本仓库，而不是复制全部代码文件。推荐目录结构如下：

```text
student-homework-repo/
├── week13/                 # 本仓库作为 submodule
├── reports/                # 实验报告、截图、视频说明
└── README.md               # 学生自己的作业说明
```

### 1. 添加 submodule

在学生自己的作业仓库根目录运行：

```bash
git submodule add https://github.com/ai-robot-class/week13.git week13
git commit -m "Add week13 quadruped RL submodule"
```

这一步会在作业仓库中生成 `.gitmodules` 文件，并记录 `week13` 子仓库当前指向的提交。

### 2. 首次拉取或换电脑后初始化

如果已经 clone 了作业仓库，但 `week13/` 目录为空，需要运行：

```bash
git submodule update --init --recursive
```

也可以在 clone 作业仓库时一次性拉取 submodule：

```bash
git clone --recurse-submodules <学生作业仓库地址>
```

### 3. 从作业仓库根目录直接运行演示

安装依赖后，在学生作业仓库根目录运行：

```bash
python3 week13/quadruped_ppo_residual_stairs.py demo --task stairs --model week13/ppo_residual_stairs.zip --stair_steps 4 --step_height 0.03 --init_x 0.00 --steps 500 --gui
```

无图形界面时可录制视频：

```bash
python3 week13/quadruped_ppo_residual_stairs.py demo --task stairs --model week13/ppo_residual_stairs.zip --stair_steps 4 --step_height 0.03 --init_x 0.00 --steps 500 --record week13/student_stairs_demo.mp4
```

### 4. 更新到课程仓库的最新版本

若课程组更新了 `ai-robot-class/week13`，学生可在作业仓库中执行：

```bash
git submodule update --remote week13
git add week13
git commit -m "Update week13 submodule"
```

注意：Git submodule 记录的是一个具体提交，而不是自动跟随最新 main 分支。因此更新后需要在作业仓库中提交新的 submodule 指针。

## 文件结构

```text
.
├── quadruped_ppo_residual_stairs.py      # PPO + residual controller 主程序
├── ppo_run_flat.zip                      # 平地跑步预训练模型
└── ppo_residual_stairs.zip               # 低台阶预训练模型
```

## 使用预训练模型

### 平地跑步演示

```bash
python3 quadruped_ppo_residual_stairs.py demo \
    --task run \
    --model ppo_run_flat.zip \
    --steps 500 \
    --gui
```

### 低台阶爬楼梯演示

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

### 录制演示视频

在无图形界面的服务器环境中，或需要保存实验结果时，可使用 `--record` 参数录制视频：

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

## 基于预训练模型继续训练

可以从低台阶预训练模型继续训练，以观察奖励函数、动作空间或训练步数调整后的变化：

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

训练完成后可录制新策略的演示视频：

```bash
python3 quadruped_ppo_residual_stairs.py demo \
    --task stairs \
    --model student_stairs_continue.zip \
    --stair_steps 4 \
    --step_height 0.03 \
    --init_x 0.00 \
    --steps 500 \
    --record student_stairs_continue.mp4
```

## 成功判定标准

本任务中，“成功”不等同于腿部高度超过台阶，也不等同于机身短暂冲上台阶。严格成功需要同时满足：

1. 机身中心到达最终台阶区域
2. 机身保持直立
3. 至少两只脚趾稳定接触最终台阶
4. 身体、大腿、小腿不能作为支撑卡在台阶上
5. 线速度和角速度足够小
6. 连续稳定保持一段时间

当前 `ppo_residual_stairs.zip` 展示了训练带来的显著阶段性进步：机器人可以爬上约三阶低台阶；但该策略尚未达到上述严格成功标准。

## 常见问题

### 无法打开显示窗口

在无图形界面的环境中，请使用 `--record` 参数录制视频，并省略 `--gui`。

### 训练速度较慢

PyBullet 使用 CPU 物理仿真，无法像 Isaac Gym / Isaac Lab 一样进行大规模 GPU 并行仿真。本仓库更适合教学演示、奖励函数调试和小规模实验。

### 不建议直接训练完整楼梯任务的原因

直接训练完整楼梯任务探索难度较高，策略容易收敛到无效行为，例如向前扑倒、抬腿后翻倒、使用身体或腿部卡住台阶。分阶段课程学习可以降低初始探索难度，使策略先获得基本前进能力，再逐步学习上台阶。

## English Summary

This repository contains the final runnable Week 13 quadruped reinforcement learning code. The main GUI demo is:

```bash
python3 quadruped_ppo_residual_stairs.py demo --task stairs --model ppo_residual_stairs.zip --stair_steps 4 --step_height 0.03 --init_x 0.00 --steps 500 --gui
```

The provided policy is a learning-progress checkpoint, not a fully solved stair-climbing policy.

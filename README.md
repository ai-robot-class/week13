# 第 13 周：四足机器人强化学习爬楼梯

这是 AI Robot 课程第 13 周的可运行代码仓库。主课程网站通过 Git Submodule 引用本仓库。

本仓库包含两类内容：

- **PPO + residual controller**：本周重点，用 PPO 在基础步态上学习残差动作，让机器狗尝试跑步、跳跃、爬低台阶。
- **CMA-ES 旧版基线**：早期用进化策略训练斜坡/楼梯的教学示例，方便对比不同强化学习方法。

## 一条命令复现爬楼梯效果

进入本仓库后，先安装依赖：

```bash
pip install pybullet numpy gymnasium stable-baselines3 torch opencv-python
```

然后运行下面这条命令，会直接打开 PyBullet 显示窗口，加载我们训练好的模型，让机器狗尝试爬低台阶：

```bash
python3 quadruped_ppo_residual_stairs.py demo --task stairs --model ppo_residual_stairs.zip --stair_steps 4 --step_height 0.03 --init_x 0.00 --steps 500 --gui
```

你应该能看到：机器狗已经可以明显向前爬上约三阶低台阶，但还没有达到“在最后一级台阶上稳定站住”的严格成功标准。

如果你是在主课程仓库 `ai-robot-class.github.io` 里使用 submodule，命令是：

```bash
git submodule update --init --recursive
python3 week13/quadruped_ppo_residual_stairs.py demo --task stairs --model week13/ppo_residual_stairs.zip --stair_steps 4 --step_height 0.03 --init_x 0.00 --steps 500 --gui
```

## 文件说明

```text
.
├── quadruped_ppo_residual_stairs.py      # PPO + residual controller 主程序
├── run_quadruped_skill_curriculum.py     # 长时间 run -> jump -> stairs 课程训练脚本
├── quadruped_training_debug_notes.md     # 本次 7 小时 22 分钟训练调试记录
├── ppo_run_flat.zip                      # 已训练好的平地跑步模型
├── ppo_residual_stairs.zip               # 已训练好的低台阶模型
├── quadruped_rl_stairs.py                # 旧版 CMA-ES 爬楼梯基线
├── quadruped_rl_slope.py                 # 旧版 CMA-ES 斜坡基线
├── train_log.json                        # 示例训练日志
└── train_log_stairs.json                 # 楼梯训练日志
```

## 直接使用训练成果

### 1. 看平地跑步

```bash
python3 quadruped_ppo_residual_stairs.py demo \
    --task run \
    --model ppo_run_flat.zip \
    --steps 500 \
    --gui
```

### 2. 看低台阶爬楼梯

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

### 3. 录制视频

如果服务器没有显示器，或者你想保存结果视频，可以去掉 `--gui`，改用 `--record`：

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

## 从头训练

建议不要一开始就直接训练楼梯。更稳的顺序是：

```text
平地跑步 -> 低平台跳跃 -> 低台阶爬楼梯
```

### 第一步：训练平地跑步

```bash
python3 quadruped_ppo_residual_stairs.py train \
    --task run \
    --timesteps 300000 \
    --num_envs 8 \
    --batch_size 2048 \
    --curriculum \
    --model student_run_flat.zip
```

### 第二步：加载跑步模型，训练低平台跳跃

```bash
python3 quadruped_ppo_residual_stairs.py train \
    --task jump \
    --load_model student_run_flat.zip \
    --timesteps 400000 \
    --num_envs 8 \
    --batch_size 2048 \
    --curriculum \
    --model student_jump_low.zip
```

### 第三步：加载跳跃模型，训练低台阶

```bash
python3 quadruped_ppo_residual_stairs.py train \
    --task stairs \
    --load_model student_jump_low.zip \
    --timesteps 500000 \
    --num_envs 8 \
    --batch_size 2048 \
    --curriculum \
    --model student_stairs_low.zip
```

## 从现有模型继续训练

如果你想在老师提供的低台阶模型基础上继续改进：

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

训练完成后录制新模型：

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

## 长时间自动训练

如果想让程序自动循环训练、录视频、整理最好片段：

```bash
python3 run_quadruped_skill_curriculum.py --hours 10 --num_envs 8
```

输出会保存到：

```text
rl_runs/<时间戳>/
├── models/             # 每轮模型
├── videos/             # 每轮 demo 视频
├── logs/               # 训练与 demo 日志
├── important_videos/   # 自动挑出的重要视频
└── SUMMARY.md          # 训练摘要
```

## 如何判断是否成功

本任务里，“成功”不是腿抬得高，也不是身体碰巧冲上台阶。严格成功需要同时满足：

1. 机身中心到达最后一级台阶区域
2. 机身保持直立
3. 至少两只脚趾稳定接触最终台阶
4. 身体、大腿、小腿不能把机器人卡在台阶上
5. 速度和角速度足够小
6. 连续稳定保持一小段时间

当前 `ppo_residual_stairs.zip` 的意义是：它展示了训练确实产生了明显进步，机器狗可以爬上约三阶低台阶；但它还不是一个严格完成任务的最终模型。

## 常见问题

### 没有显示窗口怎么办？

使用 `--record` 录制视频，不要加 `--gui`。

### 训练很慢正常吗？

正常。PyBullet 是 CPU 物理仿真，和 Isaac Gym / Isaac Lab 这类 GPU 并行仿真不同。这里更适合作为教学和调试示例。

### 为什么不直接训练完整楼梯？

直接训练完整楼梯太难，策略很容易学到“向前扑倒”“抬腿但翻车”“用身体卡台阶”等投机动作。所以我们用课程学习：先跑，再跳，再上低台阶。

## English Short Note

This repository contains the runnable Week 13 quadruped reinforcement learning code. The main demo is:

```bash
python3 quadruped_ppo_residual_stairs.py demo --task stairs --model ppo_residual_stairs.zip --stair_steps 4 --step_height 0.03 --init_x 0.00 --steps 500 --gui
```

The provided policy is a learning-progress model, not a perfect stair-climbing solution.

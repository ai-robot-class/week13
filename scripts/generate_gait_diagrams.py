"""
生成漂亮的步态相位图和速度-能耗对比图（替代 ASCII 表）
"""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch
import matplotlib.font_manager as fm

OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'assets', 'gaits')
os.makedirs(OUT_DIR, exist_ok=True)


# ============================================================
# 1. 5 种步态相位图（甘特图风格）
# ============================================================
def gen_gait_patterns():
    """5 种步态的"节拍图"，用色块代替■□"""
    fig, axes = plt.subplots(5, 1, figsize=(11, 9), dpi=120, sharex=True)
    fig.patch.set_facecolor('#fafafa')

    gaits = {
        'Walk':  {'phase': {'LF': 0.0,  'RF': 0.5, 'LH': 0.25, 'RH': 0.75}, 'duty': 0.75, 'color': '#10b981', 'desc': '4-beat · 3 legs grounded always · ultra stable, slow'},
        'Trot':  {'phase': {'LF': 0.0, 'RF': 0.5, 'LH': 0.5, 'RH': 0.0},   'duty': 0.50, 'color': '#3b82f6', 'desc': 'Diagonal pairs · dogs/horses everyday gait'},
        'Pace':  {'phase': {'LF': 0.0, 'RF': 0.5, 'LH': 0.0, 'RH': 0.5},   'duty': 0.50, 'color': '#f59e0b', 'desc': 'Same-side pairs · camels, giraffes'},
        'Bound': {'phase': {'LF': 0.0, 'RF': 0.0, 'LH': 0.5, 'RH': 0.5},   'duty': 0.45, 'color': '#ef4444', 'desc': 'Front pair + Hind pair · rabbits, cheetahs'},
        'Gallop':{'phase': {'LF': 0.0, 'RF': 0.1, 'LH': 0.45, 'RH': 0.55}, 'duty': 0.30, 'color': '#a855f7', 'desc': '4-beat asymmetric · top speed (cheetah 110 km/h)'},
    }
    leg_names = ['LF', 'RF', 'LH', 'RH']
    leg_colors_idle = '#e5e7eb'

    n_cycles = 2.5
    resolution = 200
    t = np.linspace(0, n_cycles, resolution)

    for idx, (gait_name, info) in enumerate(gaits.items()):
        ax = axes[idx]
        ax.set_facecolor('white')
        ax.set_xlim(0, n_cycles)
        ax.set_ylim(-0.5, len(leg_names) - 0.5)
        ax.set_yticks(range(len(leg_names)))
        ax.set_yticklabels(leg_names, fontsize=11, fontweight='bold')
        ax.invert_yaxis()
        ax.grid(True, axis='x', alpha=0.3)

        for leg_idx, leg in enumerate(leg_names):
            phase = info['phase'][leg]
            duty = info['duty']
            # 画"着地"的横条
            for cycle in range(int(n_cycles) + 1):
                start = cycle + phase
                end = start + duty
                if end > 0 and start < n_cycles:
                    s = max(start, 0)
                    e = min(end, n_cycles)
                    ax.add_patch(Rectangle((s, leg_idx - 0.3), e - s, 0.6,
                                            facecolor=info['color'],
                                            edgecolor='none', alpha=0.85))

        # 标题
        ax.set_title(f"{gait_name}", fontsize=13, fontweight='bold',
                     loc='left', color=info['color'])
        ax.text(n_cycles - 0.05, -0.5, info['desc'], ha='right', va='top',
                fontsize=9.5, color='#666', style='italic')

        if idx < len(gaits) - 1:
            ax.set_xticklabels([])

    axes[-1].set_xlabel('Time (cycles)', fontsize=11)
    axes[-1].set_xticks(np.arange(0, n_cycles + 0.1, 0.5))

    # 图例
    fig.suptitle('Quadruped Gait Patterns - When does each leg touch the ground?',
                 fontsize=14, fontweight='bold', y=0.995)

    # 底部图例说明
    fig.text(0.5, 0.005,
             '■ Stance (foot on ground)    □ Swing (foot in air)',
             ha='center', fontsize=10, color='#374151',
             bbox=dict(facecolor='#f3f4f6', edgecolor='#d1d5db',
                       boxstyle='round,pad=0.4'))

    plt.tight_layout(rect=[0, 0.025, 1, 0.99])
    out = os.path.join(OUT_DIR, 'gait_patterns.png')
    plt.savefig(out, dpi=150, bbox_inches='tight', facecolor='#fafafa')
    plt.close(fig)
    size_kb = os.path.getsize(out) / 1024
    print(f"  ✅ gait_patterns.png ({size_kb:.0f} KB)")


# ============================================================
# 2. 步态速度-能耗对比散点图
# ============================================================
def gen_speed_energy():
    """5 种步态的速度-能耗对比"""
    fig, ax = plt.subplots(figsize=(10, 6), dpi=120)
    fig.patch.set_facecolor('#fafafa')

    gaits = [
        # name, energy, speed, color, animal_examples
        ('Walk',   1.0, 1.5,  '#10b981', 'Large dogs\nElephants\nPatrol robots'),
        ('Trot',   2.5, 5.0,  '#3b82f6', 'Dogs/horses (daily)\nSpot, Unitree Go2'),
        ('Pace',   3.0, 5.5,  '#f59e0b', 'Camels\nGiraffes'),
        ('Bound',  4.5, 8.5,  '#ef4444', 'Rabbits\nCats (short sprint)'),
        ('Gallop', 7.0, 12.0, '#a855f7', 'Cheetahs (110 km/h)\nHorses (70 km/h)'),
    ]

    # 散点
    sizes = [350, 500, 350, 450, 600]
    for (name, e, s, c, anim_label), size in zip(gaits, sizes):
        ax.scatter(e, s, s=size, c=c, edgecolors='white',
                   linewidths=2.5, zorder=5, alpha=0.85)
        # 名字标签
        ax.text(e, s + 0.4, name, ha='center', va='bottom',
                fontsize=13, fontweight='bold', color=c)
        # 动物例子标签
        offset_y = -1.2 if name == 'Walk' else -0.4
        ax.annotate(anim_label, xy=(e, s),
                    xytext=(e + 0.4, s + offset_y),
                    fontsize=8.5, color='#475569',
                    bbox=dict(facecolor='white', edgecolor=c,
                              boxstyle='round,pad=0.3', alpha=0.95))

    # 平滑趋势线
    es = np.array([g[1] for g in gaits])
    ss = np.array([g[2] for g in gaits])
    z = np.polyfit(es, ss, 2)
    p = np.poly1d(z)
    e_smooth = np.linspace(0.5, 7.5, 100)
    ax.plot(e_smooth, p(e_smooth), '--', color='#94a3b8',
            linewidth=2, alpha=0.5, label='Trend')

    # 区域标注
    ax.axhspan(0, 2, facecolor='#dcfce7', alpha=0.3)
    ax.axhspan(2, 7, facecolor='#dbeafe', alpha=0.3)
    ax.axhspan(7, 14, facecolor='#fee2e2', alpha=0.3)

    ax.text(7.5, 1, 'STATIC\n(slow & steady)', ha='right', va='center',
            fontsize=10, color='#16a34a', fontweight='bold', alpha=0.7)
    ax.text(7.5, 4.5, 'DYNAMIC\n(daily movement)', ha='right', va='center',
            fontsize=10, color='#1e40af', fontweight='bold', alpha=0.7)
    ax.text(7.5, 10, 'AERIAL\n(high speed sprint)', ha='right', va='center',
            fontsize=10, color='#dc2626', fontweight='bold', alpha=0.7)

    ax.set_xlim(0, 8)
    ax.set_ylim(0, 14)
    ax.set_xlabel('Energy per meter (relative)  →  more energy', fontsize=12)
    ax.set_ylabel('Speed (km/h)  →  faster', fontsize=12)
    ax.set_title('Quadruped Gaits: Speed vs Energy Trade-off',
                 fontsize=14, fontweight='bold', pad=15)
    ax.grid(True, alpha=0.3)
    ax.set_facecolor('white')

    plt.tight_layout()
    out = os.path.join(OUT_DIR, 'gait_speed_energy.png')
    plt.savefig(out, dpi=150, bbox_inches='tight', facecolor='#fafafa')
    plt.close(fig)
    size_kb = os.path.getsize(out) / 1024
    print(f"  ✅ gait_speed_energy.png ({size_kb:.0f} KB)")


if __name__ == '__main__':
    print(f"📁 输出: {OUT_DIR}\n")
    print("🎨 生成漂亮的步态图（替代 ASCII）...\n")
    gen_gait_patterns()
    gen_speed_energy()
    print("\n✨ 完成！")

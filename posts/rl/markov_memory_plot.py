"""Plot a constructed weather-history example; requires Matplotlib."""
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

def visualize_markov_chains():
    """改进的可视化：二阶马尔科夫链转换为一阶链"""

    # 创建图形
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    fig.suptitle('Transforming Second-Order Markov Chain to First-Order',
                fontsize=16, fontweight='bold', y=0.95)

    # ========== 左图：二阶马尔科夫链 ==========
    ax1 = axes[0]
    ax1.set_title('Second-Order Markov Chain\n(Needs memory of past 2 days)',
                 fontsize=14, fontweight='bold', pad=20)
    ax1.set_xlim(-1, 11)
    ax1.set_ylim(-1, 11)
    ax1.axis('off')

    # 时间线标签
    time_labels = ['Day -2', 'Day -1', 'Today']
    times = [2, 5, 8]

    # 绘制时间线
    ax1.plot([1.5, 8.5], [8, 8], 'k-', linewidth=2, alpha=0.7)

    for i, (time, label) in enumerate(zip(times, time_labels)):
        ax1.text(time, 8.3, label, ha='center', fontsize=11,
                fontweight='bold', color='darkblue')
        # 时间点标记
        ax1.plot(time, 8, 'ko', markersize=10)

    # 天气示例：Sunny, Sunny, Cloudy
    weather_sequence = ['Sunny', 'Sunny', 'Cloudy']
    weather_icons = {'Sunny': '☀️', 'Cloudy': '☁️', 'Rainy': '🌧️'}

    for i, (time, weather) in enumerate(zip(times, weather_sequence)):
        ax1.text(time, 7.5, weather_icons[weather], fontsize=40, ha='center')
        ax1.text(time, 7.0, weather, ha='center', fontsize=12,
                fontweight='bold', color='darkblue')

    # 依赖箭头
    ax1.arrow(times[0], 6.8, times[1]-times[0]-0.3, -0.7,
              head_width=0.15, head_length=0.2, fc='red', ec='red', alpha=0.7)
    ax1.arrow(times[1], 6.8, times[2]-times[1]-0.3, -0.7,
              head_width=0.15, head_length=0.2, fc='red', ec='red', alpha=0.7)

    # 状态空间说明
    state_space_box = FancyBboxPatch((1, 3), 8, 2,
                                    boxstyle="round,pad=0.3",
                                    facecolor="lightcoral", alpha=0.2,
                                    edgecolor="red", linewidth=1.5)
    ax1.add_patch(state_space_box)

    ax1.text(5, 4.5, 'State Space Size Problem',
            ha='center', fontsize=12, fontweight='bold', color='darkred')
    ax1.text(5, 3.8, '3 weather types → 3² = 9 possible history pairs',
            ha='center', fontsize=10, color='darkred')
    ax1.text(5, 3.3, 'P(Today | Yesterday, Day-before-yesterday)',
            ha='center', fontsize=10, color='darkred', style='italic')

    # 内存需求
    memory_box = FancyBboxPatch((1, 0.5), 8, 1.5,
                               boxstyle="round,pad=0.3",
                               facecolor="lightblue", alpha=0.2,
                               edgecolor="blue", linewidth=1.5)
    ax1.add_patch(memory_box)

    ax1.text(5, 1.7, 'Memory Requirement',
            ha='center', fontsize=12, fontweight='bold', color='darkblue')
    ax1.text(5, 1.0, 'Need to remember last 2 states',
            ha='center', fontsize=10, color='darkblue')

    # ========== 右图：转换为一阶链 ==========
    ax2 = axes[1]
    ax2.set_title('Equivalent First-Order Markov Chain\n(Composite states)',
                 fontsize=14, fontweight='bold', pad=20)
    ax2.set_xlim(-1, 11)
    ax2.set_ylim(-1, 11)
    ax2.axis('off')

    # 复合状态的概念
    composite_box = FancyBboxPatch((1, 8), 8, 2,
                                  boxstyle="round,pad=0.3",
                                  facecolor="lightgreen", alpha=0.2,
                                  edgecolor="green", linewidth=2)
    ax2.add_patch(composite_box)

    ax2.text(5, 9.5, 'Composite State = Memory Encoded in State Name',
            ha='center', fontsize=12, fontweight='bold', color='darkgreen')

    ax2.text(5, 8.8, 'Instead of: P(Weather_today | Weather_yesterday, Weather_day-before)',
            ha='center', fontsize=9, color='darkgreen')
    ax2.text(5, 8.2, 'We use: P(Composite_today | Composite_yesterday)',
            ha='center', fontsize=9, color='darkgreen')

    # 复合状态示例
    ax2.text(3, 7.0, 'Composite State Example:', ha='left', fontsize=11, fontweight='bold')

    # 状态分解图示
    # 复合状态
    comp_state = FancyBboxPatch((3, 5.5), 4, 1,
                               boxstyle="round,pad=0.3",
                               facecolor="lightblue", alpha=0.3)
    ax2.add_patch(comp_state)
    ax2.text(5, 6.0, '"Sunny-Sunny"', ha='center', fontsize=12,
            fontweight='bold', color='darkblue')
    ax2.text(5, 5.5, 'represents: Sunny yesterday + Sunny day-before',
            ha='center', fontsize=9, color='blue')

    # 箭头到天气
    ax2.arrow(5, 5.2, 0, -1, head_width=0.2, head_length=0.15,
              fc='purple', ec='purple', alpha=0.7, linestyle='--')

    # 对应天气
    weather_box = FancyBboxPatch((3, 3.5), 4, 1,
                                boxstyle="round,pad=0.3",
                                facecolor="yellow", alpha=0.2)
    ax2.add_patch(weather_box)
    ax2.text(5, 4.0, 'Actual Weather Today', ha='center', fontsize=10, fontweight='bold')
    ax2.text(5, 3.5, 'Sunny', ha='center', fontsize=14, fontweight='bold', color='darkorange')
    ax2.text(5, 3.5, '☀️', fontsize=30, ha='center')

    # 状态转移示例
    ax2.text(3, 2.5, 'State Transition Example:', ha='left', fontsize=11, fontweight='bold')

    # 转移箭头
    ax2.plot([2, 8], [2, 2], 'k-', linewidth=1, alpha=0.5)

    # 从状态 (Sunny, Sunny)
    state1_box = FancyBboxPatch((1.5, 1.2), 2, 1,
                               boxstyle="round,pad=0.3",
                               facecolor="lightblue", alpha=0.4)
    ax2.add_patch(state1_box)
    ax2.text(2.5, 1.7, 'State: (S,S)', ha='center', fontsize=10, fontweight='bold')

    # 转移箭头
    ax2.arrow(3.5, 1.7, 2, 0, head_width=0.15, head_length=0.2,
              fc='red', ec='red', alpha=0.7)

    # 到状态 (Sunny, Cloudy)
    state2_box = FancyBboxPatch((5.5, 1.2), 2, 1,
                               boxstyle="round,pad=0.3",
                               facecolor="lightblue", alpha=0.4)
    ax2.add_patch(state2_box)
    ax2.text(6.5, 1.7, 'State: (S,C)', ha='center', fontsize=10, fontweight='bold')

    # 转移概率说明
    prob_text = 'Transition Probability:\nP((S,C) | (S,S)) = 0.4'
    ax2.text(6.5, 0.5, prob_text, ha='center', fontsize=9,
            bbox=dict(boxstyle="round,pad=0.2", facecolor="yellow", alpha=0.3))

    # 关键优势
    advantage_box = FancyBboxPatch((1, -0.5), 8, 1.5,
                                  boxstyle="round,pad=0.3",
                                  facecolor="gold", alpha=0.2,
                                  edgecolor="orange", linewidth=1.5)
    ax2.add_patch(advantage_box)

    ax2.text(5, 0.0, 'Key Advantage: Standard Markov techniques apply',
            ha='center', fontsize=11, fontweight='bold', color='darkorange')
    ax2.text(5, -0.5, 'State space: 9 composite states instead of complex memory',
            ha='center', fontsize=9, color='darkorange')

    plt.tight_layout()
    plt.show()

visualize_markov_chains()

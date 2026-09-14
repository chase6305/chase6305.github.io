#!/usr/bin/env python3
"""Regenerate editable diagram companions and the exact timing figure; standard library only."""
from pathlib import Path
import json, re, html
bundle=Path(__file__).resolve().parent
(bundle/'assets').mkdir(exist_ok=True)
font='system-ui, Noto Sans CJK SC, sans-serif'
def svg_start(title,desc,height):
    return f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1100 {height}" role="img" aria-labelledby="title desc"><title id="title">{title}</title><desc id="desc">{desc}</desc><defs><marker id="arrow" markerWidth="9" markerHeight="9" refX="7" refY="3" orient="auto"><path d="M0,0 L0,6 L8,3 Z" fill="#245b9e"/></marker></defs><rect width="1100" height="{height}" rx="22" fill="#fffef9"/><g font-family="{font}" fill="#17243c">'
def text(x,y,value,size=23,bold=False):
    return f'<text x="{x}" y="{y}" font-size="{size}" font-weight="{700 if bold else 400}">{html.escape(value)}</text>'
def box(x,y,w,h,heading,lines,fill='#edf5ff'):
    s=f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="16" fill="{fill}" stroke="#8daed2"/>'
    s+=text(x+20,y+36,heading,24,True)
    for i,line in enumerate(lines):s+=text(x+20,y+73+i*31,line,21)
    return s
def arrow(x1,y1,x2,y2):
    return f'<path d="M{x1},{y1} L{x2},{y2}" fill="none" stroke="#245b9e" stroke-width="3" marker-end="url(#arrow)"/>'
s=svg_start('数据来源与监督目标','三类来源经不同处理形成机器人动作、人体动作或时序与空间监督。',670)
s+=text(35,48,'先确认标签是什么，再选择训练目标',30,True)
rows=[(95,'真实机器人',['VR / 主从 / 自主交互','图像 + 命令 + 状态'],'执行与时间契约',['坐标、单位、控制模式','目标命令与反馈分离'],'具身动作监督',['VLA / 模仿学习','动作条件世界模型']),
(265,'人类示范',['UMI / 追踪 / 普通视频','工具、人体或仅 RGB'],'显式映射或表征学习',['姿态、计划、潜在动作','缺失量与估计量单独标记'],'分层或联合学习',['可迁移先验 + 机器人适配','无动作时序预训练']),
(435,'仿真与生成环境',['源示范 / 自动执行','场景资产 + 模拟器'],'执行与质量验证',['物理、传感器、任务检查','保留源示范与派生关系'],'交互与空间监督',['策略 / 动作后果预测','空间重建与场景扩展'])]
for y,a,al,b,bl,c,cl in rows:
    s+=box(35,y,300,140,a,al)+arrow(341,y+70,385,y+70)+box(395,y,310,140,b,bl,'#eee8f8')+arrow(711,y+70,755,y+70)+box(765,y,300,140,c,cl,'#fff2df')
s+=text(35,623,'人体姿态 ≠ 机器人命令；相机条件 ≠ 机器人动作条件',23,True)
s+='</g></svg>'
(bundle/'assets/supervision-map.svg').write_text(s)
s=svg_start('空间资产到训练轨迹','视觉重建以后还要定义物理、控制和任务，再执行并对照真机验证。',720)
s+=text(35,48,'看起来像真实场景，还需要哪些步骤？',30,True)
s+=box(35,90,480,135,'1  视觉与空间输入',['照片 / 视频 / 多视角 / 相机轨迹','重建或生成几何与外观'])
s+=box(585,90,480,135,'2  可交互环境',['碰撞、质量、摩擦、关节与接触','区分视觉背景和可交互对象'],'#eee8f8')
s+=arrow(522,155,575,155)+arrow(825,232,825,282)
s+=box(585,295,480,140,'3  执行接口与任务定义',['控制频率、执行器、传感器与延迟','初始状态、成功条件、失败与重置'],'#eee8f8')
s+=arrow(575,365,525,365)
s+=box(35,295,480,140,'4  在环境中实际执行',['人工示范 / 规划器 / 策略 rollout','记录命令、状态转移与观察'],'#edf5ff')
s+=arrow(275,442,275,492)
s+=box(35,505,480,140,'5  数据检查与训练',['同步、字段有效性、派生关系','分组划分后训练与消融'],'#fff2df')
s+=arrow(522,575,575,575)
s+=box(585,505,480,140,'6  独立真机评测',['检查预测、规划与任务执行表现','将失败反馈到环境与数据设计'],'#fff2df')
s+=text(35,687,'视觉一致性、物理可信度和闭环成功率分别验收。',22,True)
s+='</g></svg>'
(bundle/'assets/scene-to-rollout.svg').write_text(s)

s=svg_start('观测采集、到达与决策时刻','决策时只能使用已经到达的观测；训练时预测的未来动作可以作为目标。',650)
s+=text(35,48,'未来观测不能偷看；未来动作可以是训练目标',29,True)
# Exact synthetic times from data_contract_lab.py, not hardware measurements.
x=lambda milliseconds: 250+(milliseconds-980)*6.5
for ms in (980,1000,1020,1040,1060,1080,1100):
 s+=text(x(ms)-21,99,f'{ms/1000:.3f}',18)
 s+=f'<path d="M{x(ms)},115 V570" stroke="#dce4ed" stroke-width="1"/>'
s+=text(35,99,'公共时钟 / 秒',21,True)
s+=text(35,189,'图像 A',24,True)+text(35,221,'可以使用',21)
s+=arrow(x(1000),195,x(1020),195)
s+=f'<circle cx="{x(1000)}" cy="195" r="8" fill="#2d8044"/><circle cx="{x(1020)}" cy="195" r="8" fill="#2d8044"/>'
s+=text(x(1000)-40,164,'采集',20)+text(x(1020)-20,242,'到达',20)
s+=text(35,324,'图像 B',24,True)+text(35,356,'尚未到达',21)
s+=arrow(x(1025),330,x(1080),330)
s+=f'<circle cx="{x(1025)}" cy="330" r="8" fill="#ce7b18"/><circle cx="{x(1080)}" cy="330" r="8" fill="#ce7b18"/>'
s+=text(x(1025)-25,301,'采集',20)+text(x(1080)-20,376,'到达',20)
s+=f'<path d="M{x(1040)},118 V570" stroke="#17243c" stroke-width="3" stroke-dasharray="9 7"/>'
s+=text(x(1040)-69,417,'决策 t = 1.040',20,True)
s+=text(35,473,'训练动作块',24,True)+text(35,509,'未来目标',21)
for k,ms in enumerate((1040,1060,1080)):
 s+=f'<rect x="{x(ms)}" y="447" width="126" height="82" rx="8" fill="#eee8f8" stroke="#b8abd6"/>'
 s+=text(x(ms)+22,482,f'a[{k}]',22,True)
 s+=text(x(ms)+16,513,f'{ms/1000:.3f}',18)
s+=text(35,608,'A 的观测年龄 = 1.040 − 1.000 = 40 ms；到达后等待时间是另一回事。',23,True)
s+='</g></svg>'
(bundle/'assets/observation-time.svg').write_text(s)

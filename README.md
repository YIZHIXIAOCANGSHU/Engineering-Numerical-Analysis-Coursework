# MuJoCo + Rerun 六自由度机械臂避障数值实验

本项目实现一个可复现的六自由度机械臂避障路径规划实验平台：MuJoCo 负责三维仿真与碰撞检测，Rerun 负责交互式实验显示，Matplotlib 导出 LaTeX 可引用图表。

## 快速开始

```bash
python3 -m arm_planning run-demo --no-viewer
python3 -m arm_planning run-experiments --trials 2
python3 -m arm_planning export-plots
xelatex main_robot_arm.tex
```

默认使用内置 `simple_ur5e.xml` 后备模型。若要使用 MuJoCo Menagerie UR5e：

```bash
python3 scripts/fetch_menagerie.py
```

然后在 `configs/scenes.yaml` 中把 `model_xml` 指向 `third_party/mujoco_menagerie/universal_robots_ur5e/scene.xml`，并按模型实际 site 名称调整 `ee_site`。

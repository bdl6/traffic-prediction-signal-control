基于 Graph WaveNet 与 MA2C 多智能体的交通预测与信号控制
项目简介
本项目以成都市一环天府广场周边路网为仿真研究对象，实现「多尺度交通流量预测 + 区域协同智能信号控制」完整闭环系统，配套论文附录 B 全部 SUMO 工程文件，可直接复现实验全部流程与结果。
时空预测：基于 Graph WaveNet 实现 5～60 分钟多尺度交通流量预测，捕捉路网时空耦合特征
信号优化：基于 MA2C 多智能体强化学习，实现多路口协同动态配时，解决固定配时僵化问题
仿真验证：基于 SUMO + TraCI 完成微观交通仿真，对比传统固定配时与 MA2C 优化配时的效果差异
工程配套：包含完整 SUMO 路网、车流、配置文件及运行程序，可直接运行仿真
项目结构（完整可上传 GitHub，对应论文附录 A/B）
plaintext
traffic-prediction-signal-control/
├── data/                          # 基础数据文件夹
│   ├── traffic_data.csv           # 交通流训练/测试数据（用于Graph WaveNet训练）
├── models/                        # 核心算法代码（论文附录A）
│   ├── data_process.py            # 数据预处理、时空序列构建
│   ├── graph_wavenet.py           # Graph WaveNet多尺度预测模型
│   ├── ma2c_agent.py              # MA2C多智能体信号控制算法
│   └── sumo_interface.py          # SUMO仿真交互接口（状态采集/信号控制）
├── sumo_project/                  # SUMO工程文件（论文附录B）
│   ├── road.net.xml               # 成都天府广场周边路网文件（复刻真实拓扑）
│   ├── traffic.rou.xml            # 高峰时段仿真车流文件（贴合本地出行特征）
│   ├── run.sumocfg                # 仿真总配置文件（参数统一配置）
│   ├── fixed_timing.py            # 固定配时运行程序（对照实验）
│   └── optimized_timing.py        # MA2C优化配时运行程序（实验组）
├── train.py                       # Graph WaveNet预测模型训练脚本
├── test.py                        # MA2C+SUMO联合仿真测试脚本
├── requirements.txt               # 运行环境依赖清单
└── README.md                      # 项目说明、运行教程（本文档）
环境安装
基础依赖（算法 + 仿真）
bash
运行
pip install -r requirements.txt
SUMO 安装（必装，用于交通仿真）
下载地址：https://sumo.dlr.de/docs/Downloads.php
安装后配置系统环境变量（确保命令行可运行 sumo 或 sumo-gui）
验证：命令行输入 sumo --version，显示版本号即安装成功（推荐版本 1.15+）
运行步骤（完整复现论文实验）
步骤 1：训练 Graph WaveNet 流量预测模型
bash
运行
python train.py
运行后会自动加载 data/traffic_data.csv 数据，完成模型训练
训练完成后，模型权重保存为 wavenet_best.pth，用于后续信号优化的流量预测
步骤 2：运行对照实验（固定配时）
bash
运行
cd sumo_project
python fixed_timing.py
启动 SUMO 仿真，采用固定配时方案，仿真时长 10800s（3 小时，模拟早高峰）
仿真过程中实时采集平均等待时间、平均车速、平均停车次数等指标，结束后输出全局统计结果
步骤 3：运行实验组（MA2C 优化配时）
bash
运行
cd sumo_project
python optimized_timing.py
启动 SUMO 仿真，对接 MA2C 多智能体算法与 Graph WaveNet 预测模型
实时根据车流状态与预测流量，动态调整信号灯配时，实现区域协同控制
仿真结束后输出优化后的各项指标，与固定配时结果对比，验证优化效果
步骤 4：查看仿真可视化（可选）
将 sumo/run.sumocfg 文件用 SUMO-GUI 打开，可直观查看路网运行状态、车流变化、信号灯切换过程，对比固定配时与优化配时的拥堵差异。
实验场景与指标
实验场景
仿真区域：成都市一环天府广场核心周边路网（sumo_project/road.net.xml）
仿真时长：10800s（3 小时），完整模拟早高峰（7:30-10:30）密集车流场景
对照方案：传统固定配时（fixed_timing.py）VS MA2C 动态协同配时（optimized_timing.py）
核心评价指标
平均车辆等待时间（s）：表征路口通行延误，数值越小越好
平均行驶车速（km/h）：表征路网畅通程度，数值越大越好
平均停车次数（次）：表征车流平稳性，数值越小越好
单车平均 CO₂排放量（g）：表征绿色交通效益，数值越小越好
注意事项
运行 SUMO 相关程序前，确保 SUMO 已安装并配置环境变量，否则会出现 TraCI 接口调用失败
若仿真报错,查看三个核心文件（road.net.xml、traffic.rou.xml、run.sumocfg）是否完整，路径是否正确
模型训练可根据电脑配置调整 train.py 中的超参数（batch_size、epoch_num 等），GPU 加速需确保 PyTorch 支持 CUDA
车流数据与路网文件可根据实际需求修改，适配不同区域的仿真场景
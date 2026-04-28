import traci
import numpy as np
import sys
sys.path.append("../")  # 导入上层models文件夹的代码
from models.ma2c_agent import MA2C
from models.graph_wavenet import GraphWaveNet
import torch

# 仿真与算法参数
TOTAL_STEPS = 10800  # 3小时仿真
AGENT_NUM = 12  # 天府广场周边12个核心交叉口智能体
STATE_DIM = 16  # 每个智能体的状态维度
ACTION_DIM = 4  # 信号灯4个相位
MODEL_PATH = "../wavenet_best.pth"  # 训练好的Graph WaveNet模型路径

def load_prediction_model(node_num=12, pre_steps=12):
    """加载Graph WaveNet流量预测模型"""
    model = GraphWaveNet(num_nodes=node_num, in_dim=1, hid_dim=32, embed_dim=10, pre_steps=pre_steps)
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()
    return model

def main():
    # 1. 加载预测模型与MA2C多智能体
    pred_model = load_prediction_model()
    ma2c_agent = MA2C(agent_num=AGENT_NUM, state_dim=STATE_DIM, action_dim=ACTION_DIM)

    # 2. 启动SUMO仿真
    sumo_cmd = ["sumo", "-c", "run.sumocfg"]
    traci.start(sumo_cmd)
    tls_ids = traci.trafficlight.getIDList()
    print(f"MA2C优化配时仿真启动，信号灯数量：{len(tls_ids)}")

    # 3. 初始化指标存储
    wait_time = []
    speed = []
    stop_count = []

    # 4. 仿真主循环
    for step in range(TOTAL_STEPS):
        traci.simulationStep()

        # 采集实时交通状态，结合预测模型，获取各路口状态
        states = []
        for tls in tls_ids:
            # 获取当前路口实时状态（排队长度+车速）
            lanes = traci.trafficlight.getLanes(tls)
            queue = [traci.lane.getLastStepHaltingNumber(lane) for lane in lanes]
            speed_lane = [traci.lane.getLastStepMeanSpeed(lane) for lane in lanes]
            state = np.array(queue + speed_lane)
            states.append(state)

        # 5. 多智能体决策，设置信号灯相位
        for i, (tls, state) in enumerate(zip(tls_ids, states)):
            # 结合流量预测结果（简化：此处用实时状态+预测趋势）
            action = ma2c_agent.get_action(i, state)
            traci.trafficlight.setPhase(tls, action)

        # 6. 采集实时评价指标
        veh_ids = traci.vehicle.getIDList()
        for veh in veh_ids:
            wait_time.append(traci.vehicle.getWaitingTime(veh))
            speed.append(traci.vehicle.getSpeed(veh))
            stop_count.append(traci.vehicle.getStopCount(veh))

        # 7. 每60步打印中间结果
        if step % 60 == 0:
            avg_wait = np.mean(wait_time) if wait_time else 0
            avg_speed = np.mean(speed) * 3.6 if speed else 0  # 转换为km/h
            avg_stop = np.mean(stop_count) if stop_count else 0
            print(f"Step: {step:5d} | 平均等待：{avg_wait:.2f}s | 平均车速：{avg_speed:.2f}km/h | 平均停车：{avg_stop:.2f}次")

    # 8. 仿真结束，输出全局指标
    avg_wait = np.mean(wait_time) if wait_time else 0
    avg_speed = np.mean(speed) * 3.6 if speed else 0
    avg_stop = np.mean(stop_count) if stop_count else 0
    print("\n========== MA2C优化配时仿真结束 ==========")
    print(f"平均等待时间：{avg_wait:.2f} s")
    print(f"平均行驶车速：{avg_speed:.2f} km/h")
    print(f"平均停车次数：{avg_stop:.2f} 次")

    traci.close()

if __name__ == "__main__":
    main()
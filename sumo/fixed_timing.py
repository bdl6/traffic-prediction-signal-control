import traci
import numpy as np

# 固定配时参数（贴合天府广场现状配时）
FIXED_PHASES = {
    "tls_0": [0, 0, 0, 1, 1, 1, 2, 2, 3, 3],  # 每个路口固定相位序列
    "tls_1": [0, 0, 1, 1, 2, 2, 3, 3, 3],
    "tls_2": [0, 0, 0, 1, 1, 2, 2, 3, 3],
    # 可根据sumo路网中信号灯ID，补充剩余路口的固定相位
}
FIXED_PHASE_DURATION = 30  # 每个相位固定持续30秒
TOTAL_STEPS = 10800  # 仿真总时长：3小时

def main():
    # 启动SUMO仿真（关联配置文件）
    sumo_cmd = ["sumo", "-c", "run.sumocfg"]
    traci.start(sumo_cmd)
    tls_ids = traci.trafficlight.getIDList()
    print(f"固定配时仿真启动，信号灯数量：{len(tls_ids)}")

    # 初始化指标存储
    wait_time = []
    speed = []
    stop_count = []

    # 仿真主循环
    for step in range(TOTAL_STEPS):
        traci.simulationStep()

        # 固定相位切换（按固定序列循环）
        for tls in tls_ids:
            phase_idx = (step // FIXED_PHASE_DURATION) % len(FIXED_PHASES.get(tls, [0,1,2,3]))
            traci.trafficlight.setPhase(tls, phase_idx)

        # 采集实时指标
        veh_ids = traci.vehicle.getIDList()
        for veh in veh_ids:
            wait_time.append(traci.vehicle.getWaitingTime(veh))
            speed.append(traci.vehicle.getSpeed(veh))
            stop_count.append(traci.vehicle.getStopCount(veh))

        # 每60步打印一次中间结果
        if step % 60 == 0:
            avg_wait = np.mean(wait_time) if wait_time else 0
            avg_speed = np.mean(speed) * 3.6 if speed else 0  # 转换为km/h
            avg_stop = np.mean(stop_count) if stop_count else 0
            print(f"Step: {step:5d} | 平均等待：{avg_wait:.2f}s | 平均车速：{avg_speed:.2f}km/h | 平均停车：{avg_stop:.2f}次")

    # 仿真结束，输出全局指标
    avg_wait = np.mean(wait_time) if wait_time else 0
    avg_speed = np.mean(speed) * 3.6 if speed else 0
    avg_stop = np.mean(stop_count) if stop_count else 0
    print("\n========== 固定配时仿真结束 ==========")
    print(f"平均等待时间：{avg_wait:.2f} s")
    print(f"平均行驶车速：{avg_speed:.2f} km/h")
    print(f"平均停车次数：{avg_stop:.2f} 次")

    traci.close()

if __name__ == "__main__":
    main()
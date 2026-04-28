import numpy as np
from models.sumo_interface import SUMOInterface
from models.ma2c_agent import MA2C

def main():
    # 仿真与算法参数
    agent_num = 12
    state_dim = 16
    action_dim = 4
    total_sim_steps = 10800
    cfg_path = "./data/tianfu_square.sumocfg"

    # 初始化SUMO仿真环境 & MA2C多智能体
    sumo_env = SUMOInterface(cfg_path)
    ma2c = MA2C(agent_num=agent_num, state_dim=state_dim, action_dim=action_dim)
    sumo_env.start()
    tls_list = sumo_env.tls_ids

    print("===== 开始MA2C多智能体信号控制仿真 =====")
    for step in range(total_sim_steps):
        sumo_env.simulation_step()
        sumo_env.collect_metrics()

        # 逐个路口获取状态、输出动作、控制信号灯
        for i, tls in enumerate(tls_list):
            state = sumo_env.get_traffic_state(tls)
            action = ma2c.get_action(i, state)
            sumo_env.set_traffic_phase(tls, action)

        # 每60步打印一次指标
        if step % 60 == 0:
            avg_wait, avg_speed, avg_stop = sumo_env.get_average_metrics()
            print(f"Step:{step:5d} | 均速:{avg_speed:.2f} km/h | 平均等待:{avg_wait:.2f} s")

    # 仿真结束，输出最终整体指标
    avg_wait, avg_speed, avg_stop = sumo_env.get_average_metrics()
    print("\n========== 仿真结束 全局指标 ==========")
    print(f"平均等待时间：{avg_wait:.2f} s")
    print(f"平均行驶车速：{avg_speed:.2f} km/h")
    print(f"平均停车次数：{avg_stop:.2f} 次")

    sumo_env.close()

if __name__ == "__main__":
    main()
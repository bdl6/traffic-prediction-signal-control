import traci
import numpy as np

class SUMOInterface:
    def __init__(self, sumocfg_path):
        self.sumocfg = sumocfg_path
        self.wait_times = []
        self.speeds = []
        self.stops = []
        self.tls_ids = []

    def start(self):
        traci.start(["sumo", "-c", self.sumocfg])
        self.tls_ids = traci.trafficlight.getIDList()

    def step(self):
        traci.simulationStep()

    def get_state(self, tls_id):
        lanes = traci.trafficlight.getLanes(tls_id)
        q = [traci.lane.getLastStepHaltingNumber(l) for l in lanes]
        s = [traci.lane.getLastStepMeanSpeed(l) for l in lanes]
        return np.array(q + s)

    def set_phase(self, tls_id, phase):
        traci.trafficlight.setPhase(tls_id, phase)

    def collect(self):
        for veh in traci.vehicle.getIDList():
            self.wait_times.append(traci.vehicle.getWaitingTime(veh))
            self.speeds.append(traci.vehicle.getSpeed(veh))
            self.stops.append(traci.vehicle.getStopCount(veh))

    def metrics(self):
        avg_wait = np.mean(self.wait_times) if self.wait_times else 0
        avg_speed = np.mean(self.speeds) if self.speeds else 0
        avg_stop = np.mean(self.stops) if self.stops else 0
        return avg_wait, avg_speed, avg_stop

    def close(self):
        traci.close()
"""
    使用RoundRobin的方式对作业进行分配
"""
import sys

import numpy as np
from utils.util import parse_next_job, heft, update_watermark, heft_update, process_extra_idle_time, time_disassembly_to_process_and_idle,writer_result
import Constant as ct


# 这里使用roundRobin+HEFT的方式对任务进行分配
# RoundRobin负责轮询数据中心和机房HEFT负责将任务分配到对应的机器上，因为对于DAG而言，RoundRobin很难分配任务
class RoundRobinDistribution:

    def __init__(self, batch_size, mode='training'):
        self.current_data_centre_num = 0
        self.current_machine_home_num = -1
        self.current_machine_num = 0
        self.data_centre_machine_status = np.zeros([ct.DATE_CENTRE_NUM, ct.NUMBER_OF_DATA_CENTER_MACHINE_ROOMS, ct.NUMBER_OF_SERVERS])
        self.single_machine_power = ct.SINGLE_MACHINE_POWER
        self.single_machine_peak_power = ct.SINGLE_MACHINE_PEAK_POWER
        self.forced_air_cooling_ratio = ct.FORCED_AIR_COOLING_RATIO
        self.forced_water_cooling_ratio = ct.FORCED_WATER_COOLING_RATIO
        self.air_cooling_peak_power = ct.AIR_COOLING_PEAK_POWER
        self.air_cooling_start_power = ct.AIR_COOLING_START_POWER
        self.water_cooling_start_power = ct.WATER_COOLING_START_POWER
        self.offset = 0
        self.current_job_info = []
        self.current_job_id = -1
        self.batch_size = batch_size
        self.watermark_increase_rate = ct.WATERMARK_INCREASE_RATE
        self.energy_type = ct.ENERGY_TYPE
        self.energy_emission_ratio = ct.ENERGY_EMISSION_RATIO
        self.data_centre_machine_home_current_total_energy = np.zeros([ct.DATE_CENTRE_NUM, ct.NUMBER_OF_DATA_CENTER_MACHINE_ROOMS])
        self.data_centre_current_total_energy = np.zeros(ct.DATE_CENTRE_NUM)
        self.data_centre_current_total_emission = np.zeros(ct.DATE_CENTRE_NUM)
        self.data_centre_machine_home_current_cooling_type = np.zeros([ct.DATE_CENTRE_NUM, ct.NUMBER_OF_DATA_CENTER_MACHINE_ROOMS])
        self.file_path = ct.JOB_FILE_PATH if mode == "training" else ct.TEST_FILE_PATH
        # watermark
        self.watermark = -1
        self.data_centre_machine_idle_state = {}
        self.data_centre_machine_process_state = {}
        self.current_total_energy = []
        self.current_total_emission = []
        # makeSpan
        self.makespan = 0


    def job_to_machine(self):
        self.initialization_status()
        for i in range(self.batch_size):
            job_info_list, self.offset = parse_next_job(file_path=self.file_path, offset=self.offset)
            # self.distribute_job(job_info_list)
            self.distribute_job_update(job_info_list)
            self.encapsulation_processing()
            self.current_total_energy.append(np.sum(self.data_centre_current_total_energy))
            self.current_total_emission.append(np.sum(self.data_centre_current_total_emission))
        while self.watermark != 0:
            # print(self.watermark)
            self.encapsulation_processing()
        return self.data_centre_current_total_energy, self.data_centre_current_total_emission, self.makespan, self.current_total_energy, self.current_total_emission

    def distribute_job(self, job_info_list):
        # 处理数据中心和机房变化的信息
        self.current_machine_home_num += 1
        if self.current_machine_home_num >= ct.NUMBER_OF_DATA_CENTER_MACHINE_ROOMS:
            self.current_data_centre_num += 1
            self.current_machine_home_num = -1
        if self.current_data_centre_num >= ct.DATE_CENTRE_NUM:
            self.current_data_centre_num = 0
        current_machine_home = self.data_centre_machine_status[self.current_data_centre_num][self.current_machine_home_num]
        data_center_machine_room_status = heft(current_machine_home, job_info_list)
        self.data_centre_machine_status[self.current_data_centre_num][self.current_machine_home_num] = data_center_machine_room_status

    def distribute_job_update(self, job_info_list):
        self.current_machine_home_num += 1
        if self.current_machine_home_num >= ct.NUMBER_OF_DATA_CENTER_MACHINE_ROOMS:
            self.current_data_centre_num += 1
            self.current_machine_home_num = 0
        if self.current_data_centre_num >= ct.DATE_CENTRE_NUM:
            self.current_data_centre_num = 0
        # ---------------------------------
        # self.current_data_centre_num = 0
        # self.current_machine_home_num = 0
        # ---------------------------------
        current_machine_home = self.data_centre_machine_status[self.current_data_centre_num][self.current_machine_home_num]
        key = str(self.current_data_centre_num) + '_' + str(self.current_machine_home_num)
        data_center_machine_room_status, self.data_centre_machine_process_state, self.data_centre_machine_idle_state, lazy = heft_update(self.data_centre_machine_process_state, self.data_centre_machine_idle_state, current_machine_home, job_info_list, key)
        self.data_centre_machine_status[self.current_data_centre_num][self.current_machine_home_num] = data_center_machine_room_status

    def computer_energy_emission(self, current_data_centre_machine_status, extra_idle_time):
        current_data_center_num = -1
        current_data_center_home_num = -1
        for data_centre in current_data_centre_machine_status:
            current_data_center_num += 1
            for home in data_centre:
                current_data_center_home_num += 1
                home_energy = 0
                machine_num = -1
                for machine in home:
                    machine_num += 1
                    disassembly_key = str(current_data_center_num) + '_' + str(current_data_center_home_num) + '_' + str(machine_num)
                    machine_process_time = []
                    machine_idle_time = []
                    if disassembly_key in self.data_centre_machine_process_state.keys():
                        machine_process_time = self.data_centre_machine_process_state[disassembly_key]
                        machine_idle_time = self.data_centre_machine_idle_state[disassembly_key]
                    if machine > 0:
                        total_process_time, total_idle_time, return_process_time, return_idle_time = time_disassembly_to_process_and_idle(working_time=self.watermark, process_time=machine_process_time, idle_time=machine_idle_time)
                        if len(return_process_time) != 0 and len(return_idle_time) != 0:
                            self.data_centre_machine_process_state[disassembly_key] = return_process_time
                            self.data_centre_machine_idle_state[disassembly_key] = return_idle_time
                        home_energy += self.single_machine_peak_power * (total_process_time / 1000000 / 60 / 60)
                        home_energy += self.single_machine_power * (total_idle_time / 1000000 / 60 / 60)
                    else:
                        working_time = self.watermark + machine
                        total_process_time, total_idle_time, return_process_time, return_idle_time = time_disassembly_to_process_and_idle(working_time, process_time=machine_process_time, idle_time=machine_idle_time)
                        if len(return_process_time) != 0 and len(return_idle_time) != 0:
                            self.data_centre_machine_process_state[disassembly_key] = return_process_time
                            self.data_centre_machine_idle_state[disassembly_key] = return_idle_time
                        idle_time = np.abs(machine) - extra_idle_time + total_idle_time
                        home_energy += self.single_machine_peak_power * (total_process_time / 1000000 / 60 / 60)
                        home_energy += self.single_machine_power * (idle_time / 1000000 / 60 / 60)
                cooling_type = self.data_centre_machine_home_current_cooling_type[current_data_center_num][
                    current_data_center_home_num]
                if cooling_type == 0:
                    peak_machine_num = np.sum(home > 0)
                    current_energy_consumption = peak_machine_num * self.single_machine_peak_power
                    if current_energy_consumption > self.air_cooling_peak_power:
                        self.data_centre_machine_home_current_cooling_type[current_data_center_num][
                            current_data_center_home_num] = 1
                        home_energy += self.water_cooling_start_power
                    home_energy += home_energy * self.forced_air_cooling_ratio
                else:
                    peak_machine_num = np.sum(home > 0)
                    current_energy_consumption = peak_machine_num * self.single_machine_peak_power
                    if current_energy_consumption < self.air_cooling_peak_power:
                        self.data_centre_machine_home_current_cooling_type[current_data_center_num][
                            current_data_center_home_num] = 0
                        home_energy += self.air_cooling_start_power
                    home_energy += home_energy * self.forced_water_cooling_ratio
                self.data_centre_machine_home_current_total_energy[current_data_center_num][current_data_center_home_num] += home_energy
            current_data_center_home_num = -1
            self.data_centre_current_total_energy[current_data_center_num] = np.sum(self.data_centre_machine_home_current_total_energy[current_data_center_num])
            self.data_centre_current_total_emission[current_data_center_num] = self.data_centre_current_total_energy[current_data_center_num] * self.energy_emission_ratio[self.energy_type[current_data_center_num]]


    def process_extra_idle_time(self, current_data_centre_machine_status):
        min_idle_time = sys.maxsize
        for date_centre in current_data_centre_machine_status:
            for home in date_centre:
                greater_than_zero = home > 0
                greater_than_zero_num = np.sum(greater_than_zero)
                if greater_than_zero_num > 0:
                    return 0
                else:
                    idle_time = np.min(np.abs(home))
                    if idle_time < min_idle_time:
                        min_idle_time = idle_time
        return min_idle_time

    def completion_processing_time_and_lead_time(self):
        data_centre_num = self.current_data_centre_num
        machine_room_num = self.current_machine_home_num
        max_length = -sys.maxsize
        for machine_num in range(ct.NUMBER_OF_SERVERS):
            key = str(data_centre_num) + '_' + str(machine_room_num) + '_' + str(machine_num)
            if key in self.data_centre_machine_process_state.keys():
                element_num = len(self.data_centre_machine_process_state[key])
                if max_length < element_num:
                    max_length = element_num

        for current_data_centre_num in range(ct.DATE_CENTRE_NUM):
            for current_machine_room_num in range(ct.NUMBER_OF_DATA_CENTER_MACHINE_ROOMS):
                for current_machine_num in range(ct.NUMBER_OF_SERVERS):
                    key = str(current_data_centre_num) + '_' + str(current_machine_room_num) + '_' + str(current_machine_num)
                    if key in self.data_centre_machine_process_state.keys():
                        process_len = len(self.data_centre_machine_process_state[key])
                    else:
                        process_len = 0
                    # idle_len = len(self.data_centre_machine_idle_state[key])
                    for i in range(max_length-process_len):
                        if key in self.data_centre_machine_process_state.keys():
                            self.data_centre_machine_process_state[key].append(0)
                            self.data_centre_machine_idle_state[key].append(0)
                        else:
                            self.data_centre_machine_process_state[key] = [0]
                            self.data_centre_machine_idle_state[key] = [0]

    def time_disassembly_to_process_and_idle(self, working_time, disassembly_key):
        total_process_time = 0
        total_idle_time = 0
        if working_time == 0:
            return total_process_time, total_idle_time
        process_time = self.data_centre_machine_process_state[disassembly_key]
        idle_time = self.data_centre_machine_idle_state[disassembly_key]
        # if working_time != 0:
        #     print(process_time)
        #     print(idle_time)
        #     print(working_time)
        #     print("-------------------------------------------")
        # 下一个指针
        next_hand = 0
        while working_time != 0:
            mid_idle_time = working_time - idle_time[next_hand]
            if mid_idle_time > 0:
                total_idle_time += idle_time[next_hand]
                working_time -= idle_time[next_hand]
            else:
                total_idle_time += working_time
                result_idle_time = idle_time[next_hand]
                time_remaining = result_idle_time - working_time
                idle_time[next_hand] = time_remaining
                break

            idle_time[next_hand] = 0
            mid_process_time = working_time - process_time[next_hand]
            if mid_process_time > 0:
                total_process_time += process_time[next_hand]
                working_time -= process_time[next_hand]
                process_time[next_hand] = 0
                process_time.pop(next_hand)
                idle_time.pop(next_hand)
                next_hand -= 1
            else:
                total_process_time += working_time
                result_process_time = process_time[next_hand]
                time_remaining = result_process_time - working_time
                process_time[next_hand] = time_remaining
                break
            next_hand += 1
        self.data_centre_machine_process_state[disassembly_key] = process_time
        self.data_centre_machine_idle_state[disassembly_key] = idle_time
        return total_process_time, total_idle_time

    def encapsulation_processing(self):
        current_data_centre_machine_status, self.watermark = update_watermark(self.data_centre_machine_status,self.watermark_increase_rate)
        self.makespan += self.watermark
        idle_time = process_extra_idle_time(current_data_centre_machine_status)
        self.computer_energy_emission(current_data_centre_machine_status, idle_time)
        self.data_centre_machine_status = (current_data_centre_machine_status > 0) * np.abs(current_data_centre_machine_status)

    def initialization_status(self):
        self.data_centre_machine_status = np.zeros([ct.DATE_CENTRE_NUM, ct.NUMBER_OF_DATA_CENTER_MACHINE_ROOMS, ct.NUMBER_OF_SERVERS])
        self.data_centre_machine_home_current_total_energy = np.zeros([ct.DATE_CENTRE_NUM, ct.NUMBER_OF_DATA_CENTER_MACHINE_ROOMS])
        self.data_centre_current_total_energy = np.zeros(ct.DATE_CENTRE_NUM)
        self.data_centre_current_total_emission = np.zeros(ct.DATE_CENTRE_NUM)
        self.data_centre_machine_home_current_cooling_type = np.zeros([ct.DATE_CENTRE_NUM, ct.NUMBER_OF_DATA_CENTER_MACHINE_ROOMS])
        self.data_centre_machine_idle_state = {}
        self.data_centre_machine_process_state = {}
        self.current_data_centre_num = 0
        self.current_machine_home_num = -1
# env = RoundRobinDistribution(10)
# for i in range(20):
#     data_centre_current_total_energy, data_centre_current_total_emission = env.job_to_machine()
#     print(env.offset)

# data_centre_current_total_energy, data_centre_current_total_emission = RoundRobinDistribution(14).job_to_machine()
# print(data_centre_current_total_energy, data_centre_current_total_emission)


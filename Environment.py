"""
    Environment是交互的环境，我们这个环境主要便是多个云数据中心的机器，这里的设计是特别的重要的，联盟云的核心就在这里了。
    环境的预设置：
        云数据中心个数的设置：安装我开题报告中的绘图，我这里预设置有6个云数据中心。不同的云数据中心我们采用不同的能源供应。
        机房的设置：我们也将每个云数据中心设置为7个机房，每个机房的物理尺寸为9.6m*8.4m*3.6m。
        机架的设置：每个机房是包含两行，每行是有五个42U机架。
        机箱设置：每个机架包括五个机箱。
        服务器设置：每个机箱有二十个7U服务器。
        冷却方式的设置：每个云数据中心均采用两种冷却方式，分别为风冷和水冷。
        任务个数的设置：这个要根据洗出来的任务的个数来确定。
    单个机房服务器数量：10。
    单个云数据中心的服务器的数量：70。
    整个联盟云中数据中心的个数：420。

    与环境交互之后返回的结果的设置：
    当前每个服务器的状态，得到的reword。
    reword的设置：本次任务的分配所得到的总的功耗+总排放+成本
    这个reword之所以要加上成本的主要的目的是，联盟云和单云之家的差距就在这里，如果不设置成本会导致
"""
import random

import gym
import numpy as np
import torch
from gym.spaces import Box, Dict
from utils.util import parse_next_job, heft, update_watermark, heft_update, process_extra_idle_time, time_disassembly_to_process_and_idle, writer_result
import Constant as ct
import random


class FederationCloudAndTaskEnv(gym.Env):

    def __init__(self, mode="training"):
        # 定义环境是一个离散的空间,因为我的环境并不是一个连续的环境
        self.observation_space = Dict
        # 定义动作空间是一个图
        self.action_space = "Graph"
        self.data_centre_machine_status = np.zeros([ct.DATE_CENTRE_NUM, ct.NUMBER_OF_DATA_CENTER_MACHINE_ROOMS, ct.NUMBER_OF_SERVERS])
        # 因为所有的数据中心采用的机器是一样的，所以我们这里定义单台机器的静态功耗kw/h
        self.single_machine_power = ct.SINGLE_MACHINE_POWER
        # 因为所有的数据中心采用的机器是一样的，所以我们这里定义单台机器的峰值功耗kw/h
        self.single_machine_peak_power = ct.SINGLE_MACHINE_PEAK_POWER
        self.forced_air_cooling_ratio = ct.FORCED_AIR_COOLING_RATIO
        self.forced_water_cooling_ratio = ct.FORCED_WATER_COOLING_RATIO
        self.air_cooling_peak_power = ct.AIR_COOLING_PEAK_POWER
        self.air_cooling_start_power =ct.AIR_COOLING_START_POWER
        self.water_cooling_start_power = ct.WATER_COOLING_START_POWER
        self.offset = 0
        self.total_data_center_num = ct.DATE_CENTRE_NUM
        # 作业存储的路径
        self.file_path = ct.JOB_FILE_PATH if mode == "training" else ct.TEST_FILE_PATH
        # 当前的作业信息-->存储的是整个作业才分成的任务的数组
        self.current_job_info, self.offset = parse_next_job(self.file_path, self.offset)
        # 当前作业的id-->之所以记录job_id是为了偏移量的处理，因为只有在job_id不同的时候才表示一个job的任务取出完毕，才可以更新偏移量
        self.current_job_id = -1
        '''
            时间语义watermark: 这里引入flink中术语watermark是因为，当前模拟的任务分配的环境是没有时间的概念的，
            任务的推进无法根据处理时间来进行推进,所以这里使用watermark来推进时间。
            当前我的环境是有6个云数据中心，watermark的更新策略如下：可以看思路分析图中的图解，也可以看下方的文字描述：
                            机房1 --> 最短等待时间1  |
            数据中心1        机房k --> 最短等待时间2   |
                            机房7 --> 最短等待时间3    |
                            
                            机房1 --> 最短等待时间4    |
            数据中心2        机房k --> 最短等待时间5     |
                            机房7 --> 最短等待时间6     |
                            
                            机房1 --> 最短等待时间7     |
            数据中心3        机房k --> 最短等待时间8      |         ===> watermark = min(最短等待时间1,最短等待时间2,....,最短等待时间18) * 1.25
                            机房7 --> 最短等待时间9       |
                            
                            机房1 --> 最短等待时间10       |
            数据中心4        机房k --> 最短等待时间11       |
                            机房7 --> 最短等待时间12       |
                            
                            机房1 --> 最短等待时间13        |
            数据中心5        机房k --> 最短等待时间14          |
                            机房7 --> 最短等待时间15       |
                                
                            机房1 --> 最短等待时间16        |
            数据中心6        机房k --> 最短等待时间17        |
                            机房7 --> 最短等待时间18        |
            
            watermark的推进只在一个作业分配完成之后进行推进。具体的原因是，我们认为任务分配到云数据中心的过程是属于抉择的过程。
            在任务没分配之前都是在抉择，我们的作业的到来是按照顺序到来的，也就意味着，任务现在是在一个FIFO队列中，当前任务不被分配是
            不会分配下一个任务的。
        '''
        self.watermark = -1
        self.is_there_another_job = True
        self.watermark_increase_rate = ct.WATERMARK_INCREASE_RATE
        self.energy_type = ct.ENERGY_TYPE
        self.energy_type_code = self.energy_code()
        self.energy_emission_ratio = ct.ENERGY_EMISSION_RATIO
        self.energy_emission_ratio_code = ct.ENERGY_EMISSION_RATIO_CODE
        self.data_centre_machine_home_current_total_energy = np.zeros([ct.DATE_CENTRE_NUM, ct.NUMBER_OF_DATA_CENTER_MACHINE_ROOMS])
        self.data_centre_current_total_energy = np.zeros(ct.DATE_CENTRE_NUM)
        self.data_centre_current_total_emission = np.zeros(ct.DATE_CENTRE_NUM)
        self.data_centre_machine_home_current_cooling_type = np.zeros([ct.DATE_CENTRE_NUM, ct.NUMBER_OF_DATA_CENTER_MACHINE_ROOMS])
        self.data_centre_machine_idle_state = {}
        self.data_centre_machine_process_state = {}
        self.max_sub_task_num = 0
        self.max_sub_task_running_time = 0
        self.current_data_centre_num = -1
        # 记录makeSpan
        self.makespan = 0

    # 做出动作，环境的反应
    def step(self, action):

        done = False
        data_center_num = self.current_data_centre_num
        home_num = action
        if home_num != ct.NUMBER_OF_DATA_CENTER_MACHINE_ROOMS:
            key = str(data_center_num) + '_' + str(home_num)
            data_center_machine_room_status = self.data_centre_machine_status[data_center_num][home_num]
            data_center_machine_room_status, self.data_centre_machine_process_state, self.data_centre_machine_idle_state, lazy = heft_update(data_centre_machine_process_state=self.data_centre_machine_process_state, data_centre_machine_idle_state=self.data_centre_machine_idle_state, data_center_machine_room_status=data_center_machine_room_status, current_job_info=self.current_job_info, pre_key=key)
            # writer_result(ct.RL_TIME, lazy)
            self.data_centre_machine_status[data_center_num][home_num] = data_center_machine_room_status
            self.current_job_info, self.offset = parse_next_job(file_path=self.file_path, offset=self.offset)
            current_data_centre_machine_status, self.watermark = update_watermark(self.data_centre_machine_status, self.watermark_increase_rate)
            self.makespan += self.watermark
            idle_time = process_extra_idle_time(current_data_centre_machine_status)
            writer_result(ct.DELAY_TIME,idle_time)
            self.computer_energy_emission(current_data_centre_machine_status, idle_time)
            self.data_centre_machine_status = (current_data_centre_machine_status > 0) * np.abs(current_data_centre_machine_status)
            if np.size(self.current_job_info) == 0:
                # 如果已经是没有了后续的任务，将不再更新watermark,根据等待时间直接计算能耗和排放
                done = True
            # self.current_data_centre_num += 1
        return self.get_observation(), np.sum(self.data_centre_current_total_energy), np.sum(self.data_centre_current_total_emission), done

    # 重置环境后看到的是什么
    def reset(self):
        self.data_centre_machine_status = np.zeros([ct.DATE_CENTRE_NUM, ct.NUMBER_OF_DATA_CENTER_MACHINE_ROOMS, ct.NUMBER_OF_SERVERS])
        self.data_centre_machine_home_current_total_energy = np.zeros([ct.DATE_CENTRE_NUM, ct.NUMBER_OF_DATA_CENTER_MACHINE_ROOMS])
        self.data_centre_current_total_energy = np.zeros(ct.DATE_CENTRE_NUM)
        self.data_centre_current_total_emission = np.zeros(ct.DATE_CENTRE_NUM)
        self.data_centre_machine_home_current_cooling_type = np.zeros([ct.DATE_CENTRE_NUM, ct.NUMBER_OF_DATA_CENTER_MACHINE_ROOMS])
        self.data_centre_machine_idle_state = {}
        self.data_centre_machine_process_state = {}
        self.current_data_centre_num = -1
        self.offset = 0
        self.current_job_info, self.offset = parse_next_job(self.file_path, self.offset)
        return self.get_observation()

    # 画图用的
    def render(self, mode="human"):
        pass

    def heft(self, data_center_machine_room_status):
        tasks = np.array(self.current_job_info)
        job_start_time = np.min(tasks[:, 2])
        for task in tasks:
            task_start_time = tasks[2]
            the_most_suitable_machine_index = np.argmin(data_center_machine_room_status)
            the_most_suitable_machine_value = data_center_machine_room_status[the_most_suitable_machine_index]
            if task_start_time == job_start_time:
                data_center_machine_room_status[the_most_suitable_machine_index] += task[3]
            else:
                lateTime = task_start_time - job_start_time
                current_task_most_suitable_time = the_most_suitable_machine_value + lateTime
                time_proximity_of_all_machines = current_task_most_suitable_time - data_center_machine_room_status
                abs_time_proximity_of_all_machines = np.abs(time_proximity_of_all_machines)
                proximity_index = np.argmin(abs_time_proximity_of_all_machines)
                machine_time = data_center_machine_room_status[proximity_index]
                if current_task_most_suitable_time >= machine_time:
                    data_center_machine_room_status[proximity_index] += task[3]
                else:
                    data_center_machine_room_status += abs_time_proximity_of_all_machines[proximity_index] + task[3]
        return data_center_machine_room_status

    # 切换到下一个作业
    def parse_next_job(self):
        self.current_job_info = []
        with open(self.file_path, "r") as task_data:
            task_data.seek(self.offset)
            while self.is_there_another_job:
                sub_task = task_data.readline().strip()
                if len(sub_task) == 0:
                    self.is_there_another_job = False
                    break
                sub_task_info = [int(i) for i in sub_task.split(",")]
                if self.current_job_id == -1:
                    self.current_job_id = sub_task_info[0]
                if self.current_job_id == sub_task_info[0]:
                    self.current_job_info.append(sub_task_info)
                    self.offset = task_data.tell()
                else:
                    break

    # 计算watermark更新数据中心的状态
    def update_watermark(self):
        self.watermark = np.min(self.data_centre_machine_status) * (1 + self.watermark_increase_rate)
        current_data_centre_machine_status = self.data_centre_machine_status - self.watermark
        return current_data_centre_machine_status

    # 为能源编码
    def energy_code(self):
        energy_type_index = ct.ENERGY_TYPE_INDEX
        energy_type_code = np.zeros([len(self.energy_type), len(self.energy_type)])
        for name in self.energy_type:
            position_and_value = energy_type_index[name]
            energy_type_code[position_and_value[0], position_and_value[1]-1] = 1
        return energy_type_code


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
                        # home_energy += self.single_machine_peak_power * (self.watermark / 1000000 / 60 / 60)
                        home_energy += self.single_machine_peak_power * (total_process_time / 1000000 / 60 / 60)
                        home_energy += self.single_machine_power * (total_idle_time / 1000000 / 60 / 60)
                    else:
                        working_time = self.watermark + machine
                        # total_process_time, total_idle_time = self.time_disassembly_to_process_and_idle(working_time, disassembly_key)
                        total_process_time, total_idle_time, return_process_time, return_idle_time = time_disassembly_to_process_and_idle(working_time=working_time, process_time=machine_process_time, idle_time=machine_idle_time)
                        if len(return_process_time) != 0 and len(return_idle_time) != 0:
                            self.data_centre_machine_process_state[disassembly_key] = return_process_time
                            self.data_centre_machine_idle_state[disassembly_key] = return_idle_time
                        # idle_time = np.abs(machine) - extra_idle_time
                        idle_time = np.abs(machine) - extra_idle_time + total_idle_time
                        # home_energy += self.single_machine_peak_power * (working_time / 1000000 / 60 / 60)
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
                self.data_centre_machine_home_current_total_energy[current_data_center_num][
                    current_data_center_home_num] += home_energy
            current_data_center_home_num = -1
            self.data_centre_current_total_energy[current_data_center_num] = np.sum(
                self.data_centre_machine_home_current_total_energy[current_data_center_num])
            self.data_centre_current_total_emission[current_data_center_num] = self.data_centre_current_total_energy[current_data_center_num] * self.energy_emission_ratio[self.energy_type[current_data_center_num]]

    def computer_job_feature(self):
        if len(self.current_job_info) == 0:
            return 0, 0
        number_of_task = len(self.current_job_info)
        maximum_execution_time = max(np.array(self.current_job_info)[:, 3])
        if number_of_task > self.max_sub_task_num:
            self.max_sub_task_num = number_of_task
        if maximum_execution_time > self.max_sub_task_running_time:
            self.max_sub_task_running_time = maximum_execution_time
        number_of_task = 1 if number_of_task >= self.max_sub_task_num else number_of_task / self.max_sub_task_num
        maximum_execution_time = 1 if maximum_execution_time >= self.max_sub_task_running_time else maximum_execution_time / self.max_sub_task_running_time
        return number_of_task, maximum_execution_time

    def computer_data_centre_and_machine_home_feature(self):

        waiting_time = np.zeros([ct.DATE_CENTRE_NUM, ct.NUMBER_OF_DATA_CENTER_MACHINE_ROOMS, 2])
        current_data_centre = -1
        current_home = -1
        for data_centre in self.data_centre_machine_status:
            current_data_centre += 1
            for home in data_centre:
                current_home += 1
                min_waiting_time = min(home)
                max_waiting_time = max(home)
                waiting_time[current_data_centre][current_home][0] = min_waiting_time
                waiting_time[current_data_centre][current_home][1] = max_waiting_time
            current_home = -1
        waiting_time = waiting_time / (np.max(waiting_time, axis=0) + 10e-100)
        return waiting_time

    # 计算状态
    def get_observation(self):
        self.current_data_centre_num += 1
        if self.current_data_centre_num >= ct.DATE_CENTRE_NUM:
            self.current_data_centre_num = 0

        number_of_task, maximum_execution_time = self.computer_job_feature()

        waiting_time = self.computer_data_centre_and_machine_home_feature()
        waiting_time = waiting_time[self.current_data_centre_num]
        data_centre_machine_home_current_total_energy = self.data_centre_machine_home_current_total_energy / (np.max(self.data_centre_machine_home_current_total_energy) + 10e-100)
        data_centre_machine_home_current_total_energy = data_centre_machine_home_current_total_energy[self.current_data_centre_num]
        data_centre_current_total_energy = self.data_centre_current_total_energy / (np.max(self.data_centre_current_total_energy) + 10e-100)
        data_centre_current_total_energy = data_centre_current_total_energy[self.current_data_centre_num]
        data_centre_current_total_emission = self.data_centre_current_total_emission / (np.max(self.data_centre_current_total_emission) + 10e-100)
        data_centre_current_total_emission = data_centre_current_total_emission[self.current_data_centre_num]

        return torch.cat((
            torch.FloatTensor(waiting_time).reshape(1, 14),
            torch.FloatTensor(data_centre_machine_home_current_total_energy).reshape(-1, 7),
            torch.FloatTensor(self.data_centre_machine_home_current_cooling_type[self.current_data_centre_num]).reshape(-1, 7),
            torch.FloatTensor(np.array(data_centre_current_total_energy, dtype=np.float32)).reshape(-1, 1),
            torch.FloatTensor(np.array(data_centre_current_total_emission, dtype=np.float32)).reshape(-1, 1),
            torch.FloatTensor(self.energy_type_code[self.current_data_centre_num]).reshape(-1, 6),
            torch.FloatTensor(self.energy_emission_ratio_code[self.current_data_centre_num].astype(np.float32)).reshape(-1, 1),
            torch.FloatTensor(np.array(number_of_task,dtype=np.float32)).reshape(-1, 1),
            torch.FloatTensor(np.array(maximum_execution_time, dtype=np.float32)).reshape(-1, 1)
        ), dim=1)

    # 一批次结束之后将数据中心的数据任务都清空
    def encapsulation_processing(self):
        while self.watermark != 0:
            current_data_centre_machine_status, self.watermark = update_watermark(self.data_centre_machine_status,
                                                                                  self.watermark_increase_rate)
            self.makespan += self.watermark

            idle_time = process_extra_idle_time(current_data_centre_machine_status)
            self.computer_energy_emission(current_data_centre_machine_status, idle_time)
            self.data_centre_machine_status = (current_data_centre_machine_status > 0) * np.abs(current_data_centre_machine_status)
        return self.data_centre_current_total_energy, self.data_centre_current_total_emission, self.makespan







import os
import time

import numpy as np
import random
import torch
from roundrobin import RoundRobin
import Environment
import Constant as ct


def writer_result(file_path_writer, value):
    with open(file_path_writer, 'a') as file_object:
        rl_info = str(value)+"\n"
        file_object.write(rl_info)

# 判断当前数据集中有多少个DAG
def number_of_extraction_jobs(file_path):
    offset = 0
    current_job_id = -1
    job_num = 0
    is_there_other_job = True
    iter_num = 0
    with open(file_path, "r") as task_data:
        task_data.seek(offset)
        while is_there_other_job:
            iter_num += 1
            sub_task = task_data.readline().strip()
            if len(sub_task) == 0:
                is_there_other_job = False
                break
            sub_task_info = [int(i) for i in sub_task.split(",")]
            if current_job_id == -1 or current_job_id != sub_task_info[0]:
                job_num += 1
                current_job_id = sub_task_info[0]
    return job_num


if __name__ == '__main__':
    model = torch.load('G:\pycharmWorkeSapce3\RL_BasedTaskSchedulingForEnvironmentallSustatinableFCC\model_load\\actor\\1679969567.337538.pth')
    model.eval()
    env = Environment.FederationCloudAndTaskEnv(mode="test")
    # 提取出当前测试数据中的DAG的个数
    job_num = number_of_extraction_jobs(ct.TEST_FILE_PATH)
    # 初始化环境的状态
    observation = env.reset()
    # batch size
    batch_size = job_num
    # 计算round robin
    round_robin_heft = RoundRobin.RoundRobinDistribution(batch_size, mode="test")
    rrf_data_centre_current_total_energy, rrf_data_centre_current_total_emission, rrf_makespan, _, _ = round_robin_heft.job_to_machine()
    # 任务的编号
    task_num = 0
    # 是否发生变化
    is_change = False
    for _ in range(1):
        task_num = 0
        # 是否发生变化
        is_change = False
        env.reset()
        while task_num < batch_size:
            is_change = False
            start_time = time.time()
            # 得到每个数据中心，每个机房的概率
            action_probability = model(observation)
            # 从概率中抉择出是哪一个数据中心的那一台机器action_probability = [数据中心1的1号机房的概率，数据中心1的2号机房的概率，。。。。，数据中心6的6号机房的概率]
            action = random.choices(range(8), weights=action_probability[0], k=1)[0]
            if action != ct.NUMBER_OF_DATA_CENTER_MACHINE_ROOMS:
                is_change = True
                task_num += 1
            observation, rl_current_total_energy, rl_current_total_emission, done = env.step(action)
            end_time = time.time()
            # writer_result(ct.DELAY_TIME, end_time - start_time)
            # 计算reward
            if done:
                # TODO Calculate discounted reward
                rl_data_centre_current_total_energy, rl_data_centre_current_total_emission, makespan = env.encapsulation_processing()
                rl_total_energy = np.sum(rl_data_centre_current_total_energy)
                rl_total_emission = np.sum(rl_data_centre_current_total_emission)
                # TODO Estimate advantage
                rrf_total_energy = np.sum(rrf_data_centre_current_total_energy)
                rrf_total_emission = np.sum(rrf_data_centre_current_total_emission)
                current_energy = rl_total_energy
                current_emission = rl_total_emission
                energy_reward = (rrf_total_energy - rl_total_energy) / rrf_total_energy
                dif_energy_reward = rrf_total_energy - rl_total_energy
                emission_reward = (rrf_total_emission - rl_total_emission) / rrf_total_emission
                dif_emission_reward = rrf_total_emission - rl_total_emission
                print("RL Energy: ", rl_total_energy, " RL Emission: ", rl_total_emission)
                print("RRF Energy: ", rrf_total_energy, " RRF Emission: ", rrf_total_emission)
                print("Energy radio: ", dif_energy_reward, " Emission radio: ", dif_emission_reward)
                print("rrf_makeSpan",rrf_makespan,"rl_makespan",makespan)
                print("===============================================")







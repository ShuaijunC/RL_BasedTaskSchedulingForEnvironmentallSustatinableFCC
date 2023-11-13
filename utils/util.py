import random
import sys

import numpy as np


def parse_next_job(file_path, offset):
    current_job_info = []
    current_job_id = -1
    with open(file_path, "r") as task_data:
        task_data.seek(offset)
        while True:
            sub_task = task_data.readline().strip()
            if len(sub_task) == 0:
                if np.size(current_job_info) == 0:
                    offset = 0
                break
            sub_task_info = [int(i) for i in sub_task.split(",")]
            if current_job_id == -1:
                current_job_id = sub_task_info[0]
            if current_job_id == sub_task_info[0]:
                current_job_info.append(sub_task_info)
                offset = task_data.tell()
            else:
                break
    return current_job_info, offset


def heft(data_center_machine_room_status, current_job_info):
    tasks = np.array(current_job_info)
    job_start_time = np.min(tasks[:, 2])
    the_most_suitable_machine_index = np.argmin(data_center_machine_room_status)
    the_most_suitable_machine_value = data_center_machine_room_status[the_most_suitable_machine_index]
    for task in tasks:
        task_start_time = task[2]
        if task_start_time == job_start_time:
            cur_the_most_suitable_machine_index = np.argmin(data_center_machine_room_status)
            data_center_machine_room_status[cur_the_most_suitable_machine_index] += task[3]
        else:
            lateTime = task_start_time - job_start_time
            current_task_most_suitable_time = the_most_suitable_machine_value + lateTime
            time_proximity_of_all_machines = current_task_most_suitable_time - data_center_machine_room_status
            abs_time_proximity_of_all_machines = np.abs(time_proximity_of_all_machines)
            proximity_index = np.argmin(abs_time_proximity_of_all_machines)
            machine_time = data_center_machine_room_status[proximity_index]
            if current_task_most_suitable_time >= machine_time:
                data_center_machine_room_status[proximity_index] += abs_time_proximity_of_all_machines[proximity_index] + task[3]
            else:
                data_center_machine_room_status[proximity_index] += task[3]

    return data_center_machine_room_status


def heft_update(data_centre_machine_process_state, data_centre_machine_idle_state, data_center_machine_room_status, current_job_info, pre_key):
    tasks = np.array(current_job_info)
    job_start_time = np.min(tasks[:, 2])
    the_most_suitable_machine_index = np.argmin(data_center_machine_room_status)
    the_most_suitable_machine_value = data_center_machine_room_status[the_most_suitable_machine_index]
    for task in tasks:
        task_start_time = task[2]
        if task_start_time == job_start_time:
            cur_the_most_suitable_machine_index = np.argmin(data_center_machine_room_status)
            data_center_machine_room_status[cur_the_most_suitable_machine_index] += task[3]
            key = pre_key + '_' + str(cur_the_most_suitable_machine_index)
            if key in data_centre_machine_process_state.keys():
                data_centre_machine_process_state[key].append(task[3])
            else:
                data_centre_machine_process_state[key] = [task[3]]
            if key in data_centre_machine_idle_state.keys():
                data_centre_machine_idle_state[key].append(0)
            else:
                data_centre_machine_idle_state[key] = [0]
        else:
            lateTime = task_start_time - job_start_time
            current_task_most_suitable_time = the_most_suitable_machine_value + lateTime
            time_proximity_of_all_machines = current_task_most_suitable_time - data_center_machine_room_status
            abs_time_proximity_of_all_machines = np.abs(time_proximity_of_all_machines)
            proximity_index = np.argmin(abs_time_proximity_of_all_machines)
            machine_time = data_center_machine_room_status[proximity_index]
            if current_task_most_suitable_time >= machine_time:
                data_center_machine_room_status[proximity_index] += abs_time_proximity_of_all_machines[proximity_index] + task[3]
                key = pre_key + '_' + str(proximity_index)
                if key in data_centre_machine_idle_state.keys():
                    data_centre_machine_idle_state[key].append(abs_time_proximity_of_all_machines[proximity_index])
                else:
                    data_centre_machine_idle_state[key] = [abs_time_proximity_of_all_machines[proximity_index]]
                if key in data_centre_machine_process_state.keys():
                    data_centre_machine_process_state[key].append(task[3])
                else:
                    data_centre_machine_process_state[key] = [task[3]]
            else:
                data_center_machine_room_status[proximity_index] += task[3]
                key = pre_key + '_' + str(proximity_index)
                if key in data_centre_machine_process_state.keys():
                    data_centre_machine_process_state[key].append(task[3])
                else:
                    data_centre_machine_process_state[key] = [task[3]]
                if key in data_centre_machine_idle_state.keys():
                    data_centre_machine_idle_state[key].append(0)
                else:
                    data_centre_machine_idle_state[key] = [0]

    return data_center_machine_room_status,data_centre_machine_process_state, data_centre_machine_idle_state, the_most_suitable_machine_value


def update_watermark(data_centre_machine_status, watermark_increase_rate, is_done=False):
    min_value = sys.maxsize
    max_value = -1
    for data_centre in data_centre_machine_status:
        for machine_home in data_centre:
            current_max_value = np.max(machine_home)
            is_zero = machine_home == 0
            zero_to_sys_max_value = is_zero * sys.maxsize
            element_superposition = machine_home + zero_to_sys_max_value
            current_min_value = np.min(element_superposition)
            if current_min_value < min_value:
                min_value = current_min_value
            if current_max_value > max_value:
                max_value = current_max_value

    if min_value == sys.maxsize:
        min_value = 0
    watermark = round(min_value * (1 + watermark_increase_rate))
    if is_done:
        watermark = max_value
    # print("watermark: ", watermark)
    # watermark = random.randint(min_value, max_value)
    current_data_centre_machine_status = data_centre_machine_status - watermark
    return current_data_centre_machine_status, watermark


def process_extra_idle_time(current_data_centre_machine_status):
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


def time_disassembly_to_process_and_idle(working_time, process_time, idle_time):
    total_process_time = 0
    total_idle_time = 0
    if working_time == 0:
        return total_process_time, total_idle_time, process_time, idle_time
    next_hand = 0
    while working_time != 0:
        try:
            mid_idle_time = working_time - idle_time[next_hand]
        except:
            print(process_time)
            print(idle_time)
            print(working_time)
        if mid_idle_time > 0:
            total_idle_time += idle_time[next_hand]
            working_time -= idle_time[next_hand]
        else:
            total_idle_time += working_time
            result_idle_time = idle_time[next_hand]
            time_remaining = result_idle_time - working_time
            idle_time[next_hand] = time_remaining
            working_time = 0
            # 跳出
            break

        idle_time[next_hand] = 0

        mid_process_time = working_time - process_time[next_hand]
        if mid_process_time > 0:
            total_process_time += process_time[next_hand]
            working_time -= process_time[next_hand]
            # process_time的0号元素
            # process_time.pop(0)
            process_time[next_hand] = 0
            # 删除当前的元素
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
    # self.data_centre_machine_process_state[disassembly_key] = process_time
    # self.data_centre_machine_idle_state[disassembly_key] = idle_time
    return total_process_time, total_idle_time, process_time, idle_time

# 写入延迟时间
def writer_result(file_path_writer, value):
    with open(file_path_writer, 'a') as file_object:
        rl_info = str(value)+"\n"
        file_object.write(rl_info)




# path = "../data/part-r-00001"
# current_offset = 0
#
# for i in range(3):
#     job_info_list, current_offset = parse_next_job(path, current_offset)
#     print(job_info_list)



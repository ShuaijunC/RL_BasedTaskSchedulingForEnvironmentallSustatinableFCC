import argparse
import Environment
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import model
import FederationCloudAgent
parser = argparse.ArgumentParser()
import Constant as ct
import torch


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


batch_size = number_of_extraction_jobs(ct.JOB_FILE_PATH)

# 单台机器的静态功耗
parser.add_argument('--single_machine_power', type=float, default=0.15, help='power consumption in standby mode')
# 单台机器的峰值功耗
parser.add_argument('--single_machine_peak_power', type=float, default=0.25, help='peak power of single machine')
# 风冷能耗与计算能耗的比率
parser.add_argument('--forced_air_cooling_ration', type=float, default=0.0005, help='ratio of air cooling energy consumption to calculated energy consumption')
# 水冷能耗与计算能耗的比率
parser.add_argument('--force_water_cooling_ration', type=float, default=0.2, help='ratio of water cooling to calculated energy consumption')
# 风冷的峰值功率
parser.add_argument('--air_cooling_peak_power', type=int, default=200, help='peak power consumption of air cooling')
# 风冷的启动功率
parser.add_argument('--air_cooling_start_power', type=float, default=350 * 0.15, help='starting power of air cooling')
# 水冷的启动功率
parser.add_argument('--water_cooling_start_power', type=float, default=1000 * 0.15, help='starting power of water cooling')
# 模型参数的额存储路径
parser.add_argument('--model_path', type=str, default='none', help='path to load model')
# 随机数种子
parser.add_argument('--seed', type=int, default=666, help='Random seed.')
# 学习率
parser.add_argument('--lr', type=float, default=10 ** -2, help='learning rate')
# 随机种子
parser.add_argument('--eps', type=float, default=10 ** -1, help='Random seed.')
# 采用的优化器是rms
parser.add_argument('--optimizer', type=str, default='adam', help='sgd or adam or rms')
# 调度器采用的是lambda
parser.add_argument('--scheduler', type=str, default='lambda', help='lambda or cyclic')
# 循环调度器中每一步的大小
parser.add_argument('--step_up', type=float, default=100, help='step_size_up for cyclic scheduler')
# 迭代次数
parser.add_argument('--epochs', type=float, default=10000, help='Iterations')
# 批次大小
parser.add_argument('--batch_size', type=int, default=batch_size, help='A batch of data')
# 输入层维度
parser.add_argument('--input_dim', type=int, default=39, help='input the latitude of the layer')
# 输入层维度
parser.add_argument('--output_dim', type=int, default=8, help='output the latitude of the layer')
# 输入层维度
parser.add_argument('--hidden_dim', type=int, default=128, help='Latitude of hidden layer')

args = parser.parse_args()
config_enhanced = vars(args)

print("batch_size: ", config_enhanced['batch_size'])

# writer = SummaryWriter('runs')
env = Environment.FederationCloudAndTaskEnv("training")
env.reset()
actor_model = model.FederationCloudModel(args.input_dim, args.hidden_dim, args.output_dim)
critic_model = model.FederationCloudModelTD(args.input_dim, args.hidden_dim, args.output_dim)
# actor_model = torch.load('D:\\reforcemrntLearning\ReinforcementLearningForFederationCloudTaskToEnergyConservationMinMinVersionReconsitutionR\ReinforcementLearningForFederationCloudTaskToEnergyConservationMinMinVersionReconsitution\model_load\\actor\\final1679495272.900926.pth')
# critic_model = torch.load('D:\\reforcemrntLearning\ReinforcementLearningForFederationCloudTaskToEnergyConservationMinMinVersionReconsitutionR\ReinforcementLearningForFederationCloudTaskToEnergyConservationMinMinVersionReconsitution\model_load\critic\\final1679495272.9099262.pth')
a2c = FederationCloudAgent.FederationCloudAgent(env, actor_model, critic_model, config_enhanced)
a2c.training()


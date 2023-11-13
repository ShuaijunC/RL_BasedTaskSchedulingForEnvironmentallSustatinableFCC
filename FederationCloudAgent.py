import itertools
import sys
import time
import numpy as np
import os
import random
from copy import deepcopy
import torch
import torch.nn.functional as F
from torch import optim
from torch.optim.lr_scheduler import CyclicLR, LambdaLR
from collections import deque
from roundrobin import RoundRobin
import Constant as ct


# 固定随机数种子
def make_seed(seed):
    np.random.seed(seed=seed)
    torch.manual_seed(seed=seed)


def save_model(actor_path, actor_model, critic_path, critic_model, training_final=False):
    print("save model")
    # actor_save = os.path.join(str(self.writer.get_logdir()), 'model{}.pth'.format(self.random_id))
    actor_save = actor_path + "\\"
    if training_final:
        actor_save += "final"
    actor_save = actor_save + str(time.time()) + ".pth"
    print(actor_save)
    torch.save(actor_model, actor_save)
    critic_save = critic_path + "\\"
    if training_final:
        critic_save += "final"
    critic_save = critic_save + str(time.time()) + ".pth"
    torch.save(critic_model, critic_save)


def save_model_info(actor_path, actor_model, rl_total_energy, rl_total_emission, rrf_total_energy, rrf_total_emission, is_epoch):
    if is_epoch:
        actor_save = actor_path + "\\"
        actor_save = actor_save + str(time.time()) + ".pth"
        torch.save(actor_model, actor_save)
        with open(ct.EPOCH_RESULT, 'a') as file_object:
            rl_info = "RL Energy: " + str(rl_total_energy) + " RL Emission: " + str(rl_total_emission) + "\n"
            rrf_info = "RRF Energy: " + str(rrf_total_energy) + " RRF Emission: " + str(rrf_total_emission) + "\n"
            file_object.write(rl_info)
            file_object.write(rrf_info)
            file_object.write("model_path: "+actor_save+"\n")
            file_object.write("---------------------------------------------------\n")
    else:
        actor_save = actor_path + "\\"
        actor_save = actor_save + str(time.time()) + ".pth"
        torch.save(actor_model, actor_save)
        with open(ct.BEST_RESULT, 'a') as file_object:
            rl_info = "RL Energy: " + str(rl_total_energy) + " RL Emission: " + str(rl_total_emission) + "\n"
            rrf_info = "RRF Energy: " + str(rrf_total_energy) + " RRF Emission: " + str(rrf_total_emission) + "\n"
            file_object.write(rl_info)
            file_object.write(rrf_info)
            file_object.write("model_path: "+actor_save+"\n")
            file_object.write("---------------------------------------------------\n")



# 将结果写入到文件
def writer_result(file_path_writer, rl_total_energy, rl_total_emission, rrf_total_energy, rrf_total_emission):
    with open(file_path_writer, 'a') as file_object:
        rl_info = "RL Energy: "+str(rl_total_energy)+" RL Emission: "+str(rl_total_emission) + "\n"
        rrf_info = "RRF Energy: "+str(rrf_total_energy)+" RRF Emission: "+str(rrf_total_emission) + "\n"
        file_object.write(rl_info)
        file_object.write(rrf_info)
        file_object.write("---------------------------------------------------\n")


def process_reward(reward):
    for t in reversed(range(len(reward))):
        if t - 1 != -1:
            reward[t - 1] = reward[t]
    return reward


def get_advantages(deltas):
    advantages = []


    s = 0.0
    for delta in deltas[::-1]:
        s = 0.98 * 0.95 * s + delta
        advantages.append(s)


    advantages.reverse()
    return advantages


use_cuda = False
if use_cuda:
    device = torch.device('cuda')
    print("using GPU")
else:
    device = torch.device('cpu')
    print("using CPU")


class FederationCloudAgent:
    def __init__(self, env, actor_model, critic_model, config, writer=None):
        self.env = env
        self.config = config
        make_seed(config['seed'])  # 固定随机数种子
        self.actor_model = actor_model.to(device)
        self.critic_model = critic_model
        self.random_id = str(np.random.randint(0, 9, 10)).replace(' ', '_')
        if config["model_path"] is not None and config["model_path"] != 'none':
            # self.network.load_state_dict(torch.load(config['model_path']))
            self.network = torch.load(config['model_path'])
        # Their optimizers
        if config['optimizer'] == "sgd":
            self.optimizer = optim.SGD(self.network.parameters(), config['lr'])
        elif config['optimizer'] == 'adam':
            self.optimizer = optim.Adam(self.actor_model.parameters(), lr=1e-3)
            self.optimizer_td = optim.Adam(self.critic_model.parameters(), lr=1e-2)
        else:
            # self.optimizer = optim.RMSprop(self.network.parameters(), config['lr'], eps=config['eps'])
            self.optimizer = optim.Adam(self.actor_model.parameters(), lr=config['lr'])
            self.optimizer_td = optim.Adam(self.critic_model.parameters(), lr=config['lr'])
        self.writer = writer

        if config['scheduler'] == 'cyclic':
            ratio = config['sched_ratio']
            self.scheduler = CyclicLR(self.optimizer, base_lr=config['lr'] / ratio, max_lr=config['lr'] * ratio,
                                      step_size_up=config['step_up'])
        elif config['scheduler'] == 'lambda':
            lambda2 = lambda epoch: 0.99 ** epoch
            self.scheduler = LambdaLR(self.optimizer, lr_lambda=[lambda2])  # 调整学习率，
        else:
            self.scheduler = None
        self.round_robin_helf = RoundRobin.RoundRobinDistribution(self.config['batch_size'])

        self.loss_fn = torch.nn.MSELoss()

        self.actor_path = ct.ACTOR_PATH
        self.critic_path = ct.CRITIC_PATH
        self.result_save_path = ct.RESULT_SAVE_PATH
    def training(self):

        epochs = self.config['epochs']

        batch_size = self.config['batch_size']

        epoch = 0

        best_energy = sys.maxsize
        best_emission = sys.maxsize
        current_energy = 0
        current_emission = 0
        # 计算Round robin + HELF 关于任务分配所得到的能耗和排放
        rrf_data_centre_current_total_energy, rrf_data_centre_current_total_emission, rrf_makespan, rrf_current_total_energy, rrf_current_total_emission = self.round_robin_helf.job_to_machine()
        n_step = 0
        for epoch in range(epochs):
            observation = self.env.reset()
            states = []
            rewards = []
            actions = []
            next_states = []
            overs = []
            probs = []
            task_num = 0
            # 是否发生变化
            is_change = False
            while task_num < batch_size:
                is_change = False
                n_step += 1
                # 追加状态
                states.append(observation.numpy()[0])
                # 得到每个数据中心，每个机房的概率
                action_probability = self.actor_model(observation)
                # 从概率中抉择出是哪一个数据中心的那一台机器action_probability = [数据中心1的1号机房的概率，数据中心1的2号机房的概率，。。。。，数据中心6的6号机房的概率]
                action = random.choices(range(self.config['output_dim']), weights=action_probability[0], k=1)[0]  # 之所以使用这种方式进行抉择，是通过随机性来防止模型每次的抉择出现僵化
                if action != ct.NUMBER_OF_DATA_CENTER_MACHINE_ROOMS:
                    is_change = True
                    task_num += 1
                probs.append(action_probability[0][action])
                # 计算当前抉择的是哪一个数据中心: 0,1,2,3,.....,41
                actions.append(action)
                observation, rl_current_total_energy, rl_current_total_emission, done = self.env.step(action)
                next_states.append(observation.numpy()[0])
                # 计算reward
                if done:
                    # TODO Calculate discounted reward
                    rl_data_centre_current_total_energy, rl_data_centre_current_total_emission, makespan = self.env.encapsulation_processing()
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
                    print("Best Energy: ", best_energy, " Best Emission: ", best_emission)
                    print("Energy radio: ", dif_energy_reward, " Emission radio: ", dif_emission_reward)
                    print("=====================", epoch, "==========================")
                    writer_result(self.result_save_path, rl_total_energy, rl_total_emission, rrf_total_energy, rrf_total_emission)
                    # reward = energy_reward + emission_reward
                    reward = (energy_reward + emission_reward) * 100
                    rewards.append(reward)
                    # if dif_energy_reward < 0 or dif_emission_reward < 0:
                    #     rewards.append(-100.0)
                    # else:
                    #     rewards.append(100.0)
                    rewards = process_reward(rewards)
                    overs.append(True)

                    if current_emission < best_emission and current_energy < best_energy:
                        save_model(actor_path=self.actor_path, actor_model=self.actor_model, critic_path=self.critic_path, critic_model=self.critic_model)
                        save_model_info(actor_path=ct.BEST_ACTOR_PATH,actor_model=self.actor_model,rl_total_energy=rl_total_energy, rl_total_emission=rl_total_emission,rrf_total_energy=rrf_total_energy,rrf_total_emission=rrf_total_emission,is_epoch=False)
                        best_energy = current_energy
                        best_emission = current_emission
                    if epoch % 10 == 0:
                        save_model_info(actor_path=ct.EPOCH_PATH, actor_model=self.actor_model, rl_total_energy=rl_total_energy, rl_total_emission=rl_total_emission,rrf_total_energy=rrf_total_energy,rrf_total_emission=rrf_total_emission,is_epoch=True)
                else:
                    rewards.append(0)
                    overs.append(done)

            # 转换states,rewards,actions,next_states,overs为tensor

            states = torch.FloatTensor(states).reshape(-1, self.config['input_dim'])
            rewards = torch.FloatTensor(rewards).reshape(-1, 1)
            actions = torch.LongTensor(actions).reshape(-1, 1)
            next_states = torch.FloatTensor(next_states).reshape(-1, self.config['input_dim'])
            overs = torch.LongTensor(overs).reshape(-1, 1)

            values = self.critic_model(states)
            # for n in range(batch_size):
            #     values.append(self.critic_model(states[n]))
            # values = torch.tensor(values, requires_grad=True)
            targets = self.critic_model(next_states) * 0.98
            # for n in range(batch_size):
            #     targets.append(self.critic_model(next_states[n]) * 0.98)
            # targets = torch.tensor(targets, requires_grad=True)
            # targets *= (1 - overs)

            targets = targets + rewards

            deltas = (targets - values).squeeze(dim=1).tolist()
            advantages = get_advantages(deltas)
            advantages = torch.FloatTensor(advantages).reshape(-1, 1)
            # for n in range(batch_size):
            #     probs.append(self.actor_model(states[n]).reshape(42))
            # probs = torch.tensor([item.cpu().detach().numpy() for item in probs], requires_grad=True)
            #
            # probs = probs.gather(dim=1, index=actions)
            probs = torch.tensor(probs, requires_grad=True).float()
            probs = probs.detach()

            for _ in range(100):
                new_probs = self.actor_model(states)
                new_probs = new_probs.gather(dim=1, index=actions)
                new_probs = new_probs

                ratios = new_probs / probs
                surr1 = ratios * advantages
                surr2 = torch.clamp(ratios, 0.8, 1.2) * advantages
                loss = -torch.min(surr1, surr2)
                loss = loss.mean()
                values = self.critic_model(states)
                loss_td = self.loss_fn(values, targets)

                self.optimizer.zero_grad()
                loss.backward(retain_graph=True)
                self.optimizer.step()

                self.optimizer_td.zero_grad()
                loss_td.backward(retain_graph=True)
                self.optimizer_td.step()
        # 整个训练都结束时保存模型
        save_model(actor_path=self.actor_path, actor_model=self.actor_model, critic_path=self.critic_path, critic_model=self.critic_model, training_final=True)










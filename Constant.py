"""
    本文件记录的是一下常量数据
"""
import numpy as np
# 数据中心的个数
DATE_CENTRE_NUM = 6
# 每个数据中心机房的个数
NUMBER_OF_DATA_CENTER_MACHINE_ROOMS = 7
# 每个机房服务器的数量
NUMBER_OF_SERVERS = 10
# 单台机器的待机能耗
SINGLE_MACHINE_POWER = 0.15
# 因为所有的数据中心采用的机器是一样的，所以我们这里定义单台机器的峰值功耗kw/h
SINGLE_MACHINE_PEAK_POWER = 0.5
# 每个机房有一套自己的独立的散射系统，这个系统分成两部分风冷和水冷
# 单个机房风冷与机房能耗的比率
FORCED_AIR_COOLING_RATIO = 0.0005
# 单个机房水冷与机房能耗的比率
FORCED_WATER_COOLING_RATIO = 0.2
# 风冷的峰值功率kw/h
AIR_COOLING_PEAK_POWER = NUMBER_OF_SERVERS * SINGLE_MACHINE_POWER + NUMBER_OF_SERVERS * 0.05
# 风冷的启动功率kw/h
AIR_COOLING_START_POWER = NUMBER_OF_SERVERS * 0.35 * SINGLE_MACHINE_POWER
# 水冷的启动功率kw/h
WATER_COOLING_START_POWER = NUMBER_OF_SERVERS * SINGLE_MACHINE_POWER
# 能源类型
ENERGY_TYPE = ["fire_energy", "solar_energy", "wind_energy", "water_energy", "hydrogen_energy", "nuclear_energy"]
# 排放比
ENERGY_EMISSION_RATIO = {"fire_energy": 0.6, "solar_energy": 0.3, "wind_energy": 0.1, "water_energy": 0.3, "hydrogen_energy": 0.2, "nuclear_energy": 0.3}
# 比率
RATIO = [0.6, 0.3, 0.1, 0.3, 0.2, 0.3]
# 排放比编码
ENERGY_EMISSION_RATIO_CODE = np.array([[0.6], [0.3], [0.1], [0.3], [0.2], [0.3]])
# watermark的增长比例
WATERMARK_INCREASE_RATE = 0.25
# 能耗编码
ENERGY_TYPE_INDEX = {"fire_energy": [0, 1], "solar_energy": [1, 2], "wind_energy": [2, 3], "water_energy": [3, 4], "hydrogen_energy": [4, 5], "nuclear_energy": [5, 6]}
JOB = "\RL_BasedTaskSchedulingForEnvironmentallSustatinableFCC\data\\data-min"
ACTOR = "\RL_BasedTaskSchedulingForEnvironmentallSustatinableFCC\model_load\\actor3"
CRITIC = "\RL_BasedTaskSchedulingForEnvironmentallSustatinableFCC\model_load\critic3"
RESULT = "\RL_BasedTaskSchedulingForEnvironmentallSustatinableFCC\data\\result4.txt"
TEST = "\RL_BasedTaskSchedulingForEnvironmentallSustatinableFCC\data\\part-r-00000-test"
DELAY_PATH = "\RL_BasedTaskSchedulingForEnvironmentallSustatinableFCC\data\\remote.txt"

RL_DELAY_PATH = "\RL_BasedTaskSchedulingForEnvironmentallSustatinableFCC\data\\delay_time.txt"
RM_DELAY_PATH = "\RL_BasedTaskSchedulingForEnvironmentallSustatinableFCC\data\\rm_delayThree.txt"
ROUND_ROBIN_DELAY_PATH = "\RL_BasedTaskSchedulingForEnvironmentallSustatinableFCC\data\\rr_delayThree.txt"
SO_DELAY_PATH = "\RL_BasedTaskSchedulingForEnvironmentallSustatinableFCC\data\so_delayThree.txt"
GJO_DELAY_PATH = "\RL_BasedTaskSchedulingForEnvironmentallSustatinableFCC\data\gjo_delayThree.txt"
# 作业文件的路径

MODE = "local"
LOCAL_FILE_PATH = "G:\pycharmWorkeSapce3"
REMOTE_FILE_PATH = "D:\\reforcemrntLearning\ConvergentToPoorModel"
PRE_FILE_PATH = LOCAL_FILE_PATH if MODE != 'remote' else REMOTE_FILE_PATH
JOB_FILE_PATH = PRE_FILE_PATH + JOB
ACTOR_PATH = PRE_FILE_PATH + ACTOR
CRITIC_PATH = PRE_FILE_PATH + CRITIC
RESULT_SAVE_PATH = PRE_FILE_PATH + RESULT
TEST_FILE_PATH = PRE_FILE_PATH + TEST

EPOCH_PATH = PRE_FILE_PATH + "\RL_BasedTaskSchedulingForEnvironmentallSustatinableFCC\model_load\\epoch"
BEST_RESULT =PRE_FILE_PATH +  "\RL_BasedTaskSchedulingForEnvironmentallSustatinableFCC\data\\best_result.txt"
EPOCH_RESULT =PRE_FILE_PATH +  "\RL_BasedTaskSchedulingForEnvironmentallSustatinableFCC\data\\epoch_result.txt"
BEST_ACTOR_PATH =PRE_FILE_PATH +  "\RL_BasedTaskSchedulingForEnvironmentallSustatinableFCC\model_load\\best_actor"
DELAY_TIME = LOCAL_FILE_PATH + DELAY_PATH

RL_TIME = LOCAL_FILE_PATH + RL_DELAY_PATH
RM_TIME = LOCAL_FILE_PATH + RM_DELAY_PATH
ROUND_ROBIN_TIME = LOCAL_FILE_PATH + ROUND_ROBIN_DELAY_PATH
SO_TIME = LOCAL_FILE_PATH + SO_DELAY_PATH
GJO_TIME = LOCAL_FILE_PATH + GJO_DELAY_PATH




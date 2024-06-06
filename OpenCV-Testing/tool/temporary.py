import random
import string
import numpy as np
import cv2
import datetime
import struct

class MutationHistory:
    def __init__(self):
        self.history = []
        self.mutation_strategies = ['random', 'incremental', 'bit_flip']

    def add_record(self, param_name, original_value, mutated_value, result):
        record = {
            'param_name': param_name,
            'original_value': original_value,
            'mutated_value': mutated_value,
            'result': result,
            'timestamp': datetime.datetime.now()
        }
        self.history.append(record)

    def get_failure_rate(self, param_name):
        failures = sum(1 for record in self.history if record['param_name'] == param_name and not record['result'])
        total = sum(1 for record in self.history if record['param_name'] == param_name)
        return failures / total if total else 0

    def suggest_mutation_strategy(self, param_name):
        failure_rate = self.get_failure_rate(param_name)
        if failure_rate > 0.5:
            return random.choice(self.mutation_strategies)
        else:
            return 'bit_flip' if failure_rate > 0.3 else 'incremental'

    def perform_mutation(self, param, strategy, param_info):
        if strategy == 'random':
            return self.random_mutation(param, param_info)
        elif strategy == 'incremental':
            return self.incremental_mutation(param, param_info)
        elif strategy == 'bit_flip':
            return self.bit_flip_mutation(param, param_info)
        return param

    # Random mutation: generate a completely random value based on the type
    def random_mutation(self, param, param_info):
        if isinstance(param, int):
            return random.randint(-1000, 1000)
        elif isinstance(param, float):
            return random.uniform(-1000.0, 1000.0)
        elif isinstance(param, tuple):
            return tuple(random.randint(-100, 100) for _ in param)
        elif isinstance(param, np.ndarray):
            if param.dtype == np.uint8:
                return np.random.randint(0, 255, param.shape, dtype=np.uint8)
            elif param.dtype in [np.float32, np.float64]:
                return np.random.uniform(-1000.0, 1000.0, param.shape)
        elif isinstance(param, str):
            return ''.join(random.choices(string.ascii_letters + string.digits, k=len(param)))
        elif isinstance(param, cv2.KeyPoint):
            return cv2.KeyPoint(x=random.uniform(0, 1000), y=random.uniform(0, 1000), size=random.uniform(0, 100))
        return param

    # Incremental mutation: slightly modify the value
    def incremental_mutation(self, param, param_info):
        if isinstance(param, int):
            return param + random.randint(-10, 10)
        elif isinstance(param, float):
            return param + random.uniform(-1.0, 1.0)
        elif isinstance(param, tuple):
            return tuple(x + random.uniform(-1.0, 1.0) for x in param)
        elif isinstance(param, np.ndarray):
            if param.dtype == np.uint8:
                noise = np.random.randint(-10, 10, param.shape, dtype=np.int32)
                return np.clip(param.astype(np.int32) + noise, 0, 255).astype(np.uint8)
            elif param.dtype in [np.float32, np.float64]:
                noise = np.random.uniform(-1.0, 1.0, param.shape)
                return param + noise
        elif isinstance(param, str):
            return param.replace(random.choice(param), random.choice(string.ascii_letters), 1)
        elif isinstance(param, cv2.KeyPoint):
            return cv2.KeyPoint(x=param.pt[0] + random.uniform(-1, 1), y=param.pt[1] + random.uniform(-1, 1), size=param.size + random.uniform(-0.1, 0.1))
        return param

    # Bit flip mutation: flip the bits of the value
    def bit_flip_mutation(self, param, param_info):
        if isinstance(param, int):
            binary = format(param, '08b')
            flipped = ''.join('1' if b == '0' else '0' for b in binary)
            return int(flipped, 2)
        elif isinstance(param, float):
            binary = ''.join(format(struct.unpack('!I', struct.pack('!f', param))[0], '032b'))
            flipped = ''.join('1' if b == '0' else '0' for b in binary)
            return struct.unpack('!f', struct.pack('!I', int(flipped, 2)))[0]
        return param

# 示例使用
mutation_history = MutationHistory()

# 假设函数和测试
def test_param_success(mutated_arg):
    # 一个简单的假设测试函数
    return random.choice([True, False])

def mutate_parameters(args, params_info):
    mutated_args = []

    for arg, (param_name, param_info) in zip(args, params_info.items()):
        strategy = mutation_history.suggest_mutation_strategy(param_name)
        mutated_arg = mutation_history.perform_mutation(arg, strategy, param_info)

        # 假设有一个函数来测试这个参数是否会导致bug
        success = test_param_success(mutated_arg)
        mutation_history.add_record(param_name, arg, mutated_arg, success)
        mutated_args.append(mutated_arg)

    return mutated_args

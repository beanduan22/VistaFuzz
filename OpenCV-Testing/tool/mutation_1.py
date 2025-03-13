import pandas as pd
import numpy as np
import random
import cv2
import string
import struct
from datetime import datetime

class MutationHistory:
    def __init__(self, error_log_path):
        self.history = {}
        self.mutation_strategies = ['random', 'incremental', 'bit_flip']
        self.error_log_path = error_log_path

    def add_record(self, param_type, success, mutated_arg, strategy):
        if param_type not in self.history:
            self.history[param_type] = {'failures': 0, 'total': 0}
        self.history[param_type]['total'] += 1
        if not success:
            self.history[param_type]['failures'] += 1
            self.log_error(param_type, mutated_arg, strategy)

    def get_failure_rate(self, param_type):
        if param_type in self.history:
            data = self.history[param_type]
            return data['failures'] / data['total'] if data['total'] > 0 else 0
        return 0

    def suggest_mutation_strategy(self, param_type):
        failure_rate = self.get_failure_rate(param_type)
        if failure_rate > 0.7:
            return 'bit_flip'
        elif failure_rate > 0.5:
            return random.choice(self.mutation_strategies)
        return 'incremental' if failure_rate > 0.2 else 'random'

    def log_error(self, param_type, mutated_arg, detail):
        # 记录错误到Excel
        error_record = pd.DataFrame([{
            'param_type': param_type,
            'mutated_arg': mutated_arg,
            'timestamp': datetime.now(),
            'details': detail
        }])
        with pd.ExcelWriter(self.error_log_path, mode='a', if_sheet_exists='overlay') as writer:
            error_record.to_excel(writer, index=False, header=not writer.sheets)

# 实例化突变历史对象
mutation_history = MutationHistory("error_records.xlsx")

# 实例化突变历史对象
mutation_history = MutationHistory()
def mutate_parameters(args, params_info, mutation_history):
    mutated_arg = []

    for arg, (param_name, param_info) in zip(args, params_info.items()):
        param_type = param_info.get('type')
        strategy = mutation_history.suggest_mutation_strategy(param_type)

        if param_info.get('size') == 'numpy.ndarray':
            mutated_arg = apply_ndarray_mutation(arg, param_info, strategy)
        elif param_info.get('size') == 'tuple':
            mutated_arg = apply_tuple_mutation(arg, param_info, strategy)
        elif 'int' in param_type:
            mutated_arg = apply_int_type_mutation(arg, param_type, strategy)
        elif 'float' in param_type:
            mutated_arg = apply_float_type_mutation(arg, param_type, strategy)
        elif 'bool' in param_type:
            mutated_arg = apply_bool_type_mutation(arg, param_type, strategy)
        elif 'str' in param_type:
            mutated_arg = apply_str_type_mutation(arg, param_type, strategy)
        elif 'KeyPoint' in param_type:
            mutated_arg = apply_KeyPoint_type_mutation(arg, param_type, strategy)
        else:
            mutated_arg = arg

        # 假设 test_param_success 是一个存在的函数
        #success = test_param_success(mutated_arg)
        #mutation_history.add_record(param_type, arg, mutated_arg, success, strategy)
        #mutated_args.append(mutated_arg)

    return mutated_arg


def test_param_success(mutated_arg, func, mutation_history):
    try:
        result = func(*mutated_arg)

        if result is None:
            mutation_history.log_error(mutated_arg, func.__name__, "None Value")
            return False

        if isinstance(result, np.ndarray):
            if np.any(np.isnan(result)) or np.any(np.isinf(result)):
                error_type = "NaN Value" if np.any(np.isnan(result)) else "Inf Value"
                mutation_history.log_error(mutated_arg, func.__name__, error_type)
                return False

        elif isinstance(result, (float, int)):
            if np.isnan(result) or np.isinf(result):
                error_type = "NaN Value" if np.isnan(result) else "Inf Value"
                mutation_history.log_error(mutated_arg, func.__name__, error_type)
                return False

        return True
    except Exception as e:
        mutation_history.log_error(mutated_arg, func.__name__, str(e))
        return False



mutated_args = mutate_parameters(args, params_info, mutation_history)

for original_arg, mutated_arg in zip(args, mutated_args):
    success = test_param_success(mutated_arg, your_test_function, params_info)
    param_type = params_info[original_arg]['type']
    mutation_history.add_record(param_type, success, original_arg, mutated_arg, strategy)

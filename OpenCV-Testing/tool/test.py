import cv2
from API_info import *  # Assuming this contains necessary info or functions
import random
from tool.opencv_args_seed_generator import *
from tool.mutation import *
from tool.parser_from_str2funcandinfo import *
import logging
import time

# 配置日志记录
logging.basicConfig(filename='test_list.log', level=logging.INFO, filemode='w', format='%(asctime)s - %(levelname)s - %(message)s')
def log_bug_to_file(api_name, args):
    with open('C:/Users/uqbduan/Desktop/OpenCV-Testing/bug_case/bug_reports'+api_name+'.txt', 'a') as file:
        file.write(f"API: {api_name}, Args: {args}\n")
def test_one(string, iterations):
    print(string)
    api_name = string
    func, params_info = instantiate_api_and_get_params_info(api_name)
    args, outputs_num = instantiate_opencv_function_args_seed(params_info)
    #print(args)
    #args = mutate_parameters(args,params_info)
    #print(args)
    #funcs = func
    #outputs = funcs(*args)
    #if outputs_num == 1:
    #    output = outputs
    #else:
    #    output = outputs[:outputs_num]
    #print("Output:", outputs)
    # 实例化突变历史对象
    mutation_history = MutationHistory("log/error_records.xlsx")

    # 实例化突变历史对象
    #mutation_history = MutationHistory()
    for n in range(iterations):
        try:
            mutated_args = mutate_parameters(args, params_info, mutation_history)
            #print(mutated_args)
            print(n)
            mutation_history.log_case(func.__name__, mutated_args,n)
            success = test_param_success(mutated_args, func, mutation_history,n)
            #log_bug_to_file(api_name, mutated_args)
            if not success:
                # Log the failure case and continue with the mutation
                logging.error(f"API: {api_name}, Iteration: {n}, Args: {mutated_args} - Test failed.")
                log_bug_to_file(api_name, mutated_args)
                break
            for i, (param_name, param_info) in enumerate(params_info.items()):
                if param_name == 'output':
                    continue  # 跳过输出参数
                param_type = param_info['type']
                strategy = mutation_history.suggest_mutation_strategy(param_type)
                mutated_arg = mutated_args[i]  # 使用索引而不是名称访问
                # 接下来是记录突变结果的代码...
                mutation_history.add_record(api_name,param_type, success,mutated_arg, mutated_args, strategy)

        except Exception as e:
            # Handle exceptions, including crashes, and log them
            logging.exception(f"Exception for API: {api_name}, Iteration: {n}, Args: {args}. Error: {e}")
            log_bug_to_file(api_name, args)
            break

def test_list(data,iterations):
    start_time = time.time()
    for string in data:
        # 在这里处理每个字符串
        print(string)
        api_name = string
        #func, params_info = instantiate_api_and_get_params_info(api_name)
        #args, outputs_num = instantiate_opencv_function_args_seed(params_info)
        #funcs = func
        #outputs = funcs(*args)
        logging.info(f"当前处理的API名称：{api_name}")
        test_one(string, iterations)
        #if outputs_num == 1:
        #    output = outputs
        #else:
        #    output = outputs[:outputs_num]

        #print("Output:", output)
    end_time = time.time()
    runtime = end_time - start_time
    print(runtime)
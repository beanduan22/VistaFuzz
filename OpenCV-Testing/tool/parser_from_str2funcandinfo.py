import cv2
from API_info import *
def instantiate_api_and_get_params_info(api_name):
    # 分割字符串以获取模块名和函数名
    module_name, func_name = api_name.rsplit('.', 1)

    # 实例化API
    func = getattr(cv2, func_name)

    # 构建获取参数信息的函数名
    params_info_func_name = 'get_' + func_name + '_params_info'

    # 调用参数信息函数（假设它已定义）
    params_info_func = globals()[params_info_func_name]
    params_info = params_info_func()

    return func, params_info
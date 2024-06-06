import random
import numpy as np
import cv2
import string
pre_arg = ''

def apply_ndarray_mutation(arg, param_info, strategy, type, name):
    if name == 'M':
        n = random.randint(0,4)
        src = np.random.rand(4, 1, 2).astype(np.float32) * 100
        dst = np.random.rand(4, 1, 2).astype(np.float32) * 100
        arg = cv2.getPerspectiveTransform(src, dst)
        return arg

    if strategy == 'random':
        if type is None:
            available_types = ['uint8', 'float32', 'float64', 'int32', 'int16']
            chosen_type = None
            types_in_str = [dtype for dtype in available_types if dtype in param_info['type']]
            if len(types_in_str) == 1:
                chosen_type = types_in_str[0]
            elif len(types_in_str) > 1:
                chosen_type = random.choice(types_in_str)

            return (np.random.rand(*arg.shape) * 255).astype(np.dtype(chosen_type))
        elif 'uint8' in type:
            arg = (np.random.rand(*arg.shape) * 255).astype(np.uint8)
        elif 'float32' in type:
            arg = (np.random.rand(*arg.shape) * 255).astype(np.float32)
        elif 'float64' in type:
            arg = (np.random.rand(*arg.shape) * 255).astype(np.float64)
        else:
            return (np.random.rand(*arg.shape) * 255).astype(arg.dtype)
        return arg

    else:
        if type is None:
            available_types = ['uint8', 'float32', 'float64','int32', 'int16']
            chosen_type = None
            types_in_str = [dtype for dtype in available_types if dtype in param_info['type']]
            if len(types_in_str) == 1:
                chosen_type = types_in_str[0]
            elif len(types_in_str) > 1:
                chosen_type = random.choice(types_in_str)

            return  (np.random.rand(*arg.shape)* 255).astype(np.dtype(chosen_type))
        elif 'uint8' in type:
            arg = (np.random.rand(*arg.shape) * 255).astype(np.uint8)
        elif 'float32' in type:
            arg = (np.random.rand(*arg.shape) * 255).astype(np.float32)
        elif 'float64' in type:
            arg = (np.random.rand(*arg.shape) * 255).astype(np.float64)
        else:
            arg = (np.random.rand(*arg.shape) * 255).astype(arg.dtype)
        increment_value = random.randint(0,10)
        return arg + increment_value

def apply_tuple_mutation(arg, param_info, strategy,param_name, value):
    if 'odd' in param_info['description']:
        return tuple(random.randint(1, 10)*2+1 for _ in arg)
    elif 'arrow' in param_info['description']:
        return (random.randint(0,50), random.randint(0,50))
    elif param_name == 'ksize':
        n = random.randint(0, 4) * 2 + 1
        return (n, n)
    elif 'color' in param_info['description']:
        return (random.randint(0,255), random.randint(0,255), random.randint(0,255))
    elif 'imgRect' in param_name:
        x = random.randint(10, 20)
        y = random.randint(10, 20)
        w = random.randint(1, 10)
        h = random.randint(1, 10)
        return (x, y, h, w)
    elif 'rectList' in param_name:
        x = random.randint(50, 100)
        y = random.randint(50, 100)
        w = random.randint(1, 50)
        h = random.randint(1, 50)
        return [[x, x, w, h], [y, y, w, h], [x // 2, y // 2, w, h]]
    elif 'Window size' in param_info['description']:
        n = random.randint(0,20)
        return (n, n)
    elif 'seedPoint' in param_name:
        return (random.randint(0,value), random.randint(0,value))
    elif 'dsize' in param_name:
        return (random.randint(0,52767), random.randint(0,50000))
    elif (param_name == 'coeffs'):
        x = random.randint(2, value)
        y = random.randint(2, value)
        z = random.randint(1, value)
        return np.array([x, y, z], dtype=np.float32)
    elif 'criteria' in param_name:
        x = random.randint(1,20)
        x_ = random.uniform(-1.0, 1.0)
        n = random.choice([(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, x, x_), (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, x, x_)])
        return n
    elif 'rect' in param_name:
        center = (random.randint(100, 200), random.randint(100, 200))
        size = (random.randint(1, 99), random.randint(1, 99))
        angle = random.randint(-180, 180)
        return (center, size, angle)
    if arg is None:
        # 处理arg为None的情况
        return tuple()
    else:
        if strategy == 'random':
            # 随机改变tuple中的值
            return tuple(random.randint(3, 100) for _ in arg)
        elif strategy == 'bit_flip':
            # 对于整数进行位翻转操作，这里简化为对每个元素执行异或操作，以255为例
            return tuple(x ^ 255 for x in arg if isinstance(x, int))
        elif strategy == 'incremental':
            # 将每个整数元素增加1
            return tuple(x + 1 for x in arg if isinstance(x, int))
        # 添加其他策略的处理
        return arg

def apply_char_mutation(arg, param_info, strategy,param_name, value):
    characters = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S',
                  'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

    arg = random.choice(characters)
    return arg
# 类似地，为其他类型定义突变函数
def apply_int_type_mutation(arg, param_info, param_type, strategy,param_name):
    if param_name is 'distType':
        return random.choice([cv2.DIST_L1, cv2.DIST_L2])
    elif param_name is 'dx':
        return random.choice([0, 1])
    elif param_name is 'ksize':
        return random.choice([1, 3, 5, 7])
    elif (param_name is 'blockSize') or (param_name is 'blocksize') or (param_name is 'Blocksize'):
        return random.choice([3, 5, 7])
    elif ('FILTER_flags' == param_name) and ('default' in param_info) and ('cv2.' in param_info['default']):
        option = [cv2.RECURS_FILTER, cv2.NORMCONV_FILTER]
        return random.choice(option)
    elif ('CALIBflags' == param_name) :
        option = [cv2.CALIB_CB_NORMALIZE_IMAGE]
        return random.choice(option)
    elif param_name is 'cmpop':
        option = [cv2.CMP_EQ, cv2.CMP_GT, cv2.CMP_GE, cv2.CMP_LT, cv2.CMP_LE, cv2.CMP_NE]
        return random.choice(option)
    elif ('code' in param_name) and ('default' in param_info) and ('cv2.' in param_info['default']):
        option = [cv2.COLOR_YUV2BGR_NV12, cv2.COLOR_YUV2RGB_NV12, cv2.COLOR_YUV2BGRA_NV12, cv2.COLOR_YUV2RGBA_NV12,
                  cv2.COLOR_YUV2BGR_NV21, cv2.COLOR_YUV2RGB_NV21, cv2.COLOR_YUV2BGRA_NV21, cv2.COLOR_YUV2RGBA_NV21]
        return random.choice(option)
    elif ('borderType' in param_name) and ('default' in param_info) and ('cv2.' in param_info['default']):
        option = [cv2.BORDER_CONSTANT, cv2.BORDER_REPLICATE, cv2.BORDER_REFLECT]
        return random.choice(option)
    elif ('type_thresh' in param_name) and ('default' in param_info) and ('cv2.' in param_info['default']):
        option = [cv2.THRESH_BINARY, cv2.THRESH_BINARY_INV, cv2.THRESH_TRUNC, cv2.THRESH_TOZERO, cv2.THRESH_TOZERO_INV]
        return random.choice(option)
    elif ('fontFace' in param_name) and ('default' in param_info) and ('cv2.' in param_info['default']):
        option = [cv2.FONT_HERSHEY_SIMPLEX, cv2.FONT_HERSHEY_PLAIN]
        return random.choice(option)
    elif ('rotateCode' in param_name) and ('default' in param_info) and ('cv2.' in param_info['default']):
        option = [cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_180, cv2.ROTATE_90_COUNTERCLOCKWISE]
        return random.choice(option)
    elif ('HistCompMethods' in param_name) :
        option = [cv2.HISTCMP_CORREL, cv2.HISTCMP_CHISQR, cv2.HISTCMP_INTERSECT, cv2.HISTCMP_BHATTACHARYYA]
        return random.choice(option)
    elif ('Impaint_flags' in param_name) and ('default' in param_info) and ('cv2.' in param_info['default']):
        option = [cv2.INPAINT_TELEA, cv2.INPAINT_NS]
        return random.choice(option)
    elif ('Contoursmode' in param_name) and ('default' in param_info) and ('cv2.' in param_info['default']):
        option = [cv2.RETR_TREE, cv2.RETR_EXTERNAL, cv2.RETR_LIST, cv2.RETR_CCOMP]
        return random.choice(option)
    elif ('normType' in param_name):
        option = [cv2.NORM_L2, cv2.NORM_L1, cv2.NORM_L2SQR, cv2.NORM_INF]
        return random.choice(option)
    elif ('Contoursmethod' in param_name) and ('default' in param_info) and ('cv2.' in param_info['default']):
        option = [cv2.cv2.CHAIN_APPROX_SIMPLE, cv2.CHAIN_APPROX_NONE]
        return random.choice(option)
    elif ('motionType' in param_name) and ('default' in param_info) and ('cv2.' in param_info['default']):
        option = [cv2.MOTION_TRANSLATION, cv2.MOTION_EUCLIDEAN, cv2.MOTION_AFFINE, cv2.MOTION_HOMOGRAPHY]
        return random.choice(option)
    elif ('flags' in param_name) and ('default' in param_info) and ('cv2.' in param_info['default']):
        option = [cv2.INTER_LINEAR, cv2.INTER_NEAREST, cv2.WARP_FILL_OUTLIERS, cv2.WARP_INVERSE_MAP, cv2.INTER_CUBIC,
                  cv2.INTER_AREA, cv2.INTER_LANCZOS4]
        return random.choice(option)
    elif param_name is 'R':
        return 255
    elif param_name is 'colormap':
        return random.choice([cv2.COLORMAP_JET, cv2.COLORMAP_HOT])
    elif 'depth' in param_name:
        return -1
    elif (param_name is 'K') or (param_name is 'update'):
        return 0
    elif param_name == 'imageSize':
        return (random.randint(0, 20), random.randint(0, 20))
    if 'only' in param_info['description']:
        arg = arg
    else:
        if strategy == 'random':
            # 随机返回一个在-10到10之间的整数
            arg = random.randint(1, 10)
        elif strategy == 'bit_flip':
            #    # 随机选择一个位进行翻转。例如，翻转最低位:
            bit_to_flip = 1 << random.randint(1, 8)  # 对于32位整数
            arg = arg ^ bit_to_flip
        elif strategy == 'incremental':
            arg = arg + 1 if random.choice([True, False]) else arg - 1
    if 'distance' or 'threshold' or  'radius' in param_info['description']:
        arg = abs(arg)
    return arg

def apply_float_type_mutation(arg, param_info, param_type, strategy):
    if 'theta' in param_info['description']:
        return np.pi / random.randint(1, 180)
    elif 'alpha' in param_info['description']:
        return random.uniform(0, 1)
    elif '<1' in param_info['description']:
        return random.uniform(0, 1.0)
    elif 'maxValue' in param_info['description']:
        return random.uniform(10, 255)
    if strategy == 'random':
        # 随机返回一个在-10.0到10.0之间的浮点
        arg = random.uniform(-10.0, 10.0)
    elif 'angle' in param_info['description']:
        return random.uniform(0.0, 180.0)
    elif strategy == 'incremental':
        # 随机决定是增加还是减少，然后应用一个小的增量
        increment = random.uniform(-0.5, 0.5)  # 作为示例，增量范围为-0.5到0.5
        arg = arg + increment
    elif strategy == 'bit_flip':
        # 向数值添加一些随机噪声
        noise = random.uniform(-0.1, 0.1)  # 噪声范围较小，比如-0.1到0.1
        arg = arg + noise
    # 如果没有匹配的策略，返回原始值
    if 'distance' or 'threshold' or  'radius' in param_info['description']:
        arg = abs(arg)
    return arg

def apply_double_type_mutation(arg, param_type, strategy):
    if strategy == 'random':
        # 随机返回一个在-10.0到10.0之间的浮点数
        increment = random.uniform(0.0, 30.0)
        return arg + increment
    elif strategy == 'incremental':
        # 随机决定是增加还是减少，然后应用一个小的增量
        increment = random.uniform(0.0, 10.0)  # 作为示例，增量范围为-0.5到0.5
        return arg + increment
    elif strategy == 'bit_flip':
        # 向数值添加一些随机噪声
        noise = random.uniform(0.0, 0.1)  # 噪声范围较小，比如-0.1到0.1
        return arg + noise
    # 如果没有匹配的策略，返回原始值
    return arg

def apply_bool_type_mutation(arg, param_type, strategy):
    if strategy == 'random':
        # 随机改变布尔值
        return arg
    else:
        # 明确地切换布尔值
        return not arg

def apply_str_type_mutation(arg, param_type, strategy):
    if strategy == 'random':
        # 生成一个与原字符串长度相同的随机字符串
        return ''.join(random.choices(string.ascii_letters + string.digits, k=len(arg)))
    elif strategy == 'bit_flip':
        # 将字符串转换为字符列表，随机打乱，然后重新组合为字符串
        char_list = list(arg)
        random.shuffle(char_list)
        return ''.join(char_list)
    elif strategy == 'incremental':
        # 在字符串末尾添加一个随机字符
        random_char = random.choice(string.ascii_letters + string.digits)
        return arg + random_char
    # 如果没有匹配的策略，返回原始字符串
    return arg


def apply_KeyPoint_type_mutation(arg, param_type, strategy):
    if strategy == 'random':
        # 随机修改cv2.KeyPoint的属性
        return cv2.KeyPoint(x=arg.pt[0] + random.uniform(-5, 5),
                            y=arg.pt[1] + random.uniform(-5, 5),
                            size=arg.size + random.uniform(-1, 1))
    elif strategy == 'bit_flip':
        # 调整关键点的角度，角度范围为[0, 360)
        new_angle = (arg.angle + random.uniform(-180, 180)) % 360
        return cv2.KeyPoint(x=arg.pt[0], y=arg.pt[1], size=arg.size, angle=new_angle,
                            response=arg.response, octave=arg.octave, class_id=arg.class_id)
    elif strategy == 'incremental':
        # 修改关键点的响应度
        new_response = arg.response + random.uniform(-0.1, 0.1)
        return cv2.KeyPoint(x=arg.pt[0], y=arg.pt[1], size=arg.size, angle=arg.angle,
                            response=new_response, octave=arg.octave, class_id=arg.class_id)
        # 如果没有匹配的策略，返回原始关键点
    return arg


import json
import numpy as np
import cv2
from tool.API_info import *  # Assuming this contains necessary info or functions
import random
from tool.opencv_args_seed_generator import *
from tool.mutation import *
from tool.parser_from_str2funcandinfo import *
from tool.load_json_file import  *
from tool.test import  *
import venn
# 加载数据
mutation_iterations = 1
json_file_path = './API/OpenCV_API_filtered_subset.json'
data = load_json_file(json_file_path)
#test_one('cv2.findChessboardCornersSBWithMeta', mutation_iterations)
test_list(data,mutation_iterations )
#print(cv2.INPAINT_TELEA)
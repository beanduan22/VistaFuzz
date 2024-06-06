import json
import numpy as np
import cv2
from API_info import *  # Assuming this contains necessary info or functions
import random
from opencv_args_seed_generator import *
from mutation import *
from parser_from_str2funcandinfo import *
from load_json_file import  *
from test import  *
import venn
# 加载数据
mutation_iterations = 1000
json_file_path = 'OpenCV_API_filtered_subset.json'
data = load_json_file(json_file_path)
#test_one('cv2.HoughLinesP', mutation_iterations)
test_list(data,mutation_iterations )

#cv2.HoughLinesP, LUT, cv2.Laplacian

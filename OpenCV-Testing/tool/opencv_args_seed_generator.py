import numpy as np
import cv2
from API_info import *  # Assuming this contains necessary info or functions
import random
def instantiate_opencv_function_args_seed(params_info):
    # Initialize parameters based on their types and format
    initialized_params = {}
    min_values = {}
    for param, info in params_info.items():
        #check format = numpy.darray
        if 'default' in info and info['default'] is 'None':
            initialized_params[param] = None
        if 'size' in info and info['size'] == 'numpy.ndarray':
            # Handle initialization for different data types
            if 'type' in info and (('float' in info['type']) or ('int' in info['type']) ):
                if '1xN/Nx1 3-channel' in info['description']:
                    initialized_params[param] = (np.random.rand(100, 1, 3) * 256).astype(np.float32)
                elif '1xN/Nx1 2-channel' in info['description']:
                    initialized_params[param] = (np.random.rand(100, 1, 2) * 256).astype(np.float32)
                elif '1xNx2 array' in info['description']:
                    initialized_params[param] = (np.random.rand( 1, 10, 2) * 256).astype(np.float32)
                elif 'sequence of 8-bit 3-channel' in info['description']:
                    initialized_params[param] = (np.random.rand(3,100, 100, 3) * 256).astype(np.uint8)
                elif 'sequence of 8-bit 1-channel' in info['description']:
                    initialized_params[param] = (np.random.rand(3,100, 100, 1) * 256).astype(np.uint8)
                elif '8-bit 3-channel' in info['description']:
                    initialized_params[param] = (np.random.rand(100, 100, 3) * 256).astype(np.uint8)
                elif '8-bit single-channel' in info['description']:
                    initialized_params[param] = (np.random.rand(100, 100, 1) * 256).astype(np.uint8)
                elif ('points' in param) or ('2D point set' in  info['description']) or ('2D points' in  info['description']) or ('points (x, y)' in info['description']):  # Specific handling for points in HoughLinesPointSet
                    initialized_params[param] = (np.random.rand(50, 1, 2)* 256).astype(np.float32)
                elif (param == 'curve') or (param =='corners'):  # Specific handling for points in HoughLinesPointSet
                    initialized_params[param] = (np.random.rand(100, 1, 2)* 256).astype(np.float32)
                elif ('3x4 matrix' in info['description']) :
                    initialized_params[param] = (np.random.rand(4,3) * 256).astype(np.float32)
                elif ('4x3 matrix' in info['description']) :
                    initialized_params[param] = (np.random.rand(3,4) * 256).astype(np.float32)
                elif ('4x4 matrix' in info['description']) :
                    initialized_params[param] = (np.random.rand(4,4) * 256).astype(np.float32)
                elif ('distCoeffs' in param):  # Specific handling for points in HoughLinesPointSet
                    initialized_params[param] = np.zeros((5, 1), dtype=np.float32)
                elif ('histogram' in info['description']):
                    img = (np.random.rand(100, 100, 1)* 256).astype(np.float32)
                    initialized_params[param] = cv2.calcHist([img], [0], None, [256], [0, 256])
                elif param == 'probImage':  # Specific handling for image in cv2.Canny
                    initialized_params[param] = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
                elif ('Model' in param) and ('1x13' in info['description']):  # Specific handling for image in cv2.Canny
                    initialized_params[param] = np.zeros((1, 65), np.float64)
                elif ('2D point set' in  info['description']) or ('2xN/Nx2 1-channel' in info['description']) or ('1xN/Nx1 2-channel' in info['description']):
                    initialized_params[param] = (np.random.rand(100, 1, 2)* 256).astype(np.float32)
                elif param == 'moments':
                    image = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
                    _, thresh = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
                    initialized_params[param] = cv2.moments(thresh)
                elif ('identity matrix' in info['description']) or ('input map' in info['description']):
                    initialized_params[param] = (np.random.rand(10,10) * 256).astype(np.float32)
                elif ('3-element vector [x, y, 1]' in info['description']):
                    initialized_params[param] = (np.random.rand(10,10,1)).astype(np.float32)
                elif ('4-element vector' in info['description']) :
                    initialized_params[param] = (np.random.rand(4)).astype(np.float64)
                elif ('1D' in info['description']) or ('vector' in info['description']) or ('array' in info['description']):
                    initialized_params[param] = (np.random.rand(100)*256).astype(np.float32)
                elif ('2x3' in info['description']) :
                    initialized_params[param] = (np.random.rand(3,2) * 256).astype(np.float32)
                elif ('3x2' in info['description']) :
                    initialized_params[param] = (np.random.rand(2,3) * 256).astype(np.float32)
                elif ('3x3' in info['description']) or ('matrix' in info['description']):
                    initialized_params[param] = (np.random.rand(3,3) * 256).astype(np.float32)
                elif ('2x2' in info['description']) or ('matrix' in info['description']):
                    initialized_params[param] = (np.random.rand(2,2) * 256).astype(np.float32)
                elif ('warp_matrix 2x3' in info['description']):
                    initialized_params[param] = np.eye(3, 3, dtype=np.float32)
                elif ('2x4 matrix' in info['description']) or ('matrix' in info['description']):
                    initialized_params[param] = (np.random.rand(4,2) * 256).astype(np.float32)
                elif '2-channel' in info['description']:
                    initialized_params[param] = (np.random.rand(50, 50, 2) * 256).astype(np.uint8)
                elif 'Convolution kernel' in info['description']:
                    initialized_params[param] = (np.random.rand(3, 3)).astype(np.float32)
                elif ('1x3 or 3x1' in info['description']) or ('3x1' in info['description']):
                    initialized_params[param] = (np.random.rand(1, 3)).astype(np.float32)
                elif '2x3' in info['description']:
                    initialized_params[param] = (np.random.rand(3, 2)).astype(np.float32)
                elif '2x1 shape' in info['description']:
                    initialized_params[param] = np.full((1,2),100).astype(np.int32)
                elif 'Rotation 1x3' in info['description']:
                    initialized_params[param], _ = cv2.Rodrigues(np.array([0.1, 0.2, 0.3], dtype=np.float32))
                elif param == 'icovar':  # Specific handling for image in cv2.Canny
                    initialized_params[param] = np.random.randint(0, 256, (100, 100), dtype=np.float32)
                elif param == 'lut':  # Specific handling for image in cv2.Canny
                    initialized_params[param] = np.clip(np.arange(256, dtype=np.uint8) + 30, 0, 255)
                elif 'mask' == param:
                    if '2 pixels taller' in info['description']:
                        initialized_params[param] = np.zeros((102, 102)).astype(np.uint8)
                    else:
                        img = np.zeros((100, 100, 3)).astype(np.uint8)
                        initialized_params[param] = img[:-1:]
                elif param == 'M':
                    src = np.random.rand(10, 1, 2).astype(np.float32) * 100
                    dst = np.random.rand(10, 1, 2).astype(np.float32) * 100
                    initialized_params[param] = cv2.getPerspectiveTransform(src, dst)


                else:
                    initialized_params[param] = (np.random.rand(100, 100) * 256).astype(np.float32)

                if 'int32' in info['type']:
                    initialized_params[param] = np.int32(initialized_params[param])
                elif 'int16' in info['type']:
                    initialized_params[param] = np.int16(initialized_params[param])
                elif 'uint8' in info['type']:
                    initialized_params[param] = np.uint8(initialized_params[param])

            elif 'type' in info and 'uint8' in info['type']:
                if param == 'image' or param == 'img':  # Specific handling for image in cv2.Canny
                    initialized_params[param] = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
                    #if len(initialized_params[param].shape) > 2:
                    if 'description' in info and 'gray' in info['description']:
                        initialized_params[param] = cv2.cvtColor(initialized_params[param], cv2.COLOR_BGR2GRAY)
                    elif 'edge' or 'Canny' in info['description']:
                        initialized_params[param]  = cv2.Canny(initialized_params[param], random.randint(0, 100), random.randint(0, 100)+100, apertureSize=3)

                else:
                    initialized_params[param] = np.random.rand(100, 100).astype(np.uint8)
            else:
                initialized_params[param] = np.random.rand(100, 100).astype(np.float32)

        elif 'size' in info and info['size'] == 'tuple':
            # Initialize a dummy tuple
            if param == 'window':
                initialized_params[param] = (0, 0, 100, 100)
            elif param == 'criteria':
                initialized_params[param] = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.01)
            elif (param == 'coeffs'):
                initialized_params[param] = np.array([3, 2, 1], dtype=np.float32)
            elif param == 'ksize':
                n = random.randint(0, 4) * 2 + 1
                if 'odd' in info['description']:

                    initialized_params[param] = (n,n)
                else:
                    initialized_params[param] = (n,n)
            elif 'pt' in param:
                initialized_params[param] = (random.randint(0,50), random.randint(0,50))
            elif param == 'color':
                initialized_params[param] = (random.randint(0,255), random.randint(0,255), random.randint(0,255))
            elif param == 'winSize':
                initialized_params[param] = (random.randint(3,20), random.randint(3,20), random.randint(0,255))
            elif param == 'range':
                initialized_params[param] = (0, 180)
            elif param == 'channels':
                initialized_params[param] = [1]
            elif 'rect' in param:
                center = (random.randint(100,200), random.randint(100,200))
                size = (random.randint(1, 99), random.randint(1, 99))
                angle = random.randint(-180, 180)
                initialized_params[param] = (center, size, angle)
            elif 'imgRect' in param:
                x = random.randint(25, 50)
                y = random.randint(25, 50)
                w = random.randint(1, 10)
                h = random.randint(1, 10)
                initialized_params[param] = (x, y, h, w)
            elif 'rectList' in param:
                x = random.randint(50, 100)
                y = random.randint(50, 100)
                w = random.randint(1, 50)
                h = random.randint(1, 50)
                initialized_params[param] = [[x, x, w, h],[y, y, w, h],[x//2, y//2, w, h]]
            elif param == 'imageSize':
                initialized_params[param] = (random.randint(0,20), random.randint(0,20))
            elif param == 'ranges':
                initialized_params[param] = (0, 256)
            elif param == 'histSize':
                initialized_params[param] = [1]
            elif (param == 'zeroZone') or (param == 'anchor'):
                initialized_params[param] = (-1, -1)
            elif 'Vertex' in info['description']:
                n = random.randint(0,100)
                initialized_params[param] = (n, n)
            elif 'defualt' in info:
                initialized_params[param] = eval(info.get('default', 1))
            else:
                initialized_params[param] = (random.randint(3,50), random.randint(3,50))

            if info['type'] == 'int32':
                initialized_params[param] = np.int32(initialized_params[param])


        elif ('type' in info and info['type'] == 'char'):
            characters = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S',
                          'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

            initialized_params[param] = random.choice(characters)


        elif ('type' in info and info['type'] == 'int') or ('type' in info and info['type'] == 'uint8'):
            if param == 'lines_max':
                initialized_params[param] = max(1, int(info.get('default', 1)))
            elif param == 'distType':
                option = [cv2.DIST_L2, cv2.DIST_L1]
                initialized_params[param] = random.choice(option)
            elif 'min' in param:
                initialized_params[param] = max(10, int(info.get('default', 1)))
            elif 'max' in param:
                initialized_params[param] = max(10, int(info.get('default', 1)))
            elif ('ksize' in param) or ('blockSize' in param):
                kernel = [3 , 5,  7]
                initialized_params[param] = random.choice(kernel)
            elif ('cmpop' in param) :
                option = [cv2.CMP_EQ, cv2.CMP_GT, cv2.CMP_GE, cv2.CMP_LT, cv2.CMP_LE, cv2.CMP_NE]
                initialized_params[param] = random.choice(option)
            elif ('rotateCode' in param) and ('default' in info) and ('cv2.' in info['default']):
                option = [cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_180, cv2.ROTATE_90_COUNTERCLOCKWISE]
                initialized_params[param] = random.choice(option)
            elif ('fontFace' in param) and ('default' in info) and ('cv2.' in info['default']):
                option = [cv2.FONT_HERSHEY_SIMPLEX, cv2.FONT_HERSHEY_PLAIN]
                initialized_params[param] = random.choice(option)
            elif ('Impaint_flags' in param) and ('default' in info) and ('cv2.' in info['default']):
                option = [cv2.INPAINT_TELEA, cv2.INPAINT_NS]
                initialized_params[param] = random.choice(option)
            elif ('normType' in param) and ('default' in info) and ('cv2.' in info['default']):
                option = [cv2.NORM_L2, cv2.NORM_L1, cv2.NORM_L2SQR, cv2.NORM_INF]
                initialized_params[param] = random.choice(option)
            elif ('type_thresh' in param) and ('default' in info) and ('cv2.' in info['default']):
                option = [cv2.THRESH_BINARY, cv2.THRESH_BINARY_INV, cv2.THRESH_TRUNC, cv2.THRESH_TOZERO, cv2.THRESH_TOZERO_INV]
                initialized_params[param] = random.choice(option)
            elif ('flags' in param) and ('default' in info) and ('cv2.' in info['default']):
                option = [cv2.INTER_LINEAR, cv2.INTER_NEAREST, cv2.WARP_FILL_OUTLIERS, cv2.WARP_INVERSE_MAP, cv2.INTER_CUBIC, cv2.INTER_AREA, cv2.INTER_LANCZOS4]
                initialized_params[param] = random.choice(option)
            elif ('code' in param) and ('default' in info) and ('cv2.' in info['default']):
                option = [cv2.COLOR_YUV2BGR_NV12, cv2.COLOR_YUV2RGB_NV12, cv2.COLOR_YUV2BGRA_NV12, cv2.COLOR_YUV2RGBA_NV12, cv2.COLOR_YUV2BGR_NV21, cv2.COLOR_YUV2RGB_NV21, cv2.COLOR_YUV2BGRA_NV21,  cv2.COLOR_YUV2RGBA_NV21]
                initialized_params[param] = random.choice(option)
            elif ('borderType' in param) and ('default' in info) and ('cv2.' in info['default']):
                option = [cv2.BORDER_CONSTANT, cv2.BORDER_REPLICATE, cv2.BORDER_REFLECT]
                initialized_params[param] = random.choice(option)
            elif ('HistCompMethods' in param) and ('default' in info) and ('cv2.' in info['default']):
                option = [cv2.HISTCMP_CORREL, cv2.HISTCMP_CHISQR, cv2.HISTCMP_INTERSECT, cv2.HISTCMP_BHATTACHARYYA]
                initialized_params[param] = random.choice(option)
            elif ('CALIBflags' in param) :
                option = [cv2.CALIB_CB_NORMALIZE_IMAGE]
                initialized_params[param] = random.choice(option)
            elif ('Contoursmethod' in param) and ('default' in info) and ('cv2.' in info['default']):
                option = [cv2.CHAIN_APPROX_SIMPLE, cv2.CHAIN_APPROX_NONE]
                initialized_params[param] = random.choice(option)
            elif ('motionType' in param) and ('default' in info) and ('cv2.' in info['default']):
                option = [cv2.MOTION_TRANSLATION, cv2.MOTION_EUCLIDEAN, cv2.MOTION_AFFINE,  cv2.MOTION_HOMOGRAPHY]
                initialized_params[param] = random.choice(option)
            elif ('FILTER_flags' in param) and ('default' in info) and ('cv2.' in info['default']):
                option = [cv2.RECURS_FILTER , cv2.NORMCONV_FILTER]
                initialized_params[param] = random.choice(option)
            else:
                default_value = info.get('default', '0')
                if '|' in  default_value:
                    initialized_params[param] = eval(default_value)
                elif default_value.startswith('cv2.'):
                    # Use the OpenCV constant
                    initialized_params[param] = getattr(cv2, default_value.split('.')[1])
                else:
                    # Use the provided default value
                    initialized_params[param] = eval(default_value)
        elif 'type' in info and info['type'] == 'bool':
            initialized_params[param] = info.get('default', False) in [True, 'True', 'true', 1, '1']
        elif 'type' in info and info['type'] == 'float':
            if 'min' in param:
                if 'default' in info:
                    initialized_params[param] = eval(info.get('default',1))
                else:
                    min_value = 1.0  # Example value for parameters with 'min'
                    initialized_params[param] = min_value
                    min_values[param] = min_value  # Store the minimum value
            elif 'max' in param:
                if 'default' in info:
                    initialized_params[param] = eval(info.get('default',1))
                else:
                    min_param = param.replace('max', 'min')
                    min_value = min_values.get(min_param, 0.0)
                    initialized_params[param] = min_value + 100.0  # Ensure max is greater than min
            elif 'theta' in param:
                if 'default' in info:
                    initialized_params[param] = eval(info.get('default',1))
                else:
                    initialized_params[param] = np.pi/random.uniform(1,180)
            elif 'psi' in param:
                initialized_params[param] = np.pi/2
            elif 'pyr_scale' in param:
                initialized_params[param] = random.uniform(0.0, 1.0)
            elif 'angle' in param:
                initialized_params[param] = random.uniform(0.0, 180.0)
            else:
                if 'default' in info:
                    initialized_params[param] = eval(info.get('default',1))
                else:
                    initialized_params[param] = random.randint(0.0,100.0)
        elif 'type' in info and info['type'] == 'double':
            if 'description' in info and  'First threshold' in info['description']:
                initialized_params[param] = float(info.get('default', 20.0))
            elif 'description' in info and  'Second threshold' in info['description']:
                initialized_params[param] = float(info.get('default', 50.0))
            else:
                initialized_params[param] = float(info.get('default', 1.0))
        elif 'type' in info and info['type'] == 'str':
            if 'path' and 'XML' in info['description']:
                initialized_params[param] = 'example/xml/test.xml'  # Placeholder value
            else:
                initialized_params[param] = 'dummy_string_value'  # Placeholder value

        elif 'type' in info and 'KeyPoint' in info['type']:
            if 'ORB' in info['description']:
                image = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
                orb = cv2.ORB_create()
                initialized_params[param] = orb.detect(image, None)
            else :
                initialized_params[param] = cv2.KeyPoint(x = random.randint(0, 100), y = random.randint(0, 100), size=random.randint(0, 100))

    args = [initialized_params.get(param) for param in params_info if 'output' not in param]
    num_outputs = int(params_info.get('output', {}).get('numbers', 1))
    return args, num_outputs

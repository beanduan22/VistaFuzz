def get_HoughLinesPointSet_params_info():
    info = {
        'points': {
            'format': 'numpy.ndarray',
            'type': 'float32',
            'description': '2D points 2-channel array of points'
        },
        'lines_max': {
            'type': 'int',
            'default': '2',
            'flag': 'None',
            'description': 'Maximum number of output lines'
        },
        'threshold': {
            'type': 'int',
            'description': 'Accumulator threshold parameter'
        },
        'min_rho': {
            'type': 'int',
            'default':'0',
            'flag': 'None',
            'description': 'Minimum distance to the origin (rho)'
        },
        'max_rho': {
            'type': 'int',
            'default': '100',
            'flag': 'None',
            'description': 'Maximum distance to the origin (rho)'
        },
        'rho_step': {
            'type': 'int',
            'default': '1',
            'flag': 'None',
            'description': ' distance to the origin (rho)'
        },
        'min_theta': {
            'type': 'float',
            'default':'0',
            'flag':'None',
            'description': 'Minimum angle in radians'
        },
        'max_theta': {
            'type': 'float',
            'default': 'np.pi',
            'flag': 'None',
            'description': 'Maximum angle in radians'
        },
        'theta_step': {
            'type': 'int',
            'default': '1',
            'flag': 'None',
            'description': ' distance to the origin (rho)'
        },
        'output': {
            'numbers': '1'
        }
    }
    return info

def get_integral_params_info():
    info = {
        'src': {
            'format': 'numpy.ndarray',
            'type': 'uint8 or float32',
            'description': 'Input image as a single-channel 8-bit or floating-point.'
        },
        'sdepth': {
            'type': 'int',
            'default': '-1',
            'flag': 'None',
            'description': 'Depth of the integral and the tilted integral images, by default set to CV_32S.'
        },
        'output': {
            'numbers': '1'
        }
    }
    return info

def get_integral2_params_info():
    info = {
        'src': {
            'format': 'numpy.ndarray',
            'type': 'uint8 or float32 or float64',
            'description': 'Input image that is a single-channel 8-bit or floating-point.'
        },
        'sum': {
            'format': 'numpy.ndarray',
            'type': 'uint8 or float32 or float64 same type as src',
            'description': 'Integral image as (W+1)x(H+1), 32-bit integer or floating-point (32f or 64f).'
        },
        #'sqsum': {
        #    'format': 'numpy.ndarray',
        #    'type': 'CV_64F',
        #    'description': 'Integral image of squared pixel values; double-precision floating-point (64f) array.'
        #},
        'sdepth': {
            'type': 'int',
            'default': '-1',
            'description': 'Desired depth of the integral image, CV_32S, CV_32F, or CV_64F.'
        },

        'output': {
            'numbers': '2'
        }
    }
    return info

def get_integral3_params_info():
    info = {
        'src': {
            'format': 'numpy.ndarray',
            'type': 'uint8 or float32 or float64',
            'description': 'Input image as W x H, 8-bit or floating-point (32f or 64f).'
        },
        'sum': {
            'format': 'numpy.ndarray',
            'type': '(W+1) x (H+1), int32 or float32 or float64',
            'description': 'Integral image as (W+1) x (H+1), 32-bit integer or floating-point (32f or 64f).'
        },
        'sqsum': {
            'format': 'numpy.ndarray',
            'type': '(W+1) x (H+1), float64',
            'default': 'None',
            'description': 'Integral image for squared pixel values; double-precision floating-point (64f) array.'
        },
        'tilted': {
            'format': 'numpy.ndarray',
            'type': 'same as sum',
            'description': 'Integral for the image rotated by 45 degrees; same data type as sum.'
        },
        #'sdepth': {
        #    'type': 'int',
        #    'values': 'CV_32S, CV_32F, or CV_64F',
        #    'description': 'Desired depth of the integral and the tilted integral images.'
        #},
        ###'sqdepth': {
        #    'type': 'int',
        #    'values': 'CV_32F or CV_64F',
        #    'description': 'Desired depth of the integral image of squared pixel values.'
        #},
        'output': {
            'numbers': '3'
        }
    }
    return info

def get_Canny_params_info():
    info = {
        'image': {
            'format': 'numpy.ndarray',
            'type': 'uint8',
            'description': 'Input image, grayscale and 8-bit.'
        },
        'threshold1': {
            'type': 'double',
            'description': 'First threshold for the hysteresis procedure.'
        },
        'threshold2': {
            'type': 'double',
            'description': 'Second threshold for the hysteresis procedure.'
        },
        'output': {
            'numbers': '1'
        }
    }
    return info

def get_CamShift_params_info():
    info = {
        'probImage': {
            'format': 'numpy.ndarray',
            'type': 'uint8 or float32',
            'description': 'Back projection of the object histogram. An image used to guide the search for the location of the object.'
        },
        'window': {
            'format': 'tuple',
            'type': 'int',
            'description': 'Initial search window. For example: (x, y, width, height).'
        },
        'criteria': {
            'format': 'tuple',
            'type': 'int',
            'description': 'Criteria for the algorithm termination. The tuple should be in the form: (type, max_iter, epsilon).'
        },
        'output': {
            'numbers': '2'
        }
    }
    return info

def get_intersectConvexConvex_params_info():
    info = {
        '_p1': {
            'format': 'numpy.ndarray',
            'type': 'float32',
            'description': 'First input convex polygon. It is passed as a 2D point set.'
        },
        '_p2': {
            'format': 'numpy.ndarray',
            'type': 'float32',
            'description': 'Second input convex polygon. Like the first one, it is passed as a 2D point set.'
        },
        'handleNested': {
            'type': 'bool',
            'default': 'False',
            'description': 'Optional flag to handle nested polygons. If it is True, the function will correctly handle non-convex polygons.'
        },
        'output': {
            'numbers': '2',
            'description': 'Returns the intersection area and a numpy.ndarray containing the points of the intersection polygon.'
        }
    }
    return info

def get_CV_MAKETYPE_params_info():
    info = {
        'depth': {
            'type': 'int',
            'description': 'The desired depth of the matrix. It can be one of the following: CV_8U, CV_8S, CV_16U, CV_16S, CV_32S, CV_32F, CV_64F, etc.'
        },
        'cn': {
            'type': 'int',
            'description': 'The number of channels. cn can range from 1 to 512.'
        },
        'output': {
            'numbers': '1',
            'description': 'Returns an integer that represents the specific matrix type.'
        }
    }
    return info

def get_CascadeClassifier_convert_params_info():#bug
    info = {
        'oldcascade': {
            'type': 'str',
            'description': 'The path to the old cascade in XML format. This is the cascade to be converted.'
        },
        'newcascade': {
            'type': 'str',
            'description': 'The path where the new cascade will be saved. This is the destination file for the converted cascade.'
        },
        'output': {
            'numbers': '1',
            'description': 'Returns a boolean value. True if the conversion was successful, False otherwise.'
        }
    }
    return info

def get_EMD_params_info():
    info = {
        'signature1': {
            'format': 'numpy.ndarray',
            'type': 'float32',
            'description': '2x3 First signature, a numpy of type float32.'
        },
        'signature2': {
            'format': 'numpy.ndarray',
            'type': 'float32',
            'description': '2x3 Second signature, similar to signature1, where each row is a feature with weight and dimensions.'
        },
        'distType': {
            'type': 'int',
            'default': 'cv2.DIST_L2',
            'flag': 'None',
            'description': 'Distance type used for EMD calculation. Common values are cv2.DIST_L1, cv2.DIST_L2, etc.'
        },
        'output': {
            'numbers': '3',
            'description': 'Returns the calculated Earth Mover\'s Distance (float).'
        }
    }
    return info

def get_GaussianBlur_params_info():
    info = {
        'src': {
            'format': 'numpy.ndarray',
            'type': 'uint8 or float32',
            'description': 'Input image on which the Gaussian blur will be applied.'
        },
        'ksize': {
            'format': 'tuple',
            'type': 'int',
            'description': 'Gaussian kernel size. ksize.width and ksize.height can differ but they both must be positive and odd.'
        },
        'sigmaX': {
            'type': 'float',
            'description': 'Gaussian kernel standard deviation in X direction.'
        },
        'sigmaY': {
            'type': 'float',
            'description': 'Gaussian kernel standard deviation in Y direction. If sigmaY is zero, it is set to be equal to sigmaX. If both are zeros, they are computed from ksize.'
        },
        'borderType': {
            'type': 'int',
            'default': 'cv2.BORDER_DEFAULT',
            'flag': 'None',
            'description': 'Pixel extrapolation method. See borderInterpolate for details.'
        },
        'output': {
            'numbers': '1',
            'description': 'Blurred image returned as a numpy.ndarray.'
        }
    }
    return info

def get_HOGDescriptor_getDaimlerPeopleDetector_params_info():
    info = {
        # This function does not require any input parameters.
        'output': {
            'numbers': '1',
            'description': 'Returns a numpy.ndarray containing the coefficients for the Daimler people detector. These coefficients can be used to set the HOGDescriptor for people detection.'
        }
    }
    return info

def get_HOGDescriptor_getDefaultPeopleDetector_params_info():
    info = {
        # This function does not require any input parameters.
        'output': {
            'numbers': '1',
            'description': 'Returns a numpy.ndarray containing the coefficients for the Daimler people detector. These coefficients can be used to set the HOGDescriptor for people detection.'
        }
    }
    return info

def get_HoughCircles_params_info():
    info = {
        'image': {
            'format': 'numpy.ndarray',
            'type': 'uint8',
            'description': '8-bit, single-channel grayscale input image.'
        },
        'method': {
            'type': 'int',
            'default': 'cv2.HOUGH_GRADIENT',
            'flag':'None',
            'description': 'Detection method to use. Currently, the only implemented method is cv2.HOUGH_GRADIENT.'
        },
        'dp': {
            'type': 'float',
            'description': 'Inverse ratio of the accumulator resolution to the image resolution. For example, if dp=1, the accumulator has the same resolution as the input image.'
        },
        'minDist': {
            'type': 'float',
            'description': 'Minimum distance between the centers of the detected circles.'
        },
        'param1': {
            'type': 'float',
            'description': 'First method-specific parameter. In case of cv2.HOUGH_GRADIENT, it is the higher threshold of the two passed to the Canny edge detector (the lower one is twice smaller).'
        },
        'param2': {
            'type': 'float',
            'description': 'Second method-specific parameter. In case of cv2.HOUGH_GRADIENT, it is the accumulator threshold for the circle centers at the detection stage. The smaller it is, the more false circles may be detected.'
        },
        'minRadius': {
            'type': 'int',
            'default': '0',
            'description': 'Minimum circle radius.'
        },
        'maxRadius': {
            'type': 'int',
            'default': '0',
            'description': 'Maximum circle radius.'
        },
        'output': {
            'numbers': '1',
            'description': 'A numpy.ndarray of shape (N, 3) where N is the number of detected circles. Each circle is represented by 3 values: (x_center, y_center, radius).'
        }
    }
    return info

def get_HoughLines_params_info():
    info = {
        'image': {
            'format': 'numpy.ndarray',
            'type': 'uint8',
            'description': 'Input image (grayscale) on which to perform line detection. The image should be edge detected using, for example, the Canny edge detector.'
        },
        'rho': {
            'type': 'float',
            'description': 'Distance resolution of the accumulator in pixels.'
        },
        'theta': {
            'type': 'float',
            'description': 'Angle resolution of the accumulator in radians.'
        },
        'threshold': {
            'type': 'int',
            'description': 'Accumulator threshold parameter. Only those lines are returned that get enough votes (> threshold).'
        },
        'output': {
            'numbers': '1',
            'description': 'A numpy.ndarray of shape (N, 1, 2) where N is the number of detected lines. Each line is represented by the two values (rho, theta).'
        }
    }
    return info

def get_HoughLinesP_params_info():
    info = {
        'image': {
            'format': 'numpy.ndarray',
            'type': 'uint8',
            'description': 'Input edge image (grayscale) on which to perform line detection. Typically, this is an edge-detected image using the Canny edge detector.'
        },
        'rho': {
            'type': 'float',
            'description': 'Distance resolution of the accumulator in pixels.'
        },
        'theta': {
            'type': 'float',
            'default': 'np.pi',
            'description': 'Angle resolution of the accumulator in radians.'
        },
        'threshold': {
            'type': 'int',
            'description': 'Accumulator threshold parameter. It specifies the minimum number of votes needed to detect a line.'
        },
        'minLineLength': {
            'type': 'float',
            'description': 'Minimum length of a line. Line segments shorter than this are rejected.'
        },
        'maxLineGap': {
            'type': 'float',
            'description': 'Maximum allowed gap between points on the same line to link them.'
        },
        'output': {
            'numbers': '1',
            'description': 'A numpy.ndarray of shape (N, 1, 4) where N is the number of detected lines. Each line is represented by the four values (x1, y1, x2, y2), representing the two end points of the line.'
        }
    }
    return info

def get_HoughLinesWithAccumulator_params_info():
    info = {
        'image': {
            'format': 'numpy.ndarray',
            'type': 'uint8',
            'description': 'Input image (grayscale) for line detection, typically edge-detected.'
        },
        'rho': {
            'type': 'float',
            'description': 'Distance resolution of the accumulator in pixels.'
        },
        'theta': {
            'type': 'float',
            'description': 'Angle resolution of the accumulator in radians.'
        },
        'threshold': {
            'type': 'int',
            'description': 'Accumulator threshold parameter for detecting lines.'
        },
        'output': {
            'numbers': '1',
            'description': 'Output format, likely similar to cv2.HoughLines.'
        }
    }
    return info

def get_HuMoments_params_info():
    info = {
        'moments': {
            'format': 'numpy.ndarray',
            'type': 'float32 or float64',
            'description': 'Array of moments, which can be computed using cv2.moments(). It should be a 1D array of 7 elements.'
        },
        'output': {
            'numbers': '1',
            'description': 'Returns an array of seven Hu moment invariants. These are shape descriptors and are invariant to the object’s translation, scale, and rotation.'
        }
    }
    return info

def get_KeyPoint_convert_params_info():
    info = {
        'keypoints': {
            'format': 'list of cv2.KeyPoint',
            'type': 'cv2.KeyPoint',
            'description': 'A list of cv2.KeyPoint objects. These are usually obtained from feature detection algorithms like SIFT, ORB, etc.'
        },
        'keypointIndexes': {
            'format': 'list of int',
            'type': 'int',
            'default': 'None',
            'description': 'Optional list of indices to be converted. If None, all keypoints are converted.'
        },
        'output': {
            'numbers': '1',
            'description': 'Returns a numpy.ndarray of shape (N, 2) where N is the number of keypoints. Each keypoint is represented by its x and y coordinates.'
        }
    }
    return info

def get_KeyPoint_overlap_params_info():
    info = {
        'kp1': {
            'type': 'cv2.KeyPoint',
            'description': 'First keypoint. It is an instance of the cv2.KeyPoint class.'
        },
        'kp2': {
            'type': 'cv2.KeyPoint',
            'description': 'Second keypoint. It is another instance of the cv2.KeyPoint class.'
        },
        'output': {
            'numbers': '1',
            'description': 'Returns a float value representing the overlap between the two keypoints. The value is between 0.0 and 1.0, where 1.0 means complete overlap.'
        }
    }
    return info

def get_LUT_params_info():
    info = {
        'src': {
            'format': 'numpy.ndarray',
            'type': 'uint8',
            'description': 'Input array of 8-bit elements. This is the source image or array for which the LUT will be applied. There is no default value for this parameter; it must be provided by the user.'
        },
        'lut': {
            'format': 'numpy.ndarray',
            'type': 'uint8',
            'flag':'None',
            'description': 'Look-up table of 256 elements. There is no default value for this parameter; it must be provided by the user.'
        },

        'output': {
            'numbers': '1',
            'description': 'The function returns an array (dst) of the same size and number of channels as src, with each element transformed according to the lut array. The depth of the output array is the same as that of lut.'
        }
    }
    return info

def get_Laplacian_params_info():
    info = {
        'src': {
            'format': 'numpy.ndarray',
            'type': 'unit8 or float32 or float64',
            'description': 'Source image. The image from which to calculate the Laplacian.'
        },
        'ddepth': {
            'type': 'int',
            'default': '-1',
            'flag':'None',
            'description': 'Desired depth of the destination image. The depth of the output image. Must be one of the OpenCV data types. See combinations for more details.'
        },
        'output': {
            'numbers': '1',
            'description': 'The output destination image with the same size and number of channels as src, containing the Laplacian of the source image.'
        }
    }
    return info

def get_Mahalanobis_params_info():
    info = {
        'v1': {
            'format': 'numpy.ndarray',
            'type': 'float32',
            'description': '3x1'
        },
        'v2': {
            'format': 'numpy.ndarray',
            'type': 'float32',
            'description': '3x1'
        },
        'icovar': {
            'format': 'numpy.ndarray',
            'type': 'float32',
            'description': 'Inverse covariance 3x3 matrix. This matrix is used to weight the distance calculation'
        },
        'output': {
            'numbers': '1',
            'description': 'The calculated Mahalanobis distance between the two input vectors. This is a single floating-point value representing the distance.'
        }
    }
    return info

def get_PCACompute_params_info():
    info = {
        'data': {
            'format': 'numpy.ndarray',
            'type': 'float32',
            'description': 'The projected data that will be backprojected to the original space. Each row represents a data point in the reduced space.'
        },
        'mean': {
            'type': 'float32',
            'default': 'None',
            'flag': 'None',
            'description': 'The mean vector of the original data. It is subtracted from the initial data before projecting and added back during backprojection.'
        },
        'maxComponents': {
            'format': 'numpy.ndarray',
            'type': 'float32',
            'default': '3',
            'flag': 'None',
            'description': 'The matrix of eigenvectors of the PCA, where each row contains one eigenvector.'
        },
        'output': {
            'numbers': '2',
            'description': 'The result of the backprojection, returning the data to its original space. This will be a new matrix if the result parameter is not provided.'
        }
    }
    return info

def get_PSNR_params_info():
    info = {
        'src1': {
            'format': 'numpy.ndarray',
            'type': 'uint8 or float32 or float64',
            'description': 'First input array. The original image against which quality measurement is to be done.'
        },
        'src2': {
            'format': 'numpy.ndarray',
            'type': 'uint8 or float32 or float64 as same as src1',
            'description': 'Second input array of the same size and type as src1. Typically, this is the reconstructed image after compression or any other processing.'
        },
        'R': {
            'type': 'int8',
            'default': '255',
            'flag': 'None',
            'description': 'The maximum pixel value of the image data type. For 8-bit images, it is usually 255. This is used in the PSNR calculation formula.'
        },
        'output': {
            'numbers': '1',
            'description': 'The Peak Signal-to-Noise Ratio between the two arrays in decibels (dB). Higher PSNR indicates a better quality of the reconstructed image.'
        }
    }
    return info

def get_RQDecomp3x3_params_info():
    info = {
        'src': {
            'format': 'numpy.ndarray',
            'type': 'float32 or float64',
            'description': 'Input 3x3 matrix that is to be decomposed.'
        },
        'output': {
            'numbers': '6',
            'description': 'The function returns a tuple containing the three Euler angles in degrees that correspond to the rotation matrices Qx, Qy, and Qz.'
        }
    }
    return info

def get_Rodrigues_params_info():
    info = {
        'src': {
            'format': 'numpy.ndarray',
            'type': 'float32 or float64',
            'description': 'Input rotation (3x1 or 1x3) or rotation matrix (3x3).'
        },
        'output': {
            'numbers': '2',
            'description': 'The function returns a tuple where the first element is the destination array, which is the converted rotation matrix or rotation vector, and the second element is the Jacobian matrix if it was requested.'
        }
    }
    return info

def get_SVBackSubst_params_info():
    info = {
        'w': {
            'format': 'numpy.ndarray',
            'type': 'float32',
            'description': 'The singular values of the input matrix A, represented as a 1D array.'
        },
        'u': {
            'format': 'numpy.ndarray',
            'type': 'float32',
            'description': 'The matrix of left singular vectors U of the input matrix A, represented as a 2D array.'
        },
        'vt': {
            'format': 'numpy.ndarray',
            'type': 'float32',
            'description': 'The transpose of the matrix of right singular vectors V of the input matrix A, represented as a 2D array.'
        },
        'rhs': {
            'format': 'numpy.ndarray',
            'type': 'float32',
            'description': 'The right-hand side matrix B, for which the linear system is solved.'
        },
        'output': {
            'numbers': '1',
            'description': 'The result of the back substitution, which is the solution X of the system AX = B.'
        }
    }
    return info

def get_SVDecomp_params_info():
    info = {
        'src': {
            'format': 'numpy.ndarray',
            'type': 'float32 or float64',
            'description': 'Input matrix to decompose. It can be of any size and type CV_32F or CV_64F.'
        },
        'flags': {
            'type': 'int',
            'default': '0',
            'description': 'Operation flags. Pass cv2.SVD_FULL_UV to compute full-size U and V matrices, or cv2.SVD_MODIFY_A for an optimized computation if A is not needed after the operation. This is an optional parameter.'
        },
        'output': {
            'numbers': '3',
            'description': 'The function returns the singular values in `w` (as a 1D array), the left singular vectors in `u` (as a 2D array), and the right singular vectors transposed in `vt` (as a 2D array).'
        }
    }
    return info

def get_Scharr_params_info():
    info = {
        'src': {
            'format': 'numpy.ndarray',
            'type': 'uint8 or float32 or float64',
            'description': 'Input image.'
        },
        'ddepth': {
            'type': 'int',
            'default': '-1',
            'flag': 'None',
            'description': 'Output image depth. See combinations for more details. -1 to use the same depth as the source.'
        },
        'dx': {
            'type': 'int',
            'default': '0',
            'flag': 'None',
            'description': 'Order of the derivative x. With Scharr, this must be 0 or 1.'
        },
        'dy': {
            'type': 'int',
            'default': '1',
            'flag': 'None',
            'description': 'Order of the derivative y. With Scharr, this must be 0 or 1.'
        },
        'output': {
            'numbers': '1',
            'description': 'The output image with the computed Scharr derivative. The same size and the same number of channels as src.'
        }
    }
    return info

def get_Sobel_params_info():
    info = {
        'src': {
            'format': 'numpy.ndarray',
            'type': 'any',
            'description': 'Input image.'
        },
        'ddepth': {
            'type': 'int',
            'default':'-1',
            'flag': 'None',
            'description': 'Desired depth of the destination image. The depth must be one of the OpenCV data types. Using -1 will use the same depth as the source.'
        },
        'dx': {
            'type': 'int',
            'default': '0',
            'description': 'Order of the derivative x. It specifies the order of the derivative in the x direction.'
        },
        'dy': {
            'type': 'int',
            'default': '1',
            'description': 'Order of the derivative y. It specifies the order of the derivative in the y direction.'
        },
        'ksize': {
            'type': 'uint8',
            'default': '3',
            'description': 'Size of the extended Sobel kernel; it must be 1, 3, 5, or 7. A ksize of -1 corresponds to a 3x3 Scharr filter, which may give more accurate results than a 3x3 Sobel filter.'
        },
        'output': {
            'numbers': '1',
            'description': 'The destination image which will contain the results of the Sobel filter. It will be the same size and the same number of channels as src.'
        }
    }
    return info

def get_absdiff_params_info():
    info = {
        'src1': {
            'format': 'numpy.ndarray',
            'type': 'int8 or float32 or float64',
            'description': 'First input array or a scalar.'
        },
        'src2': {
            'format': 'numpy.ndarray',
            'type': 'int8 or float32 or float64 as same as src1',
            'description': 'Second input array or a scalar. It must have the same size and type as src1 if it is an array.'
        },
        'dst': {
            'format': 'numpy.ndarray',
            'type': 'optional',
            'description': 'Output array that will have the same size and type as the input arrays. This parameter is optional; if not provided, a new array will be created to hold the result.'
        },
        'output': {
            'numbers': '1',
            'description': 'The destination array which will contain the result of the absolute difference between src1 and src2.'
        }
    }
    return info

def get_accumulate_params_info():
    info = {
        'src': {
            'format': 'numpy.ndarray',
            'type': 'CV_8UC(n), CV_16UC(n), CV_32FC(n), CV_64FC(n)',
            'description': 'Input image of type CV_8UC(n), CV_16UC(n), CV_32FC(n) or CV_64FC(n), where n is a positive integer.'
        },
        'mask': {
            'format': 'numpy.ndarray',
            'type': 'optional',
            'description': 'Optional operation mask to specify which pixels are accumulated.'
        },
        'output': {
            'numbers': '1',
            'description': 'The accumulated image which gets updated with the sum of input image and the existing matrix.'
        }
    }
    return info

def get_accumulateProduct_params_info():
    info = {
        'src1': {
            'format': 'numpy.ndarray',
            'type': '8-bit or 32-bit float',
            'description': 'First input image, 1- or 3-channel, 8-bit or 32-bit floating point.'
        },
        'src2': {
            'format': 'numpy.ndarray',
            'type': '8-bit or 32-bit float',
            'description': 'Second input image of the same type and the same size as src1.'
        },

        'mask': {
            'format': 'numpy.ndarray',
            'type': 'optional',
            'description': 'Optional operation mask to specify which pixels in the input images are processed.'
        },
        'output': {
            'numbers': '1',
            'description': 'The destination image which will contain the result of the accumulation of the per-element product of src1 and src2.'
        }
    }
    return info

def get_accumulateSquare_params_info():
    info = {
        'src': {
            'format': 'numpy.ndarray',
            'type': '8-bit or 32-bit float',
            'description': 'Input image as 1- or 3-channel, 8-bit or 32-bit floating point.'
        },
        'mask': {
            'format': 'numpy.ndarray',
            'type': 'optional',
            'description': 'Optional operation mask to specify which pixels in the input image are accumulated.'
        },
        'output': {
            'numbers': '1',
            'description': 'The destination image which will contain the result of the accumulation of the square of the input image.'
        }
    }
    return info

def get_accumulateWeighted_params_info():
    info = {
        'src': {
            'format': 'numpy.ndarray',
            'type': 'float32',
            'description': 'Input image as  8-bit 3-channel, 8-bit or 32-bit floating point.'
        },

        'mask': {
            'format': 'numpy.ndarray',
            'type': 'float32 as same as src',
            'description': 'Input image as 8-bit 3-channel .'
        },
        'alpha': {
            'type': 'float',
            'default': '0.5',
            'description': 'alpha: Weight of the input image. New frame accumulation weight.'
        },
        'output': {
            'numbers': '1',
            'description': 'The destination image which will contain the result of the weighted accumulation of the input image.'
        }
    }
    return info

def get_adaptiveThreshold_params_info():
    info = {
        'src': {
            'format': 'numpy.ndarray',
            'type': 'uint8',
            'description': 'Source 8-bit single-channel image.'
        },
        'maxValue': {
            'type': 'float32',
            'description': 'maxValue Non-zero value assigned to the pixels for which the condition is satisfied.'
        },
        'adaptiveMethod': {
            'type': 'int',
            'default': 'cv2.ADAPTIVE_THRESH_MEAN_C',
            'flag': 'None',
            'description': 'Adaptive thresholding algorithm to use, see AdaptiveThresholdTypes.'
        },
        'thresholdType': {
            'type': 'int',
            'default':'cv2.THRESH_BINARY',
            'flag': 'None',
            'description': 'Thresholding type that must be either cv2.THRESH_BINARY or cv2.THRESH_BINARY_INV.'
        },
        'blockSize': {
            'type': 'uint8',
            'default': '3',
            'description': 'Size of a pixel neighborhood that is used to calculate a threshold value for the pixel: 3, 5, 7, and so on.'
        },
        'C': {
            'type': 'uint8',
            'description': 'Constant subtracted from the mean or weighted mean. Normally, it is positive but may be zero or negative as well.'
        },
        'output': {
            'numbers': '1',
            'description': 'The destination image which will contain the result of the adaptive thresholding. It will be the same size and the same type as src.'
        }
    }
    return info

def get_add_params_info():
    info = {
        'src1': {
            'format': 'numpy.ndarray',
            'type': 'uint8 or float32 or float64',
            'description': 'First input array or a scalar.'
        },
        'src2': {
            'format': 'numpy.ndarray',
            'type': 'as same as src uint8 or float32 or float64',
            'description': 'Second input array or a scalar. It must have the same size and the same number of channels as src1 if it is an array.'
        },
        'output': {
            'numbers': '1',
            'description': 'The destination image which will contain the result of the addition. It will be the same size and the same number of channels as the input arrays.'
        }
    }
    return info

def get_addText_params_info():
    info = {
        'img': {
            'format': 'numpy.ndarray',
            'type': '8-bit 3-channel',
            'description': '8-bit 3-channel image where the text should be drawn.'
        },
        'text': {
            'type': 'str',
            'description': 'Text to write on an image.'
        },
        'org': {
            'format': 'tuple',
            'type': 'int',
            'description': 'Point(x, y) where the text should start on an image.'
        },
        'nameFont': {
            'type': 'str',
            'description': 'Name of the font. The name should match the name of a system font.'
        },
        'pointSize': {
            'type': 'int',
            'default': 'system-dependent',
            'description': 'Size of the font. If not specified, a system-dependent default value is used.'
        },
        'color': {
            'format': 'tuple',
            'type': 'int',
            'description': 'Color of the font in BGRA where A = 255 is fully transparent.'
        },
        'weight': {
            'type': 'int',
            'description': 'Font weight. Can be a value from cv::QtFontWeights or a positive integer.'
        },
        'style': {
            'type': 'QtFontStyles',
            'default': 'normal',
            'description': 'Font style. Can be a value from cv::QtFontStyles.'
        },
        'spacing': {
            'type': 'int',
            'default': '0',
            'description': 'Spacing between characters. It can be negative or positive.'
        },
        'output': {
            'numbers': '0',
            'description': 'This function does not return an output. It modifies the input image in place.'
        }
    }
    return

def get_addWeighted_params_info():
    info = {
        'src1': {
            'format': 'numpy.ndarray',
            'type': 'uint8 or float32 or float64',
            'description': 'First input array.'
        },
        'alpha': {
            'type': 'float',
            'description': 'Weight of the first array elements.'
        },
        'src2': {
            'format': 'numpy.ndarray',
            'type': 'uint8 or float32 or float64 as same as src',
            'description': 'Second input array of the same size and channel number as src1.'
        },
        'beta': {
            'type': 'float',
            'description': 'Weight of the second array elements.'
        },
        'gamma': {
            'type': 'float',
            'description': 'Scalar added to each sum.'
        },
        'dtype': {
            'type': 'int',
            'default': '-1',
            'flag':'None',
            'description': 'Optional depth of the output array. When both input arrays have the same depth, dtype can be set to -1, which will use the same depth as the input arrays.'
        },
        'output': {
            'numbers': '1',
            'description': 'The destination array which will contain the result of the weighted sum.'
        }
    }
    return info

def get_applyColorMap_params_info():
    info = {
        'src': {
            'format': 'numpy.ndarray',
            'type': 'uint8',
            'description': 'The source image, grayscale or colored.'
        },
        'colormap': {
            'type': 'int',
            'defualt': 'cv2.COLORMAP_JET',
            'description': 'The colormap to apply. For the first function variant, it is an int from the ColormapTypes. For the second function variant, it is a user-defined colormap with type CV_8UC1 or CV_8UC3 and size 256.'
        },
        'output': {
            'numbers': '1',
            'description': 'The destination image which will contain the result of applying the colormap to the source image.'
        }
    }
    return info

def get_approxPolyDP_params_info():
    info = {
        'curve': {
            'format': 'numpy.ndarray',
            'type': 'float32',
            'description': 'Input vector of 2D points stored in numpy.ndarray or a list of points.'
        },
        'epsilon': {
            'type': 'float32',
            'description': 'Parameter specifying the approximation accuracy. This is the maximum distance between the original curve and its approximation.'
        },
        'closed': {
            'type': 'bool',
            'description': 'Indicates whether the approximated curve should be closed or not. If true, the first and last vertices are connected.'
        },
        'output': {
            'numbers': '1',
            'description': 'The approximated curve as a numpy.ndarray of 2D points.'
        }
    }
    return info

def get_arcLength_params_info():
    info = {
        'curve': {
            'format': 'numpy.ndarray',
            'type': 'unit8 or float32',
            'description': 'Input vector of 2D points representing the curve or contour, stored in a numpy.ndarray or a list of points.'
        },
        'closed': {
            'type': 'bool',
            'description': 'Flag indicating whether the curve is closed (True) or not (False). A closed curve represents a contour with its endpoints connected.'
        },
        'output': {
            'numbers': '1',
            'description': 'The total length of the curve or the perimeter of the contour.'
        }
    }
    return info

def get_arrowedLine_params_info():
    info = {
        'img': {
            'format': 'numpy.ndarray',
            'type': 'unit8 or float32 or float64',
            'description': 'Image on which the arrow is to be drawn.'
        },
        'pt1': {
            'format': 'tuple',
            'type': 'int',
            'description': 'The point the arrow starts from (x1, y1).'
        },
        'pt2': {
            'format': 'tuple',
            'type': 'int',
            'description': 'The point the arrow points to (x2, y2).'
        },
        'color': {
            'format': 'tuple',
            'type': 'int',
            'description': 'Line color (B,G,R).'
        },
        'thickness': {
            'type': 'int',
            'default': '1',
            'flag': 'None',
            'description': 'Line thickness.'
        },
        'line_type': {
            'type': 'int',
            'default': 'cv2.LINE_8',
            'description': 'Type of the line, whether 8-connected, anti-aliased line etc.'
        },
        'shift': {
            'type': 'int',
            'default': '0',
            'description': 'Number of fractional bits in the point coordinates.'
        },
        'tipLength': {
            'type': 'float',
            'default': '0.1',
            'description': 'The length of the arrow tip in relation to the arrow length.'
        },
        'output': {
            'numbers': '0',
            'description': 'The function does not return a result but draws directly on the image provided.'
        }
    }
    return info

def get_batchDistance_params_info():
    info = {
        'src1': {
            'format': 'numpy.ndarray',
            'type': 'float32',
            'description': 'First batch of input feature vectors.'
        },
        'src2': {
            'format': 'numpy.ndarray',
            'type': 'as same as src float32',
            'description': 'Second batch of input feature vectors.'
        },
        'dtype': {
            'type': 'int',
            'default': 'cv2.CV_32F',
            'flag': 'None',
            'description': 'Output data type for the distances. It can be CV_32S, CV_32F, etc.'
        },
        'dist': {
            'format': 'numpy.ndarray',
            'type': 'unit8',
            'description': 'Output array of distances between pairs of vectors.'
        },
        'nidx': {
            'type': 'unit8',
            'default': '2',
            'flag': 'None',
            'description': 'Output array of indices of the nearest neighbors for each vector.'
        },
        'normType': {
            'type': 'int',
            'default': 'cv2.NORM_L1',
            'flag': 'None',
            'description': 'Type of norm used to compute the distance. It can be cv2.NORM_L2, cv2.NORM_L1, etc.'
        },
        'K': {
            'type': 'int',
            'default': '2',
            'flag':'None',
            'description': 'Number of nearest neighbors to find; if 0, it is taken as the maximum possible, i.e., min(src1.rows, src2.rows).'
        },

        'output': {
            'numbers': '2',
            'description': 'The function returns a tuple of two arrays: distances and indices of the nearest neighbors.'
        }
    }
    return info

def get_bilateralFilter_params_info():
    info = {
        'src': {
            'format': 'numpy.ndarray',
            'type': 'uint8 or float32',
            'description': 'Source image.'
        },
        'd': {
            'type': 'int',
            'description': 'Diameter of each pixel neighborhood used during filtering. If non-positive, it is computed from sigmaSpace.'
        },
        'sigmaColor': {
            'type': 'float',
            'description': 'Filter sigma in the color space. A larger value of the parameter means that farther colors within the pixel neighborhood will be mixed together, resulting in larger areas of semi-equal color.'
        },
        'sigmaSpace': {
            'type': 'float',
            'description': 'Filter sigma in the coordinate space. A larger value of the parameter means that farther pixels will influence each other as long as their colors are close enough. When d>0, it specifies the neighborhood size regardless of sigmaSpace. Otherwise, d is proportional to sigmaSpace.'
        },
        'borderType': {
            'type': 'BorderTypes',
            'default': 'cv2.BORDER_DEFAULT',
            'flag':'None',
            'description': 'Border mode used to extrapolate pixels outside of the image.'
        },
        'output': {
            'numbers': '1',
            'description': 'The destination image which will contain the result of the bilateral filter.'
        }
    }
    return info

def get_bitwise_and_params_info():
    info = {
        'src1': {
            'format': 'numpy.ndarray',
            'type': 'unit8 or float32',
            'description': 'First input array or a scalar.'
        },
        'src2': {
            'format': 'numpy.ndarray',
            'type': 'unit8 or float32',
            'description': 'Second input array or a scalar. It must have the same size and the same number of channels as src1 if it is an array.'
        },
        'output': {
            'numbers': '1',
            'description': 'The destination array which will contain the result of the bitwise AND operation.'
        }
    }
    return info

def get_bitwise_not_params_info():
    info = {
        'src': {
            'format': 'numpy.ndarray',
            'type': 'unit8 or float32 or float64',
            'description': 'Input array.'
        },
        'output': {
            'numbers': '1',
            'description': 'The destination array which will contain the result of the bitwise NOT operation.'
        }
    }
    return info

def get_bitwise_or_params_info():
    info = {
        'src1': {
            'format': 'numpy.ndarray',
            'type': 'unit8 or float32 or float64',
            'description': 'First input array or a scalar.'
        },
        'src2': {
            'format': 'numpy.ndarray',
            'type': 'unit8 or float32 or float64 as same as src',
            'description': 'Second input array or a scalar. It must have the same size and the same number of channels as src1 if it is an array.'
        },
        'output': {
            'numbers': '1',
            'description': 'The destination array which will contain the result of the bitwise OR operation.'
        }
    }
    return info

def get_bitwise_xor_params_info():
    info = {
        'src1': {
            'format': 'numpy.ndarray',
            'type': 'unit8 or float32 or float64',
            'description': 'First input array or a scalar.'
        },
        'src2': {
            'format': 'numpy.ndarray',
            'type': 'unit8 or float32 or float64 as same as src',
            'description': 'Second input array or a scalar. It must have the same size and the same number of channels as src1 if it is an array.'
        },
        'output': {
            'numbers': '1',
            'description': 'The destination array which will contain the result of the bitwise OR operation.'
        }
    }
    return info

def get_blendLinear_params_info():
    info = {
        'src1': {
            'format': 'numpy.ndarray',
            'type': 'unit8 or float32',
            'description': 'First source array.'
        },
        'src2': {
            'format': 'numpy.ndarray',
            'type': 'unit8 or float32 as same as src',
            'description': 'Second source array. It must have the same size and the same number of channels as src1.'
        },
        'weights1': {
            'format': 'numpy.ndarray',
            'type': 'unit8 or float32 as same as src',
            'description': 'Weights for the first source array. Must have the same size as the source arrays.'
        },
        'weights2': {
            'format': 'numpy.ndarray',
            'type': 'unit8 or float32 as same as src',
            'description': 'Weights for the second source array. Must have the same size as the source arrays.'
        },
        'output': {
            'numbers': '1',
            'description': 'The destination array which will contain the result of the blend.'
        }
    }
    return info

def get_blur_params_info():
    info = {
        'src': {
            'format': 'numpy.ndarray',
            'type': 'unit8 or float32 or float64',
            'description': 'Input image; it can have any number of channels, which are processed independently, but the depth should be one of the specified types.'
        },
        'ksize': {
            'format': 'tuple',
            'type': 'int',
            'description': 'Blurring kernel size.'
        },
        'output': {
            'numbers': '1',
            'description': 'The destination image which will contain the blurred version of the input image.'
        }
    }
    return info

def get_borderInterpolate_params_info():
    info = {
        'p': {
            'type': 'uint8',
            'default': '-1',
            'description': '0-based coordinate of the extrapolated pixel along one of the axes, likely <0 or >= len.'
        },
        'len': {
            'type': 'int',
            'default': '1',
            'flag': 'None',
            'description': 'Length of the array along the corresponding axis.'
        },
        'borderType': {
            'type': 'int',
            'default': 'cv2.BORDER_TRANSPARENT',
            'description': 'Border type, one of the BorderTypes, except for BORDER_TRANSPARENT and BORDER_ISOLATED. When borderType==BORDER_CONSTANT, the function always returns -1, regardless of p and len.'
        },
        'output': {
            'numbers': '1',
            'description': 'The computed coordinate of the donor pixel corresponding to the specified extrapolated pixel.'
        }
    }
    return info

def get_boundingRect_params_info():
    info = {
        'array': {
            'format': 'numpy.ndarray',
            'type': 'float32',
            'description': 'Input 2D point set, stored in a numpy.ndarray or a list of points representing the contour. If the input is a grayscale image, non-zero pixels are treated as points.'
        },
        'output': {
            'numbers': '4',
            'description': 'The calculated bounding rectangle of the point set. This is a tuple (x, y, width, height) representing the top-left corner and size of the rectangle.'
        }
    }
    return info

def get_buildOpticalFlowPyramid_params_info():#can not
    info = {
        'img': {
            'format': 'numpy.ndarray',
            'type': 'uint8',
            'description': '8-bit single-channel'
        },
        'winSize': {
            'format': 'tuple',
            'type': 'int',
            'description': 'Window size of optical flow algorithm. It must be not less than winSize argument of calcOpticalFlowPyrLK.'
        },
        'maxLevel': {
            'type': 'int',
            'default': '4',
            'flag': 'None',
            'description': '0-based maximal pyramid level number.'
        },
        'output': {
            'numbers': '3',
            'description': 'The output pyramid of images and the number of levels in the constructed pyramid.'
        }
    }
    return info

def get_calcBackProject_params_info():
    info = {
        'images': {
            'format': 'numpy.ndarray',
            'type': 'uint8',
            'description': 'List of input images (source images) on which the back projection is calculated.'
        },
        'channels': {
            'format': 'tuple',
            'type': 'int',
            'flag': 'None',
            'description': 'List of the dims channels used to compute the back projection. The number of channels must match the histogram dimensionality.'
        },
        'hist': {
            'format': 'numpy.ndarray',
            'type': 'uint8 as same as ',
            'description': 'Input histogram that can be dense or sparse.'
        },
        'ranges': {
            'format': 'tuple',
            'type': 'float32',
            'flag': 'None',
            'default': '[0, 180, 0, 256]',
            'description': 'List of the histogram binning ranges.'
        },
        'scale': {
            'type': 'float',
            'default': '1',
            'flag': 'None',
            'description': 'Optional scale factor for the output back projection.'
        },
        'output': {
            'numbers': '1',
            'description': 'The destination image which will contain the back projection of the histogram.'
        }
    }
    return info

def get_calcCovarMatrix_params_info():
    info = {
        'samples': {
            'format': 'numpy.ndarray',
            'type': 'uint8 or float32',
            'description': '8-bit single-channel The type should match the ctype parameter if specified.'
        },
        'mean': {
            'format': 'numpy.ndarray',
            'type': 'uint8 or float32 as same as',
            'description': '1xN/Nx1 2-channel'
        },
        'flags': {
            'type': 'int',
            'default': '12',
            'flag': 'None',
            'description': 'Operation flags as a combination of CovarFlags.'
        },
        'output': {
            'numbers': '2',
            'description': 'The function returns the covariance matrix and the mean vector.'
        }
    }
    return info

def get_calcHist_params_info():
    info = {
        'images': {
            'format': 'numpy.ndarray',
            'type': 'uint8 or float32',
            'description': 'List of source images; they all should have the same depth, CV_8U or CV_32F, and the same size. Each of them can have an arbitrary number of channels.'
        },
        'channels': {
            'format': 'tuple',
            'type': 'uint8',
            'flag': 'None',
            'description': 'List of the dims channels used to compute the histogram. The index of the channel in images to be used.'
        },
        'mask': {
            'type': 'None',
            'default':'None',
            'flag': 'None',
            'description': 'Optional mask. If the matrix is not empty, it must be an 8-bit array of the same size as images[i]. The non-zero mask elements mark the array elements counted in the histogram.'
        },
        'histSize': {
            'format': 'tuple',
            'type': 'uint8',
            'description': 'Array of histogram sizes in each dimension.'
        },
        'ranges': {
            'format': 'tuple',
            'type': 'uint8',
            'flag': 'None',
            'description': 'Array of the dims arrays of the histogram bin boundaries in each dimension.'
        },
        'output': {
            'numbers': '1',
            'description': 'The result histogram of the calculation.'
        }
    }
    return info

def get_calcOpticalFlowFarneback_params_info():
    info = {
        'prev': {
            'format': 'numpy.ndarray',
            'type': 'uint8',
            'description': 'First 8-bit single-channel input image.'
        },
        'next': {
            'format': 'numpy.ndarray',
            'type': 'uint8 as same as',
            'description': 'Second input image of the same size and the same type as prev 8-bit single-channel.'
        },
        'flow': {
            'format': 'numpy.ndarray',
            'type': 'float32',
            'description': 'Computed flow image that has the same size as prev and type CV_32FC2 2-channel.'
        },
        'pyr_scale': {
            'type': 'float',
            'description': 'Parameter specifying the image scale (<1) to build pyramids for each image.'
        },
        'levels': {
            'type': 'int',
            'description': 'Number of pyramid layers including the initial image.'
        },
        'winsize': {
            'type': 'int',
            'description': 'Averaging window size; the larger the size, the more robust it is to noise.'
        },
        'iterations': {
            'type': 'int',
            'description': 'Number of iterations the algorithm does at each pyramid level.'
        },
        'poly_n': {
            'type': 'int',
            'description': 'Size of the pixel neighborhood used to find polynomial expansion.'
        },
        'poly_sigma': {
            'type': 'float',
            'description': 'Standard deviation of the Gaussian used to smooth derivatives.'
        },
        'flags': {
            'type': 'int',
            'description': 'Operation flags for the function.'
        },
        'output': {
            'numbers': '1',
            'description': 'The output flow matrix with the optical flow vectors for each pixel.'
        }
    }
    return info

def get_calcOpticalFlowPyrLK_params_info():
    info = {
        'prevImg': {
            'format': 'numpy.ndarray',
            'type': 'uint8',
            'description': 'First 8-bit input image or pyramid constructed by buildOpticalFlowPyramid.'
        },
        'nextImg': {
            'format': 'numpy.ndarray',
            'type': 'uint8',
            'description': 'Second input image or pyramid of the same size and the same type as prevImg.'
        },
        'prevPts': {
            'format': 'numpy.ndarray',
            'type': 'float32',
            'description': 'Vector of 2D points for which the flow needs to be found; point coordinates must be single-precision floating-point numbers.'
        },
        'nextPts': {
            'format': 'numpy.ndarray',
            'type': 'float32',
            'description': 'Vector of 2D points for which the flow needs to be found; point coordinates must be single-precision floating-point numbers.'
        },
        'output': {
            'numbers': '3',
            'description': 'The function returns nextPts, status, and err.'
        }
    }
    return info

def get_calibrateCamera_params_info():
    info = {
        'objectPoints': {
            'format': 'numpy.ndarray',
            'type': 'float32',
            'description': 'Array of object points in the object coordinate space, 3xN/Nx3 1-channel or 1xN/Nx1 3-channel,'
        },
        'imagePoints': {
            'format': 'numpy.ndarray',
            'type': 'float32',
            'description': ' 2xN/Nx2 1-channel or 1xN/Nx1 2-channel.'
        },
        'imageSize': {
            'format': 'tuple',
            'type': 'int',
            'description': 'Size of the image used only to initialize the intrinsic camera matrix.'
        },
        'cameraMatrix': {
            'type': 'None',
            'default':'None',
            'flag': 'None',
            'description': 'Input/output camera matrix of size 3x3. Initially, it should be an approximation of the camera matrix (can be set to None if cv2.CALIB_USE_INTRINSIC_GUESS is not set).'
        },
        'distCoeffs': {
            'type': 'None',
            'default': 'None',
            'flag': 'None',
            'description': 'Input/output vector of distortion coefficients (k1, k2, p1, p2[, k3[, k4, k5, k6[, s1, s2, s3, s4[, τx, τy]]]]). The parameter is modified by the function.'
        },
        'output': {
            'number': '5',
        }
    }
    return info

def get_calibrateCameraExtended_params_info():
    info = {
        'objectPoints': {
            'format': 'numpy.ndarray',
            'type': 'float32',
            'description': 'Array of object points in the calibration pattern coordinate space. The array is of the size 1xN/Nx1 3-channel, where N is the number of points in all views.'
        },
        'imagePoints': {
            'format': 'numpy.ndarray',
            'type': 'float32',
            'description': 'Array of corresponding image points, 2xN/Nx2 1-channel or 1xN/Nx1 2-channel, where N is the number of points in all views.'
        },
        'imageSize': {
            'format': 'tuple',
            'type': 'int',
            'description': 'Size of the image used only to initialize the camera intrinsic matrix.'
        },
        'cameraMatrix': {
            'format': 'numpy.ndarray',
            'type': 'float64',
            'description': 'Input/output camera matrix of size 3x3. Initially, it should be an approximation of the camera matrix.'
        },
        'distCoeffs': {
            'format': 'numpy.ndarray',
            'type': 'float64',
            'description': 'Input/output vector of distortion coefficients.'
        },
        'flags': {
            'format': 'int',
            'type': 'int',
            'default': '0',
            'description': 'Different flags for the calibration process.'
        },
        'output': {
            'number': '8',
            'description': {
                'retval': 'The overall RMS re-projection error.',
                'cameraMatrix': 'The output refined camera matrix.',
                'distCoeffs': 'The output refined distortion coefficients.',
                'rvecs': 'The output rotation vectors estimated for each pattern view.',
                'tvecs': 'The output translation vectors estimated for each pattern view.',
                'stdDeviationsIntrinsics': 'Output vector of standard deviations estimated for intrinsic parameters.',
                'stdDeviationsExtrinsics': 'Output vector of standard deviations estimated for extrinsic parameters.',
                'perViewErrors': 'Output vector of the RMS re-projection error estimated for each pattern view.'
            }
        }
    }
    return info

def get_cornerSubPix_params_info():#relateed
    info = {
        'image': {
            'format': 'numpy.ndarray',
            'type': 'uint8 or float32',
            'description': 'Input 8-bit single-channel, 8-bit or float image.'
        },
        'corners': {
            'format': 'numpy.ndarray',
            'type': 'float32',
            'description': 'Initial coordinates of the input corners and refined coordinates provided for output.'
        },
        'ksize': {
            'format': 'tuple',
            'type': 'int',
            'description': 'Half of the side length of the search window. The full search window size will be (winSize*2+1)x(winSize*2+1).'
        },
        'zeroZone': {
            'format': 'tuple',
            'type': 'int',
            'default': '(-1, -1)',
            'flag': 'None',
            'description': 'Half of the size of the dead region in the middle of the search zone over which the summation is not done to avoid possible singularities.'
        },
        'criteria': {
            'format': 'tuple',
            'type': 'int',
            'default': '(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 0.001)',
            'flag': 'None',
            'description': 'Criteria for termination of the iterative process of corner refinement. The process stops either after criteria.maxCount iterations or when the corner position moves by less than criteria.epsilon on some iteration.'
        },
        'output': {
            'number': '1',
            'description': 'Refined corner coordinates after sub-pixel adjustment.'
        }
    }
    return info

def get_decolor_params_info():
    info = {
        'src': {
            'format': 'numpy.ndarray',
            'type': 'uint8',
            'description': 'Input 8-bit 3-channel color image.'
        },
        'output': {
            'number': '2',
            'description': 'The function returns a tuple consisting of the grayscale image and the color boost image.'
        }
    }
    return info

def get_dft_params_info():
    info = {
        'src': {
            'format': 'numpy.ndarray',
            'type': 'float32 or float64',
            'description': 'Input array that could be real or complex.'
        },
        'flags': {
            'type': 'int',
            'default': '0',
            'description': 'Transformation flags, representing a combination of the cv2.DftFlags.'
        },
        'nonzeroRows': {
            'type': 'int',
            'default': '0',
            'description': 'Number of nonzero rows in the input array to consider. It can be used to optimize processing.'
        },
        'output': {
            'number': '1',
            'description': 'The transformed array in the frequency domain.'
        }
    }
    return info

def get_divide_params_info():
    info = {
        'src1': {
            'format': 'numpy.ndarray',
            'type': 'float32, float64, uint8',
            'description': 'First input array or a scalar to be divided by the second input array.'
        },
        'src2': {
            'format': 'numpy.ndarray',
            'type': 'float32, float64, uint8 same as src1',
            'description': 'Second input array or a scalar, must have the same size and type as the first input array if the first input is an array.'
        },
        'scale': {
            'type': 'double',
            'default': '1.0',
            'description': 'Scalar factor applied to the division.'
        },
        'dtype': {
            'type': 'int',
            'default': '-1',
            'flag': 'None',
            'description': 'Optional depth of the output array; if -1, dst will have depth src2.depth(). In array-by-array division, -1 can only be used when src1.depth() == src2.depth().'
        },
        'output': {
            'number': '1',
            'description': 'The result of the division operation.'
        }
    }
    return info

def get_drawMatchesKnn_params_info():
    info = {
        'img1': {
            'format': 'numpy.ndarray',
            'type': 'uint8 or float32',
            'description': 'The first source image.'
        },
        'keypoints1': {
            'type': 'cv2.KeyPoint',
            'description': 'Keypoints from the first source image.'
        },
        'img2': {
            'format': 'numpy.ndarray',
            'type': 'uint8 or float32 as same as img1',
            'description': 'The second source image.'
        },
        'keypoints2': {
            'type': 'cv2.KeyPoint',
            'description': 'Keypoints from the second source image.'
        },
        'color': {
            'format': 'tuple',
            'type': 'int',
            'description': 'Color of the lines to draw matches. If None, colors are drawn randomly.'
        },
        'flag': {
            'type': 'int',
            'default': '2',
            'flag': 'None',
            'description': 'Optional depth of the output array; if -1, dst will have depth src2.depth(). In array-by-array division, -1 can only be used when src1.depth() == src2.depth().'
        },
        'output': {
            'number': '1',
            'description': 'The resulting image where the matches are drawn.'
        }
    }
    return info

def get_exp_params_info():
    info = {
        'src': {
            'format': 'numpy.ndarray',
            'type': 'float32 or float64',
            'description': 'Input array.'
        },
        'output': {
            'number': '1',
            'description': 'The resulting array where the exponential of each element in src has been calculated.'
        }
    }
    return info

def get_extractChannel_params_info():
    info = {
        'src': {
            'format': 'numpy.ndarray',
            'type': 'uint8 or float32',
            'description': 'Input array.'
        },
        'coi': {
            'type': 'int',
            'default': '0',
            'flag': 'None',
            'description': 'Index of the channel to extract (0-based index).'
        },
        'output': {
            'number': '1',
            'description': 'The output array with the extracted channel.'
        }
    }
    return info

def get_filter2D_params_info():
    info = {
        'src': {
            'format': 'numpy.ndarray',
            'type': 'uint8',
            'description': 'Input image, 8-bit 3-channel.'
        },
        'ddepth': {
            'type': 'int',
            'default': 'cv2.CV_32F',
            'flag': 'None',
            'description': 'Desired depth of the destination image.'
        },
        'kernel': {
            'format': 'numpy.ndarray',
            'type': 'float32 or float64',
            'description': 'Convolution kernel (or rather a correlation kernel), a single-channel floating point matrix. If you need to apply different kernels to different channels, split the image into separate color planes and process them individually.'
        },
        'output': {
            'number': '1',
            'description': 'The resulting image where the kernel has been applied.'
        }
    }
    return info

def get_gemm_params_info():
    info = {
        'src1': {
            'format': 'numpy.ndarray',
            'type': 'CV_32FC1, CV_64FC1 (real), CV_32FC2, CV_64FC2 (complex)',
            'description': 'First input matrix to be multiplied, which can be real or complex.'
        },
        'src2': {
            'format': 'numpy.ndarray',
            'type': 'same as src1',
            'description': 'Second input matrix to be multiplied, of the same type as src1.'
        },
        'alpha': {
            'type': 'double',
            'description': 'Weight of the matrix product.'
        },
        'src3': {
            'format': 'numpy.ndarray',
            'type': 'same as src1',
            'description': 'Third optional delta matrix to be added to the matrix product, of the same type as src1 and src2.'
        },
        'beta': {
            'type': 'double',
            'description': 'Weight of src3.'
        },
        'flags': {
            'type': 'int',
            'default': '0',
            'flag': 'None',
            'description': 'Operation flags for the matrix multiplication. This could include flags like cv2.GEMM_1_T or cv2.GEMM_3_T which indicate transposition of src1 or src3 respectively.'
        },
        'output': {
            'number': '1',
            'description': 'The resulting matrix from the generalized matrix multiplication.'
        }
    }
    return info

def get_getRotationMatrix2D_params_info():
    info = {
        'center': {
            'format': 'tuple',
            'type': 'float32',
            'description': 'Center of the rotation in the source image. The coordinates are in the format (x, y) and represent the center point around which the image will be rotated.'
        },
        'angle': {
            'type': 'float',
            'description': 'Rotation angle in degrees. Positive values indicate counter-clockwise rotation, while negative values indicate clockwise rotation. The coordinate origin is assumed to be the top-left corner.'
        },
        'scale': {
            'type': 'float',
            'description': 'Isotropic scale factor. If the scale factor is 1, the output image will be the same size as the input. If it is less than 1, the output image will be smaller, and if greater than 1, it will be larger.'
        },
        'output': {
            'number': '1',
            'description': 'The output 2x3 transformation matrix which can be used to rotate and scale the image.'
        }
    }
    return info

def get_findContours_params_info():
    info = {
        'image': {
            'format': 'numpy.ndarray',
            'type': 'uint8',
            'description': 'Source image, where non-zero pixels are treated as 1’s and zero pixels remain 0’s, thus treated as binary. Can also be a 32-bit integer image of labels if mode is RETR_CCOMP or RETR_FLOODFILL.'
        },
        'mode': {
            'type': 'int',
            'default': 'cv2.RETR_EXTERNAL',
            'flag': 'None',
            'description': 'Contour retrieval mode (e.g., RETR_EXTERNAL, RETR_TREE).'
        },
        'method': {
            'type': 'int',
            'default': 'cv2.CHAIN_APPROX_SIMPLE',
            'flag': 'None',
            'description': 'Contour approximation method (e.g., CHAIN_APPROX_SIMPLE, CHAIN_APPROX_NONE).'
        },
        'output': {
            'number': '2',
            'description': 'The function returns two values: the contours and the hierarchy of the contours.'
        }
    }
    return info

def get_rectangle_params_info():#bug
    info = {
        'img': {
            'format': 'numpy.ndarray',
            'type': 'uint8 or float32',
            'description': 'Image on which the rectangle is to be drawn.'
        },
        'pt1': {
            'format': 'tuple',
            'type': 'int',
            'description': 'Vertex of the rectangle. It is a tuple of 2 integers representing the top left corner.'
        },
        'pt2': {
            'format': 'tuple',
            'type': 'int',
            'description': 'Vertex of the rectangle opposite to pt1. It is a tuple of 2 integers representing the bottom right corner.'
        },
        'color': {
            'format': 'tuple',
            'type': 'int',
            'description': 'Rectangle color or brightness (grayscale image). It is a tuple of 3 integers representing the BGR (Blue, Green, Red) color.'
        },
        'thickness': {
            'type': 'int',
            'default': '1',
            'flag': 'None',
            'description': 'Thickness of the lines that make up the rectangle. Negative values, like cv2.FILLED, mean that the rectangle is filled.'
        },
        'output': {
            'number': '1',
            'description': 'The image with the rectangle drawn on it.'
        }
    }
    return info

def get_cvtColor_params_info():
    info = {
        'src': {
            'format': 'numpy.ndarray',
            'type': 'uint8 or float32',
            'description': 'Input image: 8-bit 3-channel, or single-precision floating-point.'
        },
        'code': {
            'type': 'int',
            'default': '6',
            'flag': 'None',
            'description': 'Color space conversion code that specifies the type of transformation (e.g., COLOR_BGR2GRAY, COLOR_RGB2BGR, COLOR_BGR2HSV).'
        },
        'output': {
            'number': '1',
            'description': 'The output image after the color space conversion.'
        }
    }
    return info

def get_seamlessClone_params_info():#bug
    info = {
        'src': {
            'format': 'numpy.ndarray',
            'type': '8-bit 3-channel',
            'description': 'Input source image that contains the object you want to insert into the destination image.'
        },
        'dst': {
            'format': 'numpy.ndarray',
            'type': '8-bit 3-channel',
            'description': 'Input destination image where the source object will be placed.'
        },
        'mask': {
            'format': 'numpy.ndarray',
            'type': 'float',
            'flag': 'None',
            'description': 'Mask image that defines the region of the source image to be cloned into the destination image. The mask can be a binary mask with non-zero pixels representing the area of the source image to clone.'
        },
        'p': {
            'format': 'tuple',
            'type': 'int',
            'description': 'Point in the destination image where the center of the source object will be placed.'
        },
        'flags': {
            'type': 'int',
            'default': 'cv2.NORMAL_CLONE',
            'flag': 'None',
            'description': 'Cloning method flag that could be NORMAL_CLONE, MIXED_CLONE, or MONOCHROME_TRANSFER. This determines the blending method to be used.'
        },
        'output': {
            'number': '1',
            'description': 'The resulting image where the source image has been cloned into the destination image seamlessly.'
        }
    }
    return info

def get_drawKeypoints_params_info():
    info = {
        'image': {
            'format': 'numpy.ndarray',
            'type': 'uint8',
            'description': 'Source image 8-bit 3-channel on which the keypoints will be drawn.'
        },
        'keypoints': {
            'type': 'KeyPoint',
            'description': 'ORB Keypoints detected in the source image. Each keypoint is specified by its 2D position, size, angle, response, octave (pyramid layer), and class_id.'
        },
        'color': {
            'format': 'tuple',
            'type': 'int',
            'description': 'Color of keypoints. If the color is None, keypoints are drawn in random colors.'
        },
        'output': {
            'number': '1',
            'description': 'The result image with keypoints drawn.'
        }
    }
    return info


def get_resize_params_info():#bug
    info = {
        'src': {
            'format': 'numpy.ndarray',
            'type': 'uint8 or float32 or float64',
            'description': 'Input image.'
        },
        'dsize': {
            'format': 'tuple',
            'type': 'int',
            'description': 'Desired size for the output image. If the tuple is zero, the size will be computed from the source image dimensions, the scaling factors fx and fy, and the interpolation method.'
        },
        'fx': {
            'type': 'double',
            'default': '0',
            'description': 'Scale factor along the horizontal axis; when it equals 0, it is computed as (double)dsize.width/src.cols.'
        },
        'fy': {
            'type': 'double',
            'default': '0',
            'description': 'Scale factor along the vertical axis; when it equals 0, it is computed as (double)dsize.height/src.rows.'
        },
        'interpolation': {
            'type': 'int',
            'default': 'cv2.INTER_AREA',
            'description': 'Interpolation method. Examples include INTER_LINEAR, INTER_CUBIC, INTER_AREA, etc.'
        },
        'output': {
            'number': '1',
            'description': 'The resulting image after resizing.'
        }
    }
    return info

def get_polylines_params_info():
    info = {
        'img': {
            'format': 'numpy.ndarray',
            'type': 'uint8 or float32',
            'description': 'Image on which the polygonal curves will be drawn.'
        },
        'pts': {
            'format': 'numpy.ndarray',
            'type': 'int32',
            'description': 'Array of polygonal curves. Each polygonal curve is represented by a numpy.ndarray of points (x, y).'
        },
        'isClosed': {
            'type': 'bool',
            'description': 'Flag indicating whether the drawn polylines are closed or not. If True, the function draws a line from the last vertex of each curve to its first vertex.'
        },
        'color': {
            'format': 'tuple',
            'type': 'int',
            'description': 'Color of the polyline edges. For grayscale images, this is a scalar value. For color images, it is a tuple of three values representing BGR (Blue, Green, Red).'
        },
        'thickness': {
            'type': 'int',
            'default': '1',
            'description': 'Thickness of the polyline edges. If it is negative (e.g., cv2.FILLED), the polygonal curves are filled.'
        },
        'lineType': {
            'type': 'int',
            'default': 'cv2.LINE_8',
            'description': 'Type of the line segments. Determines how the lines will be drawn. For example, cv2.LINE_8, cv2.LINE_AA, etc.'
        },
        'shift': {
            'type': 'int',
            'default': '0',
            'description': 'Number of fractional bits in the point coordinates. This allows for subpixel accuracy.'
        },
        'output': {
            'number': '1',
            'description': 'The image with the polylines drawn on it.'
        }
    }
    return info

def get_imread_params_info():
    info = {
        'filename': {
            'format': 'string',
            'type': 'str',
            'description': 'Name of the file from which the image is to be loaded.'
        },
        'flags': {
            'type': 'int',
            'default': 'cv2.IMREAD_COLOR',
            'description': 'Flag that specifies the way the image should be read. It defaults to cv2.IMREAD_COLOR, which loads a color image and ignores any transparency. Other flags include cv2.IMREAD_GRAYSCALE and cv2.IMREAD_UNCHANGED, among others.'
        },
        'output': {
            'number': '1',
            'description': 'The result of the image reading operation. It is either the image matrix or None if the reading operation fails.'
        }
    }
    return info

def get_matchTemplate_params_info():#Bug
    info = {
        'image': {
            'format': 'numpy.ndarray',
            'type': 'uint8 or float32',
            'description': 'Image where the search is running. Must be the same data type as templ. 8-bit 3-channel '
        },
        'templ': {
            'format': 'numpy.ndarray',
            'type': 'uint8 or float32 as same as image',
            'description': 'Searched template; must be not greater than the source image and have the same data type as the image. 8-bit 3-channel '
        },
        'method': {
            'type': 'int',
            'default': 'cv2.TM_SQDIFF_NORMED',
            'flag': 'None',
            'description': 'cv2.TM_SQDIFF , cv2.TM_SQDIFF_NORMED, cv2.TM_CCORR, cv2.TM_CCORR_NORMED, cv2.TM_CCOEFF, cv2.TM_CCOEFF_NORMED, Comparison method specified by one of the values in TemplateMatchModes.'
        },
        'output': {
            'number': '1',
            'description': 'The result of the template matching procedure. It is an image where each pixel denotes how much does the neighbourhood of that pixel match with template.'
        }
    }
    return info

def get_imshow_params_info():
    info = {
        'winname': {
            'type': 'str',
            'description': 'Name of the window where the image is to be displayed.'
        },
        'mat': {
            'format': 'numpy.ndarray',
            'type': 'uint8, float32, float64',
            'description': 'The image to be shown. Pixel value scaling is applied based on the image type:'
                           ' 8-bit images are displayed as is; '
                           ' 16-bit images are scaled down by 256; '
                           ' 32-bit and 64-bit floating-point images have their pixel values multiplied by 255.'
        },
        'output': {
            'number': 'None',
            'description': 'This function does not return a value.'
        }
    }
    return info


def get_floodFill_params_info():
    info = {
        'image': {
            'format': 'numpy.ndarray',
            'type': 'uint8 or float32',
            'description': '8-bit 3-channel Input/output image that will be modified by the function unless the FLOODFILL_MASK_ONLY flag is set.'
        },
        'mask': {
            'format': 'numpy.ndarray',
            'type': 'uint8',
            'description': 'Operation mask that must be 2 pixels wider and 2 pixels taller than the image. It is modified to indicate which pixels were filled.'
        },
        'seedPoint': {
            'format': 'tuple',
            'type': 'int',
            'description': 'Starting point for flood filling.'
        },
        'color': {
            'format': 'tuple',
            'type': 'int',
            'description': 'New value (color or brightness) to assign to the repainted domain pixels.'
        },
        'output': {
            'number': '4',
            'description': 'The function returns a 4-element tuple (retval, image, mask, rect), where retval is the number of pixels filled, image is the modified input image, mask is the modified operation mask, and rect is the bounding rectangle of the filled region.'
        }
    }
    return info


def get_putText_params_info():
    info = {
        'img': {
            'format': 'numpy.ndarray',
            'type': 'uint8 or float32 or float64',
            'description': 'The image on which to draw the text.'
        },
        'text': {
            'type': 'str',
            'description': 'The text string to be drawn.'
        },
        'org': {
            'format': 'tuple',
            'type': 'int',
            'description': 'The bottom-left corner of the text string in the image.'
        },
        'fontFace': {
            'type': 'int',
            'description': 'The font type to use for rendering the text.'
        },
        'fontScale': {
            'type': 'float',
            'description': 'The font scale factor that is multiplied by the font-specific base size.'
        },
        'color': {
            'format': 'tuple',
            'type': 'int',
            'description': 'The color of the text to be drawn. For grayscale images, it is a scalar value. For color images, it is a tuple of 3 values representing the BGR (Blue, Green, Red) colors.'
        },
        'thickness': {
            'type': 'int',
            'default': '1',
            'description': 'The thickness of the lines used to draw the text.'
        },
        'lineType': {
            'type': 'int',
            'default': 'cv2.LINE_8',
            'flag': 'None',
            'description': 'The type of the line to use.'
        },
        'bottomLeftOrigin': {
            'type': 'bool',
            'default': 'False',
            'description': 'When true, the image data origin is at the bottom-left corner. Otherwise, it is at the top-left corner.'
        },
        'output': {
            'number': '1',
            'description': 'The image with the text drawn onto it.'
        }
    }
    return info

def get_waitKey_params_info():
    info = {
        'delay': {
            'type': 'int',
            'default': '0',
            'description': 'The number of milliseconds the function will wait for a key event. A value of 0 waits indefinitely for a key press.'
        },
        'output': {
            'number': '1',
            'description': 'The key code of the pressed key. Key codes can be used to identify which key was pressed.'
        }
    }
    return info

def get_UMat_context_params_info():
    info = {
            'output': {
                'number': '1',
                'description': 'The UMat object that is managed by the context. '
                               'When the context block is exited, the UMat object is released and its resources are freed.'
            }
        }
    return info

def get_copyMakeBorder_params_info():
    info = {
        'src': {
            'format': 'numpy.ndarray',
            'type': 'uint8 or float32 or float64',
            'description': 'Source image.'
        },
        'top': {
            'type': 'int',
            'description': 'Number of pixels in the top part of the border.'
        },
        'bottom': {
            'type': 'int',
            'description': 'Number of pixels in the bottom part of the border.'
        },
        'left': {
            'type': 'int',
            'description': 'Number of pixels in the left part of the border.'
        },
        'right': {
            'type': 'int',
            'description': 'Number of pixels in the right part of the border.'
        },
        'borderType': {
            'type': 'int',
            'default': 'cv2.BORDER_CONSTANT',
            'flag': 'None',
            'description': 'Border type. One of cv2.BORDER_CONSTANT, cv2.BORDER_REFLECT, cv2.BORDER_REFLECT_101, cv2.BORDER_REPLICATE, cv2.BORDER_WRAP.'
        },
        'color': {
            'format': 'tuple',
            'type': 'int',
            'default': '(0, 0, 0)',
            'description': 'Border value if borderType==BORDER_CONSTANT. It has as many elements as the number of channels in src.'
        },
        'output': {
            'number': '1',
            'description': 'The output image with the border added around the original image.'
        }
    }
    return info

def get_warpPerspective_params_info():
    info = {
        'src': {
            'format': 'numpy.ndarray',
            'type': 'uint8 or float32 or float64',
            'description': 'Input image.'
        },
        'M': {
            'format': 'numpy.ndarray',
            'type': 'float64',
            'description': 'Transformation matrix.'
        },
        'dsize': {
            'format': 'tuple',
            'type': 'int',
            'description': 'Size of the output image.'
        },

        'borderMode': {
            'type': 'int',
            'default': 'cv2.BORDER_CONSTANT',
            'flag': 'None',
            'description': 'Pixel extrapolation method.'
        },
        'borderValue': {
            'type': 'int',
            'default': '0',
            'flag': 'None',
            'description': 'Value used in case of a constant border; by default, it equals 0.'
        },
        'output': {
            'number': '1',
            'description': 'The output image with the applied perspective transformation.'
        }
    }
    return info

def get_projectPoints_params_info():
    info = {
        'objectPoints': {
            'format': 'numpy.ndarray',
            'type': 'float32',
            'description': 'Array of object points in the world coordinate space. 3x3'
        },
        'rvec': {
            'format': 'numpy.ndarray',
            'type': 'float32',
            'description': ' Rotation 1x3 that, along with tvec, performs a change of basis from world to camera coordinate system.'
        },
        'tvec': {
            'format': 'numpy.ndarray',
            'type': 'float32',
            'description': '3x3 Translation.'
        },
        'cameraMatrix': {
            'format': 'numpy.ndarray',
            'type': 'float32',
            'description': '3x3 Camera intrinsic matrix.'
        },
        'distCoeffs': {
            'format': 'numpy.ndarray',
            'type': 'float32',
            'flag': 'None',
            'description': 'Input of vector distortion coefficients. If empty, zero coefficients are assumed.',
        },
        'output':{
            'number':'2'
        }
    }
    return info

def get_getAffineTransform_params_info():
    info = {
        'src': {
            'format': 'numpy.ndarray',
            'type': 'float32',
            'description': 'Coordinates of triangle vertices in the source image 2x3.'
        },
        'retval': {
            'format': 'numpy.ndarray',
            'type': 'float32',
            'description': 'The output affine transformation, 2x3 floating-point matrix.'
        },
        'output':{
            'number':'1'
        }
    }
    return info

def get_namedWindow_params_info():
    info = {
        'winname': {
            'type': 'str',
            'description': 'Name of the window to be created, used as a window identifier.'
        },
        'flags': {
            'type': 'int',
            'default': 'cv2.WINDOW_NORMAL',
            'flag':'None',
            'description': 'cv.WINDOW_AUTOSIZE | cv.WINDOW_KEEPRATIO | cv.WINDOW_GUI_EXPANDED Flags for the window. Default flags mean the window size is automatically adjusted to fit the displayed image, the image ratio is kept, and an enhanced GUI is used.'
        },
        'output':{
            'number':'1'
        }
    }
    return info

def get_warpAffine_params_info():
    info = {
        'src': {
            'format': 'numpy.ndarray',
            'type': 'float32 or float64',
            'description': 'Input image to be transformed.'
        },
        'Mat': {
            'format': 'numpy.ndarray',
            'type': 'float32 or float64 as same as src',
            'description': '3x2 transformation.'
        },
        'dsize': {
            'format': 'tuple',
            'type': 'int',
            'description': 'Size of the output image (width, height).'
        },
        'output': {
            'number': '1',
            'description': 'The transformed image is returned.'
        }
    }
    return info

def get_fillPoly_params_info():#bug
    info = {
        'img': {
            'format': 'numpy.ndarray',
            'type': 'uint8',
            'description': 'Image on which the polygons will be drawn.'
        },
        'points': {
            'format': 'numpy.ndarray',
            'type': 'int32',
            'description': 'Array of polygons where each polygon is represented as an array of points (x, y).'
        },
        'color': {
            'format': 'tuple',
            'type': 'int',
            'description': 'Polygon color or brightness (grayscale image).'
        },
        'output': {
            'number': '1',
            'description': 'The image with filled polygons is returned.'
        }
    }
    return info

def get_destroyAllWindows_params_info():
    info = {
            'output': {
            'number': '1',
            'description': 'This function does not return any output or have output parameters. It simply closes all open HighGUI windows.'
        }
    }
    return info

def get_VideoWriter_fourcc_params_info():
    info = {
        'c1': {
            'type': 'char',
            'description': 'First character of the four-character code.'
        },
        'c2': {
            'type': 'char',
            'description': 'Second character of the four-character code.'
        },
        'c3': {
            'type': 'char',
            'description': 'Third character of the four-character code.'
        },
        'c4': {
            'type': 'char',
            'description': 'Fourth character of the four-character code.'
        },
        'output': {
            'type': '1',
            'description': 'The fourcc code that represents a specific video codec.'
        }
    }
    return info

def get_kmeans_params_info():
    info = {
        'data': {
            'format': 'numpy.ndarray',
            'type': 'float32',
            'description': 'Data for clustering. An array of N-Dimensional with float coordinates.'
        },
        'K': {
            'type': 'int',
            'default': '5',
            'flag': 'None',
            'description': 'Number of clusters to split the set by.'
        },
        'bestLabels': {
            'type': 'int',
            'default': '1',
            'flag': 'None',
            'description': 'Input/output integer array that stores the cluster indices for every sample.'
        },
        'criteria': {
            'format': 'tuple',
            'type': 'int',
            'default': '(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)',
            'flag': 'None',
            'description': 'The algorithm termination criteria, specifying the maximum number of iterations and/or the desired accuracy.'
        },
        'attempts': {
            'type': 'int',
            'default':'10',
            'flag':'None',
            'description': 'Number of times the algorithm is executed using different initial labellings.'
        },
        'flags': {
            'type': 'int',
            'default': 'cv2.KMEANS_RANDOM_CENTERS',
            'flag': 'None',
            'description': 'Flag that can take values of cv::KmeansFlags.'
        },
        'output': {
            'number': '3'
        }
    }
    return info

def get_setNumThreads_params_info():
    info = {
        'nthreads': {
            'type': 'int',
            'default': '5',
            'description': 'Specifies the number of threads OpenCV will try to use for parallel regions. A value of 1 disables threading optimizations, running functions sequentially. Values less than 0 reset the thread count to the system default.'
        },
        'output': {
            'number': '1',
            'description': 'This function does not return a value but affects the execution of subsequent OpenCV functions by setting the global number of threads used.'
        }
    }
    return info

def get_circle_params_info():#bug
    info = {
        'img': {
            'format': 'numpy.ndarray',
            'type': 'uint8',
            'description': 'Image where the circle is to be drawn. This image gets modified in place.'
        },
        'center': {
            'format': 'tuple',
            'type': 'int',
            'description': 'Center of the circle, represented as a tuple of two integers (x, y).'
        },
        'radius': {
            'type': 'int',
            'default': '3',
            'description': 'Radius of the circle in pixels.'
        },
        'color': {
            'format': 'tuple',
            'type': 'int',
            'description': 'Circle color. For grayscale images, a single integer. For color images, a tuple of three integers (B, G, R).'
        },
        'thickness': {
            'type': 'int',
            'default': '-1',
            'flag': 'None',
            'description': 'Thickness of the circle outline, if positive. Negative values, like cv2.FILLED, indicate that a filled circle is to be drawn.'
        },
        'lineType': {
            'type': 'int',
            'default': 'cv2.LINE_8',
            'flag': 'None',
            'description': 'Type of the circle boundary. See cv2.LineTypes.'
        },
        'shift': {
            'type': 'int',
            'default': '0',
            'description': 'Number of fractional bits in the coordinates of the center and in the radius value.'
        },
        'output': {
            'number': '1',
            'description': 'The image with the drawn circle. Note that the drawing is performed in place, so this is the same image object as the input.'
        }
    }
    return info

def get_merge_params_info():
    info = {
        'mv': {
            'format': 'numpy.ndarray',
            'type': 'Same depth',
            'description': 'Input vector of matrices to be merged. All matrices in mv must have the same size and depth.'
        },

        'output': {
            'number': '1',
            'description': 'The resulting image after merging the input matrices. The output has the same size and depth as the input matrices, with the number of channels being the sum of the channels of all input matrices.'
        }
    }
    return info

def get_flip_params_info():
    info = {
        'src': {
            'format': 'numpy.ndarray',
            'type': 'uint8 or float32',
            'description': 'Input array to be flipped.'
        },
        'flipCode': {
            'type': 'int',
            'description': 'A flag to specify how to flip the array. 0 means flipping around the x-axis (vertical flipping), positive value (e.g., 1) means flipping around the y-axis (horizontal flipping). Negative value (e.g., -1) means flipping around both axes.'
        },
        'output': {
            'number': '1',
            'description': 'The function returns the flipped array.'
        }
    }
    return info

def get_solve_params_info():
    info = {
        'src1': {
            'format': 'numpy.ndarray',
            'type': 'float32 or float64',
            'description': 'Input matrix on the left-hand side of the system.'
        },
        'src2': {
            'format': 'numpy.ndarray',
            'type': 'float32 or float64 as same as src1',
            'description': 'Input matrix on the right-hand side of the system.'
        },
        'flags': {
            'type': 'int',
            'default': 'cv2.DECOMP_LU',
            'flag': 'None',
            'description': 'Solution (matrix inversion) method. Can be one of the DecompTypes like cv2.DECOMP_LU, cv2.DECOMP_CHOLESKY, cv2.DECOMP_SVD, cv2.DECOMP_QR, or cv2.DECOMP_NORMAL.'
        },
        'output': {
            'number': '2',
            'description': 'The solution of the system. This is the same as the dst parameter if it was provided.'
        }
    }
    return info

def get_solveCubic_params_info():
    info = {
        'coeffs': {
            'format': 'tuple',
            'type': 'float32',
            'description': '3x1 Equation coefficients, If it is a 4-element vector, it represents the coefficients of x^3, x^2, x, and the constant term, respectively, in the cubic equation coeffs[0]*x^3 + coeffs[1]*x^2 + coeffs[2]*x + coeffs[3] = 0. If it is a 3-element vector, it assumes the coefficient of x^3 is 1 and represents the remaining coefficients of x^2, x, and the constant term, respectively, in the equation x^3 + coeffs[0]*x^2 + coeffs[1]*x + coeffs[2] = 0.'
        },
        'output': {
            'number': '1 or 3',
            'description': 'The number of real roots found by the function. It can be 0, 1, or 3, indicating how many real solutions exist for the cubic equation.'
        }
    }
    return info

def get_split_params_info():
    info = {
        'm': {
            'format': 'numpy.ndarray',
            'type': 'int8 or float32 or float64',
            'description': 'Input multi-channel array. It can be an image or any multi-dimensional array where each channel is to be split into separate arrays.'
        },
        'output': {
            'number': '1',
            'description': 'The function returns the list of single-channel arrays, each representing one channel of the input multi-channel array. If the mv parameter is provided, it is filled with the output arrays and returned.'
        }
    }
    return info

def get_sort_params_info():
    info = {
        'src': {
            'format': 'numpy.ndarray',
            'type': 'int8 or float32 or float64',
            'description': 'Input single-channel array. This array will be sorted according to the specified flags.'
        },
        'flags': {
            'type': 'int',
            'default': 'cv2.SORT_ASCENDING',
            'flag': 'None',
            'description': 'Operation flags, a combination of SortFlags. These flags determine the sorting order (ascending or descending) and whether rows or columns should be sorted. The flags can be cv2.SORT_EVERY_ROW to sort each row, cv2.SORT_EVERY_COLUMN to sort each column, and cv2.SORT_ASCENDING or cv2.SORT_DESCENDING to control the sorting direction.'
        },
        'output': {
            'number': '1',
            'description': 'The function returns the sorted array. If the dst parameter is provided, it is filled with the sorted elements and returned.'
        }
    }
    return info

def get_sortIdx_params_info():
    info = {
        'src': {
            'format': 'numpy.ndarray',
            'type': 'int8 or float32 or float64',
            'description': 'Input single-channel array. The function will sort the elements of this array based on the specified flags, but instead of rearranging the elements, it will produce an array of indices that correspond to the sorted order.'
        },
        'flags': {
            'type': 'int',
            'default': 'cv2.SORT_ASCENDING',
            'flag': 'None',
            'description': 'Operation flags, a combination of SortFlags. These flags determine the sorting order (ascending or descending) and whether rows or columns should be sorted. The flags can be cv2.SORT_EVERY_ROW to sort each row, cv2.SORT_EVERY_COLUMN to sort each column, and cv2.SORT_ASCENDING or cv2.SORT_DESCENDING to control the sorting direction.'
        },
        'output': {
            'number': '1',
            'description': 'The function returns the output array of indices that represent the sorted order of the input array. This output is essentially the dst parameter if it was provided; otherwise, a new array is created and returned.'
        }
    }
    return info

def get_spatialGradient_params_info():
    info = {
        'src': {
            'format': 'numpy.ndarray',
            'type': 'uint8',
            'description': '8-bit single-channel Input image. The function calculates the spatial gradient (first-order derivative) of this image in both the x and y directions.'
        },
        'ksize': {
            'type': 'int',
            'default': '3',
            'flag': 'None',
            'description': 'Size of the Sobel kernel. For the `cv2.spatialGradient` function, it must be 3.'
        },
        'borderType': {
            'type': 'int',
            'default': 'cv2.BORDER_DEFAULT',
            'flag': 'None',
            'description': 'Pixel extrapolation method, specified as one of the BorderTypes. Only BORDER_DEFAULT (BORDER_REFLECT_101) and BORDER_REPLICATE are supported for this function.'
        },
        'output': {
            'number': '2',
            'description': 'The function returns two output images, dx and dy, representing the first-order derivatives of the input image along the x and y axes, respectively.'
        }
    }
    return info


def get_sqrt_params_info():
    info = {
        'src': {
            'format': 'numpy.ndarray',
            'type': 'int8 or float32 or float64',
            'description': 'Input floating-point array. The function calculates the square root of each element in this array. If the array is multi-channel, each channel is processed independently.'
        },
        'output': {
            'number': '1',
            'description': 'The function returns an array containing the square root of each element from the input array. This is the dst array if it was provided; otherwise, a new array is created and returned.'
        }
    }
    return info

def get_stackBlur_params_info():
    info = {
        'src': {
            'format': 'numpy.ndarray',
            'type': 'int8 or float32',
            'description': 'Input image. The number of channels can be arbitrary, but the depth should be one of CV_8U, CV_16U, CV_16S, or CV_32F.'
        },
        'ksize': {
            'format': 'tuple',
            'type': 'int',
            'description': 'Stack-blurring kernel size. If an integer is provided, it is used for both the width and height. The kernel size must be positive and odd. The function supports non-uniform kernel sizes specified by a tuple of two integers (width, height).'
        },
        'output': {
            'number': '1',
            'description': 'The function returns the blurred image. This is the dst array if it was provided; otherwise, a new array is created and returned.'
        }
    }
    return info

def get_subtract_params_info():
    info = {
        'src1': {
            'format': 'numpy.ndarray',
            'type': 'int8, float32, or float64',
            'description': 'First input array or a scalar. When both src1 and src2 are arrays, they must have the same size and number of channels. When one is a scalar, it is subtracted from each element in the array.'
        },
        'src2': {
            'format': 'numpy.ndarray or scalar',
            'type': 'int8, float32, or float64 as same as src1',
            'description': 'Second input array or a scalar. It should be of the same type and size as src1 if it is an array. If src1 is a scalar and src2 is an array, the scalar is subtracted from each element in src2.'
        },
        'dtype': {
            'type': 'int',
            'default': '-1',
            'flag': 'None',
            'description': 'Optional depth of the output array. When -1, the output array will have the same depth as the input array. Allows for specifying different types for the output array.'
        },
        'output': {
            'number': '1',
            'description': 'The function returns the result of the subtraction. If a dst array is provided, it is used to store the output; otherwise, a new array is created and returned.'
        }
    }
    return info

def get_trace_params_info():
    info = {
        'mtx': {
            'format': 'numpy.ndarray',
            'type': 'int8, float32, or float64',
            'description': 'Input matrix. The function computes the sum of the diagonal elements of this matrix.'
        },
        'output': {
            'number': '1',
            'description': 'The function returns the trace of the matrix, which is the sum of its diagonal elements. This is returned as a single scalar value.'
        }
    }
    return info


def get_transform_params_info():
    info = {
        'src': {
            'format': 'numpy.ndarray',
            'type': 'float32',
            'description': 'matrix Input array that must have as many channels (1 to 4) as m.cols or m.cols-1. This array represents the elements (e.g., points, colors) to be transformed.'
        },
        'm': {
            'format': 'numpy.ndarray',
            'type': 'float32',
            'description': 'Transformation 2x2 or 2x3 floating-point matrix. The number of rows in the matrix (2) determines the dimensionality of the output vectors, while the number of columns (equal to the number of channels in src or one more) determines how the transformation is applied.'
        },
        'output': {
            'number': '1',
            'description': 'The function returns the transformed array, dst. If the dst parameter is provided, it is filled with the transformed elements; otherwise, a new array is created and returned.'
        }
    }
    return info

def get_transpose_params_info():
    info = {
        'src': {
            'format': 'numpy.ndarray',
            'type': 'int8, float32, or float64',
            'description': 'Input array. The function transposes this matrix, meaning it swaps its rows and columns.'
        },
        'output': {
            'number': '1',
            'description': 'The function returns the transposed array, dst. If the dst parameter is provided, it is filled with the transposed matrix; otherwise, a new array is created and returned.'
        }
    }
    return info

def get_vconcat_params_info():
    info = {
        'src': {
            'format': 'numpy.ndarray',
            'type': 'int8, float32, or float64',
            'description': 'sequence of 8-bit 3-channel Input array or vector of matrices to be concatenated vertically. All of the matrices must have the same number of columns and the same depth.'
        },
        'output': {
            'number': '1',
            'description': 'The function returns the vertically concatenated array, dst. If the dst parameter is provided, it is filled with the result of the concatenation; otherwise, a new array is created and returned.'
        }
    }
    return info

def get_setIdentity_params_info():
    info = {
        'mtx': {
            'format': 'numpy.ndarray',
            'type': 'int8, float32, or float64',
            'description': 'Matrix to initialize as a scaled identity matrix. The matrix does not need to be square.'
        },
        'output': {
            'number': '1',
            'description': 'The function returns the input matrix after it has been modified in-place to become a scaled identity matrix.'
        }
    }
    return info

def get_scaleAdd_params_info():
    info = {
        'src1': {
            'format': 'numpy.ndarray',
            'type': 'int8, float32, or float64',
            'description': 'First input array. This array is scaled by the factor alpha before addition.'
        },
        'alpha': {
            'type': 'float',
            'description': 'Scale factor for the first array. Each element of src1 is multiplied by this factor before being added to the corresponding element of src2.'
        },
        'src2': {
            'format': 'numpy.ndarray',
            'type': 'int8, float32, or float64 as same as src1',
            'description': 'Second input array of the same size and type as src1. The elements of this array are added to the scaled elements of src1.'
        },
        'output': {
            'number': '1',
            'description': 'The function returns the destination array containing the result of the operation, which is the sum of the scaled src1 and src2. If the dst parameter is provided, it is used to store the output; otherwise, a new array is created and returned.'
        }
    }
    return info

def get_repeat_params_info():
    info = {
        'src': {
            'format': 'numpy.ndarray',
            'type': 'int8, float32, or float64',
            'description': 'Input array to be replicated.'
        },
        'ny': {
            'type': 'int',
            'description': 'Number of times the `src` is repeated along the vertical axis.'
        },
        'nx': {
            'type': 'int',
            'description': 'Number of times the `src` is repeated along the horizontal axis.'
        },
        'output': {
            'number': '1',
            'description': 'The function returns the output array containing the repeated copies of the input array. If the dst parameter is provided, it is used to store the output; otherwise, a new array is created and returned.'
        }
    }
    return info

def get_reduce_params_info():
    info = {
        'src': {
            'format': 'numpy.ndarray',
            'type': 'int8, float32, or float64',
            'description': 'Input 2D matrix. '
        },
        'dim': {
            'type': 'int',
            'default': '0',
            'flag': 'None',
            'description': 'Dimension index along which the matrix is reduced. 0 means that the matrix is reduced to a single row. 1 means that the matrix is reduced to a single column.'
        },
        'rtype': {
            'type': 'int',
            'default': 'cv2.REDUCE_SUM',
            'flag': 'None',
            'description': 'Reduction operation that could be one of ReduceTypes (e.g., cv2.REDUCE_SUM, cv2.REDUCE_AVG, cv2.REDUCE_MAX, cv2.REDUCE_MIN, cv2.REDUCE_SUM2). The choice of operation affects the way the matrix is reduced.'
        },
        'output': {
            'number': '1',
            'description': 'The function returns the reduced vector (dst). If the dst parameter is provided, it is filled with the result of the reduction; otherwise, a new array is created and returned.'
        }
    }
    return info

def get_randn_params_info():
    info = {
        'dst': {
            'format': 'numpy.ndarray',
            'type': 'float32',
            'description': 'Output array of random numbers; the array must be pre-allocated and can have from 1 to 4 channels. The function fills this array with normally distributed random numbers, adjusted to fit the value range of the output array data type.'
        },
        'mean': {
            'type': 'float',
            'description': 'Mean value (expectation) of the generated random numbers. This value specifies the center of the distribution.'
        },
        'stddev': {
            'type': 'float',
            'description': 'Standard deviation of the generated random numbers. This value determines the width of the distribution. It can be specified as either a vector (implying a diagonal standard deviation matrix) or a square matrix, defining how the standard deviation varies for each channel.'
        },
        'output': {
            'number': '1',
            'description': 'The function returns the dst array after filling it with random numbers. The elements of dst will have values distributed according to the normal distribution specified by the mean and stddev parameters.'
        }
    }
    return info

def get_randu_params_info():
    info = {
        'dst': {
            'format': 'numpy.ndarray',
            'type': 'float32',
            'description': 'Output array of random numbers; the array must be pre-allocated and can have from 1 to 4 channels. The function fills this array with normally distributed random numbers, adjusted to fit the value range of the output array data type.'
        },
        'low': {
            'type': 'float',
            'default': '1',
            'flag': 'None',
            'description': 'Mean value (expectation) of the generated random numbers. This value specifies the center of the distribution.'
        },
        'high': {
            'type': 'float',
            'default': '100',
            'flag': 'None',
            'description': 'Standard deviation of the generated random numbers. This value determines the width of the distribution. It can be specified as either a vector (implying a diagonal standard deviation matrix) or a square matrix, defining how the standard deviation varies for each channel.'
        },
        'output': {
            'number': '1',
            'description': 'The function returns the dst array after filling it with random numbers. The elements of dst will have values distributed according to the normal distribution specified by the mean and stddev parameters.'
        }
    }
    return info

def get_pyrUp_params_info():
    info = {
        'src': {
            'format': 'numpy.ndarray',
            'type': 'int8, float32, or float64',
            'description': 'Input image to be upsampled. The function increases the size of this image, usually by a factor of 2.'
        },
        'output': {
            'number': '1',
            'description': 'The function returns the output image `dst` after the upsampling and blurring operations. If the `dst` parameter is provided, it is filled with the result; otherwise, a new array is created and returned.'
        }
    }
    return info

def get_pyrDown_params_info():
    info = {
        'src': {
            'format': 'numpy.ndarray',
            'type': 'int8, float32, or float64',
            'description': '8-bit 3-channel Input image to be downsampled. The function decrease the size of this image, usually by a factor of 2.'
        },
        'output': {
            'number': '1',
            'description': 'The function returns the output image `dst` after the upsampling and blurring operations. If the `dst` parameter is provided, it is filled with the result; otherwise, a new array is created and returned.'
        }
    }
    return info


def get_multiply_params_info():
    info = {
        'src1': {
            'format': 'numpy.ndarray',
            'type': 'int8, float32, or float64',
            'description': 'First input array. Each element of this array is multiplied by the corresponding element in src2, potentially scaled by the scale factor.'
        },
        'src2': {
            'format': 'numpy.ndarray',
            'type': 'int8, float32, or float64 as same as src1',
            'description': 'Second input array of the same size and type as src1. Each element of this array is multiplied by the corresponding element in src1, potentially scaled by the scale factor.'
        },
        'output': {
            'number': '1',
            'description': 'The function returns the output array dst, containing the scaled per-element products of src1 and src2. If the dst parameter is provided, it is filled with the result; otherwise, a new array is created and returned.'
        }
    }
    return info

def get_minMaxLoc_params_info():
    info = {
        'src': {
            'format': 'numpy.ndarray',
            'type': 'int8, float32, or float64',
            'description': 'Input single-channel array. The function searches for the minimum and maximum element values within this array.'
        },
        'output': {
            'number': '4',
            'description': 'The function returns the output image `dst` after the upsampling and blurring operations. If the `dst` parameter is provided, it is filled with the result; otherwise, a new array is created and returned.'
        }
    }
    return info

def get_minEnclosingTriangle_params_info():
    info = {
        'points': {
            'format': 'numpy.ndarray',
            'type': 'float32',
            'description': 'Input vector of 2D points, which can be stored in a numpy.ndarray or a list. The points can be of depth CV_32S (32-bit integer) or CV_32F (32-bit floating point).'
        },
        'output': {
            'number': '2',
        }
    }
    return info

def get_minEnclosingCircle_params_info():
    info = {
        'points': {
            'format': 'numpy.ndarray',
            'type': 'float32',
            'description': 'Input vector of 2D points. These points can be stored in a numpy.ndarray or a list and represent the set of points for which the minimal enclosing circle will be found.'
        },
        'output': {
            'number': '2',
        }
    }
    return info

def get_minAreaRect_params_info():
    info = {
        'points': {
            'format': 'numpy.ndarray',
            'type': 'float32',
            'description': 'Input vector of 2D points. These points can be stored in a numpy.ndarray or a list. The function finds the smallest rectangle that encloses all the points in this set. This rectangle can be rotated.'
        },
        'output': {
            'number': '1',
        }
    }
    return info

def get_min_params_info():
    info = {
        'src1': {
            'format': 'numpy.ndarray',
            'type': 'int8, float32, or float64',
            'description': 'First input array. It can be compared with another array or a scalar to find the per-element minimum.'
        },
        'src2': {
            'format': 'numpy.ndarray',
            'type': 'int8, float32, or float64 as same as src1',
            'description': 'Second input, which can either be an array of the same size and type as src1 or a scalar value. The function computes the minimum of src1 and src2 on an element-by-element basis.'
        },
        'output': {
            'number': '1',
        }
    }
    return info

def get_max_params_info():
    info = {
        'src1': {
            'format': 'numpy.ndarray',
            'type': 'int8, float32, or float64',
            'description': 'First input array. It can be compared with another array or a scalar to find the per-element minimum.'
        },
        'src2': {
            'format': 'numpy.ndarray',
            'type': 'int8, float32, or float64 as same as src1',
            'description': 'Second input, which can either be an array of the same size and type as src1 or a scalar value. The function computes the minimum of src1 and src2 on an element-by-element basis.'
        },
        'output': {
            'number': '1',
        }
    }
    return info

def get_mean_params_info():
    info = {
        'src': {
            'format': 'numpy.ndarray',
            'type': 'int8, float32, or float64',
            'description': 'Input array for which the mean value is calculated. The array should have from 1 to 4 channels so that the result can be stored in a Scalar.'
        },
        'output': {
            'number': '1',
            'description': 'The function returns the mean value as a Scalar. Each element of the Scalar corresponds to the mean value of a channel of the input array. If all elements of the mask are 0, the function returns Scalar::all(0).'
        }
    }
    return info

def get_meanShift_params_info():
    info = {
        'probImage': {
            'format': 'numpy.ndarray',
            'type': 'int8, float32, or float64',
            'description': 'Back projection of the object histogram. This is a grayscale image where each pixel denotes how much does the pixels color match the objects color distribution. For better results, the image should be pre-processed to remove noise, such as by using morphological operations.'
        },
        'window': {
            'format': 'tuple',
            'type': 'int',
            'description': 'Initial search window. This is a rectangle defined as (x, y, w, h) where (x, y) is the top-left corner, and (w, h) are the width and height of the rectangle.'
        },
        'criteria': {
            'type': 'int',
            'default': '(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)',
            'flag': 'None',
            'description': 'Stop criteria for the iterative search algorithm. It can be a combination of the maximum number of iterations and the minimum shift in the window position for the algorithm to terminate. This is defined as cv2.TermCriteria (type, maxCount, epsilon).'
        },
        'output': {
            'number': '1',
            'description': 'The function returns the mean value as a Scalar. Each element of the Scalar corresponds to the mean value of a channel of the input array. If all elements of the mask are 0, the function returns Scalar::all(0).'
        }
    }
    return info


def get_meanStdDev_params_info():
    info = {
        'src': {
            'format': 'numpy.ndarray',
            'type': 'int8, float32, or float64',
            'description': 'Input array for which the mean and standard deviation are calculated. The array should have from 1 to 4 channels so that the results can be stored in Scalars.'
        },
        'output': {
            'number': '2',
            'description': 'The function returns the mean and standard deviation of the specified array elements or array elements within the mask if provided. The results are returned as two Scalars or numpy arrays, one for mean and one for standard deviation, corresponding to each channel of the input array.'
        }
    }
    return info

def get_logPolar_params_info():
    info = {
        'src': {
            'format': 'numpy.ndarray',
            'type': 'int8, float32, or float64',
            'description': 'Source image to be transformed to semilog-polar coordinates space.'
        },
        'center': {
            'format': 'tuple',
            'type': 'int8',
            'description': 'The transformation center in the source image; where the output precision is maximal. It is a point (x, y) around which the transformation is applied.'
        },
        'input': {
            'type': 'int',
            'description': '3x2'
        },
        'flags': {
            'type': 'int',
            'default': 'cv2.INTER_LINEAR',
            'description': '(cv2.INTER_LINEAR or cv2.INTER_NEAREST) and optional flags (cv2.WARP_FILL_OUTLIERS and/or cv2.WARP_INVERSE_MAP).'
        },
        'output': {
            'number': '1',
            'description': 'The function returns the destination image after applying the semilog-polar transformation. This image will have the same size and type as the source image.'
        }
    }
    return info

def get_log_params_info():
    info = {
        'src': {
            'format': 'numpy.ndarray',
            'type': 'int8, float32, or float64',
            'description': 'Source image to be transformed to semilog-polar coordinates space.'
        },
        'output': {
            'number': '1',
            'description': 'The function returns the destination image after applying the semilog-polar transformation. This image will have the same size and type as the source image.'
        }
    }
    return info

def get_isContourConvex_params_info():
    info = {
        'contour': {
            'format': 'numpy.ndarray',
            'type': 'float32',
            'description': 'Input vector of 2D points representing the contour. These points can be stored in a numpy.ndarray or a list. The contour must be simple, meaning it should not have self-intersections, for the function to determine its convexity accurately.'
        },
        'output': {
            'number': '1',
            'description': 'The function returns a boolean value. True indicates that the contour is convex, and False indicates that it is not convex.'
        }
    }
    return info

def get_invert_params_info():
    info = {
        'src': {
            'format': 'numpy.ndarray',
            'type': 'int8, float32, or float64',
            'description': 'Input floating-point M x N matrix to be inverted.'
        },

        'output': {
            'number': '2',
            'description': 'The function returns a boolean value. True indicates that the contour is convex, and False indicates that it is not convex.'
        }
    }
    return info

def get_insertChannel_params_info():
    info = {
        'src1': {
            'format': 'numpy.ndarray',
            'type': 'uint8, float32, or float64',
            'description': '8-bit single-channel. This array should be of the same size as one channel of the dst array.'
        },
        'src2': {
            'format': 'numpy.ndarray',
            'type': 'uint8, float32, or float64 as same as src1',
            'description': '8-bit 3-channel'
        },
        'coi': {
            'type': 'int',
            'default': '0',
            'flag': 'None',
            'description': 'Index of the channel where the src array is to be inserted. This index is 0-based.'
        },
        'output': {
            'number': '1',
            'description': 'The function updates the dst array by inserting the src channel at the specified index (coi) and returns it. The dst parameter is modified in place.'
        }
    }
    return info

def get_idct_params_info():
    info = {
        'src': {
            'format': 'numpy.ndarray',
            'type': 'int8, float32, or float64',
            'description': 'Input floating-point single-channel array. This is the source data for which the inverse DCT is to be calculated.'
        },
        'output': {
            'number': '1',
            'description': 'The function returns the output array containing the inverse DCT of the input array. If the dst parameter is provided, it is filled with the result; otherwise, a new array is created and returned.'
        }
    }
    return info

def get_hconcat_params_info():
    info = {
        'src': {
            'format': 'numpy.ndarray',
            'type': 'float32',
            'description': 'sequence of 8-bit 3-channel Input array or vector of matrices to be concatenated horizontally. All the matrices in `src` must have the same number of rows and the same depth.'
        },
        'output': {
            'number': '1',
            'description': 'The function returns the horizontally concatenated array, `dst`. If the `dst` parameter is provided, it is filled with the result of the concatenation; otherwise, a new array is created and returned.'
        }
    }
    return info

def get_hasNonZero_params_info():
    info = {
        'src': {
            'format': 'numpy.ndarray',
            'type': 'int8, float32, or float64',
            'description': 'Single-channel array to be checked for non-zero elements.'
        },
        'output': {
            'number': '1',
            'description': 'The function returns a boolean value. True if there is at least one non-zero element in the array, False otherwise.'
        }
    }
    return info

def get_getRotationMatrix2D_params_info():
    info = {
        'center': {
            'format': 'tuple',
            'type': 'int8',
            'description': 'Center of the rotation in the source image. This is a point represented as a tuple of two floating-point values (x, y), where (x, y) are the coordinates of the center point around which the image will be rotated.'
        },
        'angle': {
            'type': 'float',
            'description': 'Rotation angle in degrees. Positive values indicate counter-clockwise rotation, while negative values indicate clockwise rotation. The coordinate origin is assumed to be the top-left corner of the image.'
        },
        'scale': {
            'type': 'float',
            'description': 'Isotropic scale factor. This value allows you to uniformly scale the image during rotation. A value of 1.0 means no scaling; greater than 1.0 means enlargement; between 0 and 1.0 means downsizing.'
        },
        'output': {
            'number': '1',
            'description': 'The function returns the affine transformation matrix as a 2x3 numpy.ndarray. This matrix can then be passed to warpAffine or other functions to perform the actual image rotation and scaling.'
        }
    }
    return info

def get_getGaussianKernel_params_info():
    info = {
        'ksize': {
            'type': 'int',
            'description': 'Aperture size. It should be odd ($texttt{ksize} \mod 2 = 1$) and positive. The size of the kernel determines the extent of the smoothing effect.'
        },
        'sigma': {
            'type': 'float',
            'description': 'Gaussian standard deviation. If it is non-positive, it is computed from ksize using the formula `sigma = 0.3*((ksize-1)*0.5 - 1) + 0.8`. This parameter controls the spread of the Gaussian kernel and, consequently, the degree of smoothing.'
        },
        'output': {
            'number': '1',
            'description': 'The function returns the $exttt{ksize} times 1$ matrix of Gaussian filter coefficients. These coefficients are calculated based on the provided kernel size and sigma values.'
        }
    }
    return info

def get_flipND_params_info():
    info = {
        'src': {
            'format': 'numpy.ndarray',
            'type': 'int8, float32, or float64',
            'description': 'Input array. This is the n-dimensional array to be flipped.'
        },
        'axis': {
            'type': 'int',
            'flag':'None',
            'default': '0',
            'description': 'Axis along which the flip is performed. The value of axis must be in the range $0 \leq axis < src.dims$, where $src.dims$ is the number of dimensions in the source array.'
        },
        'output': {
            'number': '1',
            'description': 'The function returns the flipped array. If the dst parameter is provided, it is filled with the result of the flip; otherwise, a new array is created and returned.'
        }
    }
    return info

def get_fitEllipseDirect_params_info():
    info = {
        'points': {
            'format': 'numpy.ndarray',
            'type': 'float32',
            'description': 'Input 2D point set. These points can be stored in a numpy.ndarray or a list. The function calculates the best-fitting ellipse for these points.'
        },
        'output': {
            'number': '1',
            'description': 'The function returns a RotatedRect object. This object represents the rotated rectangle in which the fitted ellipse is inscribed. The RotatedRect contains information about the center of the ellipse, the dimensions (width and height) of the ellipse, and the angle of rotation of the ellipse.'
        }
    }
    return info

def get_fitEllipseAMS_params_info():
    info = {
        'points': {
            'format': 'numpy.ndarray',
            'type': 'float32',
            'description': 'Input 2D point set. These points can be stored in a numpy.ndarray or a list. The function calculates the best-fitting ellipse for these points using the AMS method, which is particularly designed to fit ellipses to data accurately.'
        },
        'output': {
            'number': '1',
            'description': 'The function returns a RotatedRect object. This object represents the rotated rectangle in which the fitted ellipse is inscribed. The RotatedRect contains information about the center of the ellipse, the dimensions (major and minor axes lengths) of the ellipse, and the angle of rotation of the ellipse.'
        }
    }
    return info

def get_fitEllipse_params_info():
    info = {
        'points': {
            'format': 'numpy.ndarray',
            'type': 'float32',
            'description': 'Input 2D point set. These points can be stored in a numpy.ndarray or a list. The function calculates the best-fitting ellipse for these points by minimizing the sum of the squared distances (in a least-squares sense) from the points to the ellipse’s circumference.'
        },
        'output': {
            'number': '1',
            'description': 'The function returns a RotatedRect object. This object represents the rotated rectangle in which the fitted ellipse is inscribed. The RotatedRect contains information about the center of the ellipse, the dimensions (width and height, which correspond to the major and minor axes of the ellipse) of the ellipse, and the angle of rotation of the ellipse.'
        }
    }
    return info

def get_findNonZero_params_info():
    info = {
        'src': {
            'format': 'numpy.ndarray',
            'type': 'int8, float32, or float64',
            'description': 'Single-channel input array, typically a binary image obtained from operations like thresholding, comparisons, etc. The function will find and return the locations of all non-zero pixels in this array.'
        },
        'output': {
            'number': '1',
            'description': 'The function returns the locations of non-zero pixels in the `src` array. These locations are returned through the `idx` parameter, which can be a cv::Mat in C++ or a numpy.ndarray (or list of Points) in Python, containing the coordinates of each non-zero pixel.'
        }
    }
    return info

def get_fastAtan2_params_info():
    info = {
        'x': {
            'format': 'numpy.ndarray',
            'type': 'float32',
            'description': 'The x-coordinate of the vector. This represents the horizontal component of the vector for which the angle to the positive x-axis is being calculated.'
        },
        'y': {
            'format': 'numpy.ndarray',
            'type': 'float32',
            'description': 'The y-coordinate of the vector. This represents the vertical component of the vector for which the angle to the positive x-axis is being calculated.'
        },
        'output': {
            'number': '1',
            'description': 'The function returns the angle of the vector (x, y) in degrees. The angle is measured in the clockwise direction from the positive x-axis, varying from 0 to 360 degrees with an accuracy of about 0.3 degrees.'
        }
    }
    return info

def get_equalizeHist_params_info():
    info = {
        'src': {
            'format': 'numpy.ndarray',
            'type': 'uint8',
            'description': 'Source image. It must be an 8-bit single-channel image, typically a grayscale image.'
        },
        'output': {
            'number': '1',
            'description': 'The function returns the histogram-equalized image. If the dst parameter is provided, it is filled with the result of the histogram equalization; otherwise, a new array is created and returned.'
        }
    }
    return info


def get_determinant_params_info():
    info = {
        'mtx': {
            'format': 'numpy.ndarray',
            'type': 'float32',
            'description': 'Input matrix for which the determinant is to be calculated. The matrix must be of type CV_32FC1 (32-bit floating-point single channel) or CV_64FC1 (64-bit floating-point single channel) and must have a square size (the same number of rows and columns).'
        },
        'output': {
            'number': '1',
            'description': 'The function returns the determinant of the specified matrix as a floating-point value. The determinant can be used to determine if the matrix is invertible (a non-zero determinant means the matrix is invertible).'
        }
    }
    return info

def get_dct_params_info():
    info = {
        'src': {
            'format': 'numpy.ndarray',
            'type': 'float32',
            'description': 'Input floating-point array. This can be a 1D vector or a 2D matrix, and it represents the data on which the DCT will be applied. The function performs differently based on the dimensionality and flags.'
        },
        'output': {
            'number': '1',
            'description': 'The function returns the output array containing the result of the DCT. If the dst parameter is provided, it is filled with the result; otherwise, a new array is created and returned.'
        }
    }
    return info

def get_convertScaleAbs_params_info():
    info = {
        'src': {
            'format': 'numpy.ndarray',
            'type': 'int8, float32, or float64',
            'description': 'Input array. This array undergoes scaling and absolute value calculation. The function processes each element of the input array independently, applicable to both single-channel and multi-channel arrays.'
        },
        'alpha': {
            'type': 'double',
            'default': '1',
            'description': 'Optional scale factor. Each element of the input array is first multiplied by this factor.'
        },
        'beta': {
            'type': 'double',
            'default': '0',
            'description': 'Optional delta added to the scaled values. This value is added to each element of the scaled input array before converting it to an absolute value.'
        },
        'output': {
            'number': '1',
            'description': 'The function returns the output array containing the scaled, absolute, and 8-bit converted values from the input array. If the dst parameter is provided, it is filled with the result; otherwise, a new array is created and returned.'
        }
    }
    return info

def get_convertPointsToHomogeneous_params_info():
    info = {
        'points': {
            'format': 'numpy.ndarray',
            'type': 'int8, float32, or float64',
            'description': 'Input vector of N-dimensional points. This can be a list of points or a numpy.ndarray with points. Each point is represented as a tuple or array of coordinates in Euclidean space.'
        },
        'output': {
            'number': '1',
            'description': 'The function returns the output array containing the points converted to homogeneous space. If the dst parameter is provided, it is filled with the result; otherwise, a new array is created and returned.'
        }
    }
    return info

def get_convertPointsFromHomogeneous_params_info():
    info = {
        'points': {
            'format': 'numpy.ndarray',
            'type': 'int8, float32, or float64',
            'description': 'Input vector of N-dimensional points in homogeneous space. This can be a list of points or a numpy.ndarray, where each point is represented as an array of coordinates, including the homogeneous coordinate.'
        },
        'output': {
            'number': '1',
            'description': 'The function returns the output array containing the points converted to Euclidean space. If the dst parameter is provided, it is filled with the result; otherwise, a new array is created and returned.'
        }
    }
    return info

def get_convertFp16_params_info():
    info = {
        'src': {
            'format': 'numpy.ndarray',
            'type': 'float32',
            'description': 'Input array. The function supports converting from FP32 (single precision floating point, CV_32F) to FP16 (half precision floating point, represented as CV_16S) and vice versa. The input array must have a type of CV_32F for conversion to FP16 or CV_16S for conversion back to FP32.'
        },
        'output': {
            'number': '1',
            'description': 'The function returns the output array containing the converted values in FP16 or FP32 format, depending on the input array type. If the dst parameter is provided, it is filled with the result; otherwise, a new array is created and returned.'
        }
    }
    return info

def get_sumElems_params_info():
    info = {
        'src': {
            'format': 'numpy.ndarray',
            'type': 'int8, float32, or float64',
            'description': 'Input array for which the sum of elements is calculated. This array must have from 1 to 4 channels, and the function calculates the sum for each channel independently.'
        },
        'output': {
            'number': '1',
            'description': 'The function returns the sum of array elements as a Scalar. Each element of the Scalar corresponds to the sum of a specific channel of the input array. For a single-channel array, the result will be a Scalar with only one element.'
        }
    }
    return info

def get_preCornerDetect_params_info():
    info = {
        'src': {
            'format': 'numpy.ndarray',
            'type': 'uint8 or float32',
            'description': 'Source image. This is a single-channel image, either 8-bit or floating-point. It is the input image for which the feature map for corner detection is calculated.'
        },
        'ksize': {
            'type': 'int',
            'description': 'Aperture size of the Sobel derivative used. It must be odd and greater than 1. A larger aperture size will find features that are more like corners.'
        },
        'output': {
            'number': '1',
            'description': 'The function returns the destination image containing the corner detection feature map. High values in the map correspond to regions of the image with high corner-like responses. To identify actual corners, look for local maxima in this map.'
        }
    }
    return info

def get_patchNaNs_params_info():
    info = {
        'a': {
            'format': 'numpy.ndarray',
            'type': 'float32',
            'description': 'Input/output matrix of type CV_32F. This matrix is both the input to the function and is modified in-place to replace any NaN values with the specified value.'
        },
        'output': {
            'number': '1',
            'description': 'The function operates in-place and replaces NaN values in the input matrix `a` with the specified value `val`. Since the operation is in-place, the modified matrix is the same as the input matrix `a`.'
        }
    }
    return info

def get_medianBlur_params_info():
    info = {
        'src': {
            'format': 'numpy.ndarray',
            'type': 'uint8',
            'description': 'Input image for median blurring. The image can be 1-, 3-, or 4-channel. For ksize of 3 or 5, the image depth should be CV_8U, CV_16U, or CV_32F. For larger aperture sizes, the image depth can only be CV_8U.'
        },
        'ksize': {
            'type': 'int',
            'description': 'Aperture linear size; it must be an odd number greater than 1, such as 3, 5, 7, etc. This size determines the size of the neighborhood used to compute the median which affects the amount of blurring.'
        },
        'output': {
            'number': '1',
            'description': 'The function returns the blurred image. Median blurring is applied using the specified ksize, and each channel of the image is processed independently. If the dst parameter is provided, it is filled with the result; otherwise, a new array is created and returned.'
        }
    }
    return info

def get_linearPolar_params_info():
    info = {
        'src': {
            'format': 'numpy.ndarray',
            'type': 'int8, float32, or float64',
            'description': 'Source image to be transformed to polar coordinates. This is the input image for which the polar transformation is applied.'
        },
        'center': {
            'format': 'tuple',
            'type': 'uint8',
            'description': 'The center of the transformation in the source image; where the output precision is maximal. It is a point around which the transformation is applied.'
        },
        'maxRadius': {
            'type': 'float',
            'description': 'The maximum radius in the original image to consider for the transformation. It determines the inverse magnitude scale parameter as well, affecting the spread of the polar coordinates.'
        },
        'flags': {
            'type': 'int',
            'default': 'cv2.INTER_LINEAR',
            'description': '(cv2.INTER_LINEAR or cv2.INTER_NEAREST) and optional flags (cv2.WARP_FILL_OUTLIERS and/or cv2.WARP_INVERSE_MAP).'
        },
        'output': {
            'number': '1',
            'description': 'The function returns the destination image containing the transformed source image in polar coordinates. If the dst parameter is provided, it is filled with the result; otherwise, a new array is created and returned.'
        }
    }
    return info

def get_illuminationChange_params_info():
    info = {
        'src': {
            'format': 'numpy.ndarray',
            'type': 'uint8',
            'description': 'Input image. It must be an 8-bit 3-channel image. This is the image whose illumination is to be changed.'
        },
        'mask': {
            'format': 'numpy.ndarray',
            'type': 'uint8',
            'description': 'Mask image. It can be an 8-bit 1 or 3-channel image. The areas of the source image where the illumination change will be applied are specified by the non-zero regions of the mask.'
        },
        'output': {
            'number': '1',
            'description': 'The function returns the destination image containing the result of the applied illumination change. If the dst parameter is provided, it is filled with the result; otherwise, a new array is created and returned.'
        }
    }
    return info

def get_compare_params_info():
    info = {
        'src1': {
            'format': 'numpy.ndarray',
            'type': 'int8, float32, or float64',
            'description': 'it must have a single channel.'
        },
        'src2': {
            'format': 'numpy.ndarray',
            'type': 'int8, float32, or float64 as same as src1',
            'description': 'it must also have a single channel'
        },
        'cmpop': {
            'type': 'int',
            'description': 'A flag that specifies the type of comparison operation to be performed (cv::CmpTypes). The available comparison operations are cv2.CMP_EQ, cv2.CMP_GT, cv2.CMP_GE, cv2.CMP_LT, cv2.CMP_LE, and cv2.CMP_NE.'
        },
        'output': {
            'number': '1',
            'description': 'The function returns the destination array containing the result of the comparison. Each element in the dst array corresponds to the result of comparison operation applied to the elements of src1 and src2 at the same location. If the dst parameter is provided, it is filled with the result; otherwise, a new array is created and returned.'
        }
    }
    return info

def get_setLogLevel_params_info():
    info = {
        'level': {
            'type': 'int',
            'description': 'Specifies the logging level for OpenCV operations. Depending on the implementation, this could be an integer value or a string representing the desired level of logging detail. Common levels include Error, Warning, Info, and Debug, with Error showing the least detail and Debug the most.'
        },
        'output': {
            'number': '1',
            'description': 'The function typically does not return a value. It sets the global logging level for OpenCV operations, affecting the verbosity of the messages that are output during execution.'
        }
    }
    return info

def get_seamlessClone_params_info():
    info = {
        'src': {
            'format': 'numpy.ndarray',
            'type': 'uint8',
            'description': 'Input 8-bit 3-channel source image from which a region is cloned onto the destination image.'
        },
        'dst': {
            'format': 'numpy.ndarray',
            'type': 'uint8',
            'description': 'Input 8-bit 3-channel destination image onto which the source image region is cloned.'
        },
        'mask': {
            'format': 'numpy.ndarray',
            'type': 'uint8',
            'description': 'Input 8-bit 3-channel mask image. The mask defines the region of the source image to be cloned onto the destination image. The non-zero pixels in the mask correspond to the region of the source image that will be cloned.'
        },
        'p': {
            'format': 'tuple',
            'type': 'int8',
            'description': 'Point in the destination image where the center of the source image region (defined by the mask) is placed. This point is specified as a tuple (x, y).'
        },
        'flags': {
            'type': 'int',
            'default': 'cv2.NORMAL_CLONE',
            'description': 'Cloning method that could be cv2.NORMAL_CLONE, cv2.MIXED_CLONE, or cv2.MONOCHROME_TRANSFER. This parameter specifies the algorithm used for seamless cloning.'
        },
        'output': {
            'number': '1',
            'description': 'The function returns the output image containing the seamlessly cloned region. If the blend parameter is provided, it is filled with the result; otherwise, a new array is created and returned.'
        }
    }
    return info

def get_sampsonDistance_params_info():
    info = {
        'pt1': {
            'format': 'numpy.ndarray',
            'type': 'float64',
            'description': 'First homogeneous 2D point, typically represented as a 3-element vector [x, y, 1] in homogeneous coordinates.'
        },
        'pt2': {
            'format': 'numpy.ndarray',
            'type': 'float64 as same as pt1',
            'description': 'Second homogeneous 2D point. 3-element vector [x, y, 1] as a 3-element vector [x, y, 1].'
        },
        'F': {
            'format': 'numpy.ndarray',
            'type': 'float64',
            'description': 'The fundamental 3x3 matrix relating are drawn. The fundamental matrix can be computed using functions like cv2.findFundamentalMat.'
        },
        'output': {
            'number': '1',
            'description': 'The function returns the computed Sampson distance between the two points given the fundamental matrix. This distance is a measure of how well the points conform to the epipolar constraint imposed by the fundamental matrix, providing a scalar value indicative of the geometric error.'
        }
    }
    return info

def get_rotatedRectangleIntersection_params_info():
    info = {
        'rect1': {
            'format': 'tuple',
            'type': 'int8',
            'description': 'First rotated rectangle. This is specified as a RotatedRect object, which includes the center point, size, and the angle of rotation.'
        },
        'rect2': {
            'format': 'tuple',
            'type': 'int8',
            'description': 'Second rotated rectangle, also specified as a RotatedRect object. Similar to the first rectangle, it includes the center, size, and angle of rotation.'
        },
        'output': {
            'number': '2',
            'description': 'The function returns a tuple containing an integer flag and the intersecting region. The integer flag indicates one of the RectanglesIntersectTypes, which can denote no intersection, partial intersection, or full enclosure. The intersectingRegion part of the tuple contains the vertices of the intersecting region if such an intersection exists.'
        }
    }
    return info

def get_rotate_params_info():
    info = {
        'src': {
            'format': 'numpy.ndarray',
            'type': 'int8, float32, or float64',
            'description': 'Input array to be rotated. This can be an image or any 2D array.'
        },
        'rotateCode': {
            'type': 'int',
            'default': 'cv2.ROTATE_90_CLOCKWISE',
            'description': 'An enum to specify how to rotate the array. cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_180, cv2.ROTATE_90_COUNTERCLOCKWISE'
        },
        'output': {
            'number': '1',
            'description': 'The function returns the destination array containing the rotated image or 2D array. If the dst parameter is provided, it is filled with the result; otherwise, a new array is created and returned.'
        }
    }
    return info

def get_randShuffle_params_info():
    info = {
        'dst': {
            'format': 'numpy.ndarray',
            'type': 'int8, float32, or float64',
            'description': 'The input/output numerical 1D array that is to be shuffled. The array is modified in-place, meaning that the shuffling effect is applied directly to this array.'
        },
        'output': {
            'number': '1',
            'description': 'The function operates in-place and returns the shuffled array. Since the operation modifies the input array directly, the returned array is the same object as the input with its elements shuffled.'
        }
    }
    return info

def get_pow_params_info():
    info = {
        'src': {
            'format': 'numpy.ndarray',
            'type': 'float32',
            'description': 'Input array. Each element of this array will be raised to the specified power. The array can be of any numerical type.'
        },
        'power': {
            'type': 'float32',
            'description': 'The exponent to which each element of the input array is raised. If power is an integer, each element is raised to this power directly. For non-integer powers, the absolute value of each element is raised to the power, with additional steps needed to handle negative values correctly.'
        },
        'output': {
            'number': '1',
            'description': 'The function returns the destination array containing the elements of the input array raised to the specified power. If the dst parameter is provided, it is filled with the result; otherwise, a new array is created and returned.'
        }
    }
    return info

def get_perspectiveTransform_params_info():
    info = {
        'ptx': {
            'format': 'numpy.ndarray',
            'type': 'float32',
            'description': '1xNx2 array Input two-channel or three-channeion. The array should be of a floating-point type.'
        },
        'm': {
            'format': 'numpy.ndarray',
            'type': 'float32',
            'description': '3x3 matrix for transforming 3D vectors. The matrix should be of a floating-point type.'
        },
        'output': {
            'number': '1',
            'description': 'The function returns the destination array containing the transformed vectors. If the dst parameter is provided, it is filled with the result; otherwise, a new array is created and returned.'
        }
    }
    return info

def get_pencilSketch_params_info():
    info = {
        'src': {
            'format': 'numpy.ndarray',
            'type': 'uint8',
            'description': 'Input image. It must be an 8-bit 3-channel image. This is the image that will be processed to generate a pencil sketch effect.'
        },
        'output': {
            'number': '1',
            'description': 'The function returns two images. The first output (dst1) is a grayscale image that resembles a pencil sketch. The second output (dst2) contains a color version of the pencil sketch. These images provide a non-photorealistic rendering of the input image akin to a hand-drawn pencil sketch.'
        }
    }
    return info

def get_watershed_params_info():
    info = {
        'image': {
            'format': 'numpy.ndarray',
            'type': 'uint8',
            'description': 'Input image for segmentation. It must be an 8-bit 3-channel image. The function applies a watershed algorithm to this image based on the provided markers.'
        },
        'makers': {
            'format': 'numpy.ndarray',
            'type': 'int32',
            'description': 'Input image for segmentation. It must be an 8-bit single-channel image. The function applies a watershed algorithm to this image based on the provided markers.'
        },
        'output': {
            'number': '1',
            'description': 'The function operates in-place and modifies the markers array to represent the segmentation result. Each pixel in the markers array is set to a value corresponding to the region it belongs to, with -1 indicating boundaries between regions. There is no separate return value; the markers array itself is updated.'
        }
    }
    return info

def get_warpPolar_params_info():
    info = {
        'src': {
            'format': 'numpy.ndarray',
            'type': 'int8, float32, or float64',
            'description': 'Source image to be transformed. It can be of any type and number of channels.'
        },
        'dsize': {
            'format': 'tuple',
            'type': 'int8',
            'default': '(200,200)',
            'flag': 'None',
            'description': 'The destination image size in the format (width, height). Depending on the values, the area and dimensions of the destination image will adapt accordingly.'
        },
        'center': {
            'format': 'tuple',
            'type': 'int8',
            'description': 'The transformation center point in the source image. It determines the origin of the polar coordinate system.'
        },
        'maxRadius': {
            'type': 'float',
            'default': '20.0',
            'flag': 'None',
            'description': 'The radius of the bounding circle of the source image to transform. This parameter influences the scale of the magnitude in the polar transformation.'
        },
        'flags': {
            'type': 'int',
            'default': 'cv2.INTER_LINEAR',
            'description': 'A combination of interpolation and operation flags. This includes interpolation methods (e.g., cv2.INTER_LINEAR), WarpPolarMode (e.g., cv2.WARP_POLAR_LINEAR for linear polar mapping, cv2.WARP_POLAR_LOG for semilog mapping), and cv2.WARP_INVERSE_MAP for reverse mapping.'
        },
        'output': {
            'number': '1',
            'description': 'The function returns the destination image containing the transformed result. If the dst parameter is provided, it is filled with the result; otherwise, a new array is created and returned.'
        }
    }
    return info

def get_threshold_params_info():
    info = {
        'src': {
            'format': 'numpy.ndarray',
            'type': 'uint8 or float32',
            'description': 'Input array (8-bit single-channel or 32-bit floating point). This is the source image that will be thresholded.'
        },
        'thresh': {
            'type': 'float',
            'default': '127',
            'flag': 'None',
            'description': 'Threshold value. Every pixel value higher than this threshold is set according to the type parameter, and every pixel value less than or equal to the threshold is set to another value (or remains unchanged for some types).'
        },
        'maxval': {
            'type': 'float',
            'default': '255',
            'flag': 'None',
            'description': 'Maximum value to use with the THRESH_BINARY and THRESH_BINARY_INV thresholding types. This is the value that pixels exceeding the threshold will be set to, depending on the specified thresholding type.'
        },
        'type_thresh': {
            'type': 'int',
            'default': 'cv2.THRESH_BINARY',
            'description': 'Thresholding type. This parameter defines the action taken for pixel values relative to the threshold. The different types are defined in cv2.ThresholdTypes, including binary thresholding, inverse binary thresholding, and others. THRESH_OTSU or THRESH_TRIANGLE may also be specified here to use Otsu’s or Triangle method for automatic threshold determination.'
        },
        'output': {
            'number': '1',
            'description': 'The function returns a tuple (retval, dst). retval is the computed threshold value if Otsu’s or Triangle methods are used. dst is the output image that has been thresholded.'
        }
    }
    return info

def get_stylization_params_info():
    info = {
        'src': {
            'format': 'numpy.ndarray',
            'type': 'uint8',
            'description': 'Input image. It must be an 8-bit 3-channel image. This is the image that will be processed to achieve a stylized effect.'
        },
        'output': {
            'number': '1',
            'description': 'The function returns the destination array containing the stylized image. If the dst parameter is provided, it is filled with the result; otherwise, a new array is created and returned.'
        }
    }
    return info

def get_solvePoly_params_info():
    info = {
        'coeffs': {
            'format': 'numpy.ndarray',
            'type': 'float32 or float64',
            'description': '1xN/Nx1 2-channel Array of polynomial coefficients. The coefficients are in decreasing powers, where coeffs[0] is the coefficient of the highest power and coeffs[n] is the constant term.'
        },
        'output': {
            'number': '2',
            'description': 'The function returns a tuple (retval, roots). retval is the number of roots found (it can be lesser than the degree of the polynomial if some roots are missed), and roots is the array containing the roots of the polynomial.'
        }
    }
    return info

def get_solvePnPRefineVVS_params_info():
    info = {
        'objectPoints': {
            'format': 'numpy.ndarray',
            'type': 'float32',
            'description': '1xN/Nx1 3-channel'
        },
        'imagePoints': {
            'format': 'numpy.ndarray',
            'type': 'float32 as same as ',
            'description': '1xN/Nx1 2-channel'
        },
        'cameraMatrix': {
            'format': 'numpy.ndarray',
            'type': 'float32',
            'description': 'Input camera intrinsic matrix. This matrix includes the focal lengths and the optical centers and is used to project 3D points to the image plane.'
        },
        'distCoeffs': {
            'format': 'numpy.ndarray',
            'type': 'float32',
            'flag': 'None',
            'description': 'Input of vector distortion coefficients. If empty, zero coefficients are assumed.',
        },
        'rvec': {
            'format': 'numpy.ndarray',
            'type': 'float32',
            'description': '3x1'
        },
        'tvec': {
            'format': 'numpy.ndarray',
            'type': 'float32',
            'description': '3x1'
        },
        'criteria': {
            'format': 'tuple',
            'type': 'uint8',
            'default': '(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6)',
            'flag': 'None',
            'description': 'Input/Output translation vector. Initial values are used as a starting point for the refinement.'
        },
        'output': {
            'number': '2',
            'description': 'The function returns refined rotation and translation vectors. These vectors minimize the projection error, effectively refining the pose of the object with respect to the camera.'
        }
    }
    return info

def get_solvePnPRefineLM_params_info():
    info = {
        'objectPoints': {
            'format': 'numpy.ndarray',
            'type': 'float32',
            'description': '1xN/Nx1 3-channel'
        },
        'imagePoints': {
            'format': 'numpy.ndarray',
            'type': 'float32 as same as ',
            'description': '1xN/Nx1 2-channel'
        },
        'cameraMatrix': {
            'format': 'numpy.ndarray',
            'type': 'float32',
            'description': 'Input camera intrinsic matrix. This matrix includes the focal lengths and the optical centers and is used to project 3D points to the image plane.'
        },
        'distCoeffs': {
            'format': 'numpy.ndarray',
            'type': 'float32',
            'flag': 'None',
            'description': 'Input of vector distortion coefficients. If empty, zero coefficients are assumed.',
        },
        'rvec': {
            'format': 'numpy.ndarray',
            'type': 'float32',
            'description': '3x1'
        },
        'tvec': {
            'format': 'numpy.ndarray',
            'type': 'float32',
            'description': '3x1'
        },
        'criteria': {
            'format': 'tuple',
            'type': 'uint8',
            'default': '(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6)',
            'flag': 'None',
            'description': 'Input/Output translation vector. Initial values are used as a starting point for the refinement.'
        },
        'output': {
            'number': '2',
            'description': 'The function returns refined rotation and translation vectors. These vectors minimize the projection error, effectively refining the pose of the object with respect to the camera.'
        }
    }
    return info

def get_solvePnPRansac_params_info():
    info = {
        'objectPoints': {
            'format': 'numpy.ndarray',
            'type': 'float32',
            'description': '1xN/Nx1 3-channel'
        },
        'imagePoints': {
            'format': 'numpy.ndarray',
            'type': 'float32 as same as ',
            'description': '1xN/Nx1 2-channel'
        },
        'cameraMatrix': {
            'format': 'numpy.ndarray',
            'type': 'float32',
            'description': 'Input camera intrinsic matrix. This matrix includes the focal lengths and the optical centers and is used to project 3D points to the image plane.'
        },
        'distCoeffs': {
            'format': 'numpy.ndarray',
            'type': 'float32',
            'flag': 'None',
            'description': 'Input of vector distortion coefficients. If empty, zero coefficients are assumed.',
        },
        'output': {
            'number': '4',
            'description': 'The function returns refined rotation and translation vectors. These vectors minimize the projection error, effectively refining the pose of the object with respect to the camera.'
        }
    }
    return info

def get_solvePnPGeneric_params_info():
    info = {
        'objectPoints': {
            'format': 'numpy.ndarray',
            'type': 'float32',
            'description': '1xN/Nx1 3-channel'
        },
        'imagePoints': {
            'format': 'numpy.ndarray',
            'type': 'float32 as same as ',
            'description': '1xN/Nx1 2-channel'
        },
        'cameraMatrix': {
            'format': 'numpy.ndarray',
            'type': 'float32',
            'description': 'Input camera intrinsic matrix. This matrix includes the focal lengths and the optical centers and is used to project 3D points to the image plane.'
        },
        'distCoeffs': {
            'format': 'numpy.ndarray',
            'type': 'float32',
            'flag': 'None',
            'description': 'Input of vector distortion coefficients. If empty, zero coefficients are assumed.',
        },
        'output': {
            'number': '4',
            'description': 'The function returns refined rotation and translation vectors. These vectors minimize the projection error, effectively refining the pose of the object with respect to the camera.'
        }
    }
    return info

def get_solvePnP_params_info():
    info = {
        'objectPoints': {
            'format': 'numpy.ndarray',
            'type': 'float32',
            'description': '1xN/Nx1 3-channel'
        },
        'imagePoints': {
            'format': 'numpy.ndarray',
            'type': 'float32 as same as ',
            'description': '1xN/Nx1 2-channel'
        },
        'cameraMatrix': {
            'format': 'numpy.ndarray',
            'type': 'float32',
            'description': 'Input camera intrinsic matrix. This matrix includes the focal lengths and the optical centers and is used to project 3D points to the image plane.'
        },
        'distCoeffs': {
            'format': 'numpy.ndarray',
            'type': 'float32',
            'flag': 'None',
            'description': 'Input of vector distortion coefficients. If empty, zero coefficients are assumed.',
        },
        'output': {
            'number': '3',
            'description': 'The function returns refined rotation and translation vectors. These vectors minimize the projection error, effectively refining the pose of the object with respect to the camera.'
        }
    }
    return info

def get_boxFilter_params_info():#bug
    info = {
        'src': {
            'format': 'numpy.ndarray',
            'type': 'uint8 or float32',
            'description': 'Input image to be blurred.'
        },
        'ddepth': {
            'type': 'int',
            'default': '-1',
            'flag': 'None',
            'description': 'The desired depth of the output image. Use -1 to use the same depth as the source image.'
        },
        'ksize': {
            'format': 'tuple',
            'type': 'int',
            'description': 'Size of the blurring kernel. Specified as a tuple (width, height).'
        },
        'output': {
            'number': '1',
            'description': 'The function returns the blurred output image.'
        }
    }
    return info

def get_broadcast_params_info():
    info = {
        'src': {
            'format': 'numpy.ndarray',
            'type': 'int32',
            'description': 'Input array to be broadcasted.'
        },
        'shape': {
            'format': 'numpy.ndarray',
            'type': 'int32',
            'description': '2x1 shape'
        },
        'output': {
            'number': '1',
            'description': 'The function returns the broadcasted output array, reshaped to the specified target shape.'
        }
    }
    return info

def get_calibrateCameraRO_params_info():
    info = {
        'objectPoints': {
            'format': 'numpy.ndarray',
            'type': 'float32',
            'description': '1xN/Nx1 3-channel'
        },
        'imagePoints': {
            'format': 'numpy.ndarray',
            'type': 'float32 as same as ',
            'description': '1xN/Nx1 2-channel'
        },
        'imageSize': {
            'format': 'tuple',
            'type': 'int',
            'description': 'Size of the image used only to initialize the intrinsic camera matrix.'
        },
        'iFixedPoint': {
            'type': 'int',
            'default': '-1',
            'flag': 'None',
            'description': 'Specifies which one of the object points is fixed. If you are using a symmetric pattern like a checkerboard, the function uses some of the detected feature points as fixed points. This parameter is the index of the fixed point in the objectPoints array.'
        },
        'cameraMatrix': {
            'format': 'numpy.ndarray',
            'type': 'float32',
            'description': 'Input camera intrinsic matrix. This matrix includes the focal lengths and the optical centers and is used to project 3D points to the image plane.'
        },
        'distCoeffs': {
            'format': 'numpy.ndarray',
            'type': 'float32',
            'flag': 'None',
            'description': 'Input of vector distortion coefficients. If empty, zero coefficients are assumed.',
        },
        'rvecs': {
            'format': 'None',
            'type': 'float32',
            'flag': 'None',
            'description': 'Input camera intrinsic matrix. This matrix includes the focal lengths and the optical centers and is used to project 3D points to the image plane.'
        },
        'tvecs': {
            'format': 'None',
            'type': 'float32',
            'flag': 'None',
            'description': 'Input of vector distortion coefficients. If empty, zero coefficients are assumed.',
        },

        'output': {
            'number': '6',
            'description': 'The function returns the reprojection error, the refined camera intrinsic matrix, distortion coefficients, rotation and translation vectors, and optionally the updated object points.'
        }
    }
    return info

def get_calibrateCameraROExtended_params_info():
    info = {
        'objectPoints': {
            'format': 'numpy.ndarray',
            'type': 'float32',
            'description': '1xN/Nx1 3-channel'
        },
        'imagePoints': {
            'format': 'numpy.ndarray',
            'type': 'float32 as same as ',
            'description': '1xN/Nx1 2-channel'
        },
        'imageSize': {
            'format': 'tuple',
            'type': 'int',
            'description': 'Size of the image used only to initialize the intrinsic camera matrix.'
        },
        'iFixedPoint': {
            'type': 'int',
            'default': '-1',
            'flag': 'None',
            'description': 'Specifies which one of the object points is fixed. If you are using a symmetric pattern like a checkerboard, the function uses some of the detected feature points as fixed points. This parameter is the index of the fixed point in the objectPoints array.'
        },
        'cameraMatrix': {
            'format': 'numpy.ndarray',
            'type': 'float32',
            'description': 'Input camera intrinsic matrix. This matrix includes the focal lengths and the optical centers and is used to project 3D points to the image plane.'
        },
        'distCoeffs': {
            'format': 'numpy.ndarray',
            'type': 'float32',
            'flag': 'None',
            'description': 'Input of vector distortion coefficients. If empty, zero coefficients are assumed.',
        },
        'rvecs': {
            'format': 'None',
            'type': 'float32',
            'flag': 'None',
            'description': 'Input camera intrinsic matrix. This matrix includes the focal lengths and the optical centers and is used to project 3D points to the image plane.'
        },
        'tvecs': {
            'format': 'None',
            'type': 'float32',
            'flag': 'None',
            'description': 'Input of vector distortion coefficients. If empty, zero coefficients are assumed.',
        },
        'flags': {
            'type': 'int',
            'default': 'cv2.CALIB_USE_INTRINSIC_GUESS',
            'flag': 'None',
            'description': 'Different operation flags that can be zero or a combination of the following values: cv2.CALIB_USE_INTRINSIC_GUESS, cv2.CALIB_FIX_ASPECT_RATIO, cv2.CALIB_FIX_PRINCIPAL_POINT, etc.'
        },
        'criteria': {
            'format': 'tuple',
            'type': 'int',
            'default': '(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6)',
            'flag': 'None',
            'description': 'Termination criteria for the iterative optimization algorithm.'
        },
        'output': {
            'number': '10',
            'description': 'The function returns the reprojection error, the refined camera intrinsic matrix, distortion coefficients, rotation and translation vectors, and optionally the updated object points.'
        }
    }
    return info

def get_calibrateHandEye_params_info():
    info = {
        'R_gripper2base': {
            'format': 'numpy.ndarray',
            'type': 'float32',
            'description': 'Array containing rotation matrices (3x3) or rotation vectors (3x1) for the transformations from the gripper frame to the robot base frame.'
        },
        't_gripper2base': {
            'format': 'numpy.ndarray',
            'type': 'float32',
            'description': '3x1) for the transformations from the gripper frame to the robot base frame.'
        },
        'R_target2cam': {
            'format': 'numpy.ndarray',
            'type': 'float32',
            'description': 'Array containing rotation matrices (3x3) or rotation vectors (3x1) for the transformations from the calibration target frame to the camera frame.'
        },
        't_target2cam': {
            'format': 'numpy.ndarray',
            'type': 'float32',
            'description': '3x1) for the transformations from the calibration target frame to the camera frame.'
        },
        'method': {
            'type': 'int',
            'default' : 'cv2.CALIB_HAND_EYE_TSAI',
            'flag': 'None',
            'description': 'cv2.CALIB_HAND_EYE_TSAI is a common choice, but other methods like cv2.CALIB_HAND_EYE_PARK, cv2.CALIB_HAND_EYE_HORAUD, or cv2.CALIB_HAND_EYE_ANDREFF might be more suitable depending on your specific scenario.'
        },
        'output': {
            'number': '2',
            'description': 'The function returns two main outputs: the rotation matrix/vector and the translation vector. These define the estimated transformation from the camera frame to the gripper frame.'
        }

    }
    return info

def get_calibrateRobotWorldHandEye_params_info():
    info = {
        'R_world2cam': {
            'format': 'numpy.ndarray',
            'type': 'float32',
            'description': 'Array containing rotation matrices (3x3) or rotation vectors (3x1) for the transformations from the world frame to the camera frame.'
        },
        't_world2cam': {
            'format': 'numpy.ndarray',
            'type': 'float32',
            'description': 'Array containing translation vectors (3x1) for the transformations from the world frame to the camera frame.'
        },
        'R_base2gripper': {
            'format': 'numpy.ndarray',
            'type': 'float32',
            'description': 'Array containing rotation matrices (3x3) or rotation vectors (3x1) for the transformations from the robot base frame to the gripper frame.'
        },
        't_base2gripper': {
            'format': 'numpy.ndarray',
            'type': 'float32',
            'description': 'Array containing translation vectors (3x1) for the transformations from the robot base frame to the gripper frame.'
        },
        'method': {
            'type': 'int',
            'default': 'cv2.CALIB_ROBOT_WORLD_HAND_EYE_SHAH',
            'flag': 'None',
            'description': 'Method for solving the Robot-World/Hand-Eye calibration problem. cv2.CALIB_ROBOT_WORLD_HAND_EYE_SHAH is a common choice, but other methods like cv2.CALIB_ROBOT_WORLD_HAND_EYE_LI may be more suitable depending on the scenario.'
        },
        'output': {
            'number': '4',
            'description': 'The function returns four main outputs: the rotation matrix and the translation vector that define the transformation from the robot base frame to the world frame, and the rotation matrix and the translation vector that define the transformation from the gripper frame to the camera frame.'
        }
    }
    return info

def get_calibrationMatrixValues_params_info():
    info = {
        'cameraMatrix': {
            'format': 'numpy.ndarray',
            'type': 'float32',
            'description': 'Input camera intrinsic matrix. This matrix includes the focal lengths and the optical centers. It can be estimated by calibrateCamera or stereoCalibrate.'
        },
        'imageSize': {
            'format': 'tuple',
            'type': 'int',
            'description': 'Input image size in pixels. It is used to initialize intrinsic camera matrix.'
        },
        'apertureWidth': {
            'type': 'float',
            'description': 'Physical width in mm of the sensor. It affects the field of view calculations.'
        },
        'apertureHeight': {
            'type': 'float',
            'description': 'Physical height in mm of the sensor. It affects the field of view calculations.'
        },
        'output': {
            'number': '5',
            'description': 'The function returns the field of view in degrees along both the horizontal and vertical sensor axis, the focal length of the lens in mm, the principal point in mm, and the aspect ratio \f$f_y/f_x\f$.'
        }
    }
    return info

def get_cartToPolar_params_info():
    info = {
        'x': {
            'format': 'numpy.ndarray',
            'type': 'float32 or float64',
            'description': 'Input array of x-coordinates. Must be a single or double-precision floating-point array.'
        },
        'y': {
            'format': 'numpy.ndarray',
            'type': 'float32 or float64 as same as x',
            'description': 'Input array of y-coordinates. Must have the same size and type as the x array.'
        },
        'magnitude': {
            'format': 'numpy.ndarray',
            'type': 'float32 or float64 as same as x',
            'description': 'Output array of magnitudes. It will have the same size and type as the input arrays.'
        },
        'angle': {
            'format': 'numpy.ndarray',
            'type': 'float32 or float64 as same as x',
            'description': 'Output array of angles. It will have the same size and type as the input arrays. The angles are measured in radians by default or in degrees if angleInDegrees is True.'
        },
        'angleInDegrees': {
            'type': 'bool',
            'default': 'False',
            'description': 'Flag indicating whether the angles are measured in radians (False) or degrees (True).'
        },
        'output': {
            'number': '2',
            'description': 'The function returns two arrays: magnitude and angle of the 2D vectors constructed from the input x and y arrays.'
        }
    }
    return info

def get_checkChessboard_params_info():
    info = {
        'img': {
            'format': 'numpy.ndarray',
            'type': 'uint8',
            'description': '8-bit single-channel'
        },
        'size': {
            'format': 'tuple',
            'type': 'int',
            'description': 'Size of the chessboard pattern, indicating the number of inner corners per a chessboard row and column (width, height).'
        },
        'output': {
            'number': '1',
            'description': 'The function returns a boolean value indicating whether a chessboard pattern could be detected in the input image or not.'
        }
    }
    return info

def get_checkHardwareSupport_params_info():
    info = {
        'feature': {
            'type': 'int',
            'default': '3',
            'description': 'The feature of interest to check for hardware support. This parameter should be one of the cv::CpuFeatures values, such as cv::CPU_MMX, cv::CPU_SSE, cv::CPU_SSE2, cv::CPU_SSE3, cv::CPU_SSSE3, cv::CPU_SSE4_1, cv::CPU_SSE4_2, cv::CPU_POPCNT, cv::CPU_FP16, cv::CPU_AVX, cv::CPU_AVX2, cv::CPU_FMA3, cv::CPU_AVX_512F, cv::CPU_AVX_512BW, cv::CPU_AVX_512CD, cv::CPU_AVX_512DQ, cv::CPU_AVX_512ER, cv::CPU_AVX_512IFMA, cv::CPU_AVX_512PF, cv::CPU_AVX_512VBMI, cv::CPU_AVX_512VL, cv::CPU_NEON.'
        },
        'output': {
            'number': '1',
            'description': 'The function returns a boolean value indicating whether the specified hardware feature is supported by the host.'
        }
    }
    return info

def get_checkRange_params_info():
    info = {
        'a': {
            'format': 'numpy.ndarray',
            'type': 'float32 or float64',
            'description': 'Input array to be checked.'
        },
        'quiet': {
            'type': 'bool',
            'default': 'False',
            'description': 'If true, the function returns false when the array elements are out of range without throwing an exception. If false, an exception is thrown.'
        },
        'minVal': {
            'type': 'float',
            'default': '-255',
            'flag': 'None',
            'description': 'Inclusive lower boundary of valid values range. Python supports -float("inf") for negative infinity.'
        },
        'maxVal': {
            'type': 'float',
            'default': '255',
            'flag': 'None',
            'description': 'Exclusive upper boundary of valid values range. Python supports float("inf") for positive infinity.'
        },
        'output': {
            'number': '2',
            'description': 'The function returns a boolean value indicating if all array elements are within the specified range. The position of the first outlier is also returned if found and if pos parameter is not NULL.'
        }
    }
    return info

def get_clipLine_params_info():
    info = {
        'imgRect': {
            'format': 'tuple',
            'type': 'int',
            'description': 'The rectangle representing the image boundaries. It should be specified as a tuple (x, y, width, height), where (x, y) is the top-left corner, and width and height are the dimensions of the rectangle.'
        },
        'pt1': {
            'format': 'tuple',
            'type': 'uint8',
            'description': 'The first point of the line segment. Specified as a tuple (x1, y1).'
        },
        'pt2': {
            'format': 'tuple',
            'type': 'uint8',
            'description': 'The second point of the line segment. Specified as a tuple (x2, y2).'
        },
        'output': {
            'number': '3',
            'description': 'Returns a boolean value indicating whether the line segment is inside the image rectangle. If true, it also returns the clipped line points within the image rectangle.'
        }
    }
    return info


def get_colorChange_params_info():
    info = {
        'src': {
            'format': 'numpy.ndarray',
            'type': 'uint8',
            'description': 'Input image. Must be an 8-bit 3-channel image.'
        },
        'mask': {
            'format': 'numpy.ndarray',
            'type': 'uint8 as same as src',
            'description': 'Mask image. Can be an 8-bit 1-channel or 3-channel image. The mask specifies the area which will be affected by the color change.'
        },
        'red_mul': {
            'type': 'float',
            'default': '1.0',
            'flag': 'None',
            'description': 'Multiplication factor for the red channel. Adjusts the red component of the color.'
        },
        'green_mul': {
            'type': 'float',
            'default': '1.0',
            'flag': 'None',
            'description': 'Multiplication factor for the green channel. Adjusts the green component of the color.'
        },
        'blue_mul': {
            'type': 'float',
            'default': '1.0',
            'flag': 'None',
            'description': 'Multiplication factor for the blue channel. Adjusts the blue component of the color.'
        },
        'output': {
            'number': '1',
            'description': 'The function returns an output image where the specified color changes have been applied to the area of the input image defined by the mask.'
        }
    }
    return info

def get_compareHist_params_info():
    info = {
        'H1': {
            'format': 'numpy.ndarray',
            'type': 'float32',
            'description': 'First histogram to compare. Must be of the same size and type as H2.'
        },
        'H2': {
            'format': 'numpy.ndarray',
            'type': 'float32 as same as H1',
            'description': 'Second histogram to compare. Must be of the same size and type as H1.'
        },
        'HistCompMethods': {
            'type': 'int',
            'defualt': 'cv2.HISTCMP_CORREL',
            'description': 'Comparison method to be used. Must be one of the predefined HistCompMethods.'
        },
        'output': {
            'number': '1',
            'description': 'The function returns a distance or similarity measure between the two histograms, depending on the comparison method.'
        }
    }
    return info

def get_composeRT_params_info():
    info = {
        'rvec1': {
            'format': 'numpy.ndarray',
            'type': 'float32 or float64',
            'description': '3x1 The first rotation vector.'
        },
        'tvec1': {
            'format': 'numpy.ndarray',
            'type': 'float32 or float64 as same as',
            'description': '3x1 The first translation vector.'
        },
        'rvec2': {
            'format': 'numpy.ndarray',
            'type': 'float32 or float64 as same as',
            'description': '3x1 The second rotation vector.'
        },
        'tvec2': {
            'format': 'numpy.ndarray',
            'type': 'float32 or float64 as same as',
            'description': '3x1 The second translation vector.'
        },
        'output': {
            'number': '2',
            'description': 'The function returns the output rotation vector, output translation vector, and optionally, the derivatives of these outputs with respect to the input rotation and translation vectors.'
        }
    }
    return info

def get_computeCorrespondEpilines_params_info():
    info = {
        'points': {
            'format': 'numpy.ndarray',
            'type': 'float32',
            'description': 'Input points. Matrix of type CV_32FC2 or a vector of Point2f. Represents either points in the first image or points in the second image.'
        },
        'whichImage': {
            'type': 'int',
            'default': '1',
            'flag': 'None',
            'description': 'Index of the image (1 or 2) that contains the points. This parameter specifies whether the input points are from the first or the second image of the stereo pair.'
        },
        'F': {
            'format': 'numpy.ndarray',
            'type': 'float32 or float64',
            'description': 'Fundamental matrix that can be estimated using findFundamentalMat or stereoRectify.'
        },
        'output': {
            'number': '1',
            'description': 'The function computes and returns the epipolar lines corresponding to the points in the specified image.'
        }
    }
    return info

def get_computeECC_params_info():
    info = {
        'templateImage': {
            'format': 'numpy.ndarray',
            'type': 'uint8',
            'description': '8-bit single-channel'
        },
        'inputImage': {
            'format': 'numpy.ndarray',
            'type': 'uint8',
            'description': '8-bit single-channel'
        },
        'output': {
            'number': '1',
            'description': 'The function computes and returns the Enhanced Correlation Coefficient (ECC) value which indicates the similarity between the template image and the input image.'
        }
    }
    return info

def get_connectedComponents_params_info():
    info = {
        'image': {
            'format': 'numpy.ndarray',
            'type': 'uint8',
            'description': 'The 8-bit single-channel image to be labeled.'
        },
        'connectivity': {
            'type': 'int',
            'default': '8',
            'flag':'None',
            'description': 'Connectivity to be used for labeling: 8 or 4 for 8-way or 4-way connectivity, respectively.'
        },
        'ltype': {
            'type': 'int',
            'default': 'cv2.CV_32S',
            'flag': 'None',
            'description': 'Output image label type. Currently, CV_32S and CV_16U are supported.'
        },
        'output': {
            'number': '2',
            'description': 'The function returns the number of connected components found and the labeled output image.'
        }
    }
    return info

def get_connectedComponentsWithAlgorithm_params_info():
    info = {
        'image': {
            'format': 'numpy.ndarray',
            'type': 'uint8',
            'description': 'The 8-bit single-channel image to be labeled.'
        },
        'connectivity': {
            'type': 'int',
            'default': '8',
            'flag': 'None',
            'description': 'Connectivity for labeling: 8 or 4 for 8-way or 4-way connectivity, respectively.'
        },
        'ltype': {
            'type': 'int',
            'default': 'cv2.CV_32S',
            'flag': 'None',
            'description': 'Output image label type. Supports CV_32S and CV_16U.'
        },
        'ccltype': {
            'type': 'int',
            'default': 'cv2.CCL_WU',
            'flag': 'None',
            'description': 'Algorithm type for connected components labeling. Supported algorithms: Bolelli (Spaghetti), Grana (BBDT), and Wu’s (SAUF).'
        },
        'output': {
            'number': '2',
            'description': 'Returns the number of connected components found and the labeled output image.'
        }
    }
    return info

def get_connectedComponentsWithStats_params_info():
    info = {
        'image': {
            'format': 'numpy.ndarray',
            'type': 'uint8',
            'description': 'The 8-bit single-channel image to be labeled.'
        },
        'connectivity': {
            'type': 'int',
            'default': '8',
            'flag': 'None',
            'description': 'Connectivity for labeling: 8 or 4 for 8-way or 4-way connectivity, respectively.'
        },
        'ltype': {
            'type': 'int',
            'default': 'cv2.CV_32S',
            'flag': 'None',
            'description': 'Output image label type. Supports CV_32S and CV_16U.'
        },
        'output': {
            'number': '4',
            'description': 'Returns the number of connected components found, the labeled output image, statistics for each label, and centroids for each label.'
        }
    }
    return info

def get_connectedComponentsWithStatsWithAlgorithm_params_info():
    info = {
        'image': {
            'format': 'numpy.ndarray',
            'type': 'uint8',
            'description': 'The 8-bit single-channel image to be labeled.'
        },
        'connectivity': {
            'type': 'int',
            'default': '8',
            'flag': 'None',
            'description': 'Connectivity for labeling: 8 or 4 for 8-way or 4-way connectivity, respectively.'
        },
        'ltype': {
            'type': 'int',
            'default': 'cv2.CV_32S',
            'flag': 'None',
            'description': 'Output image label type. Supports CV_32S and CV_16U.'
        },
        'ccltype': {
            'type': 'int',
            'default': 'cv2.CCL_DEFAULT',
            'flag': 'None',
            'description': 'Connected components algorithm type. Options include cv2.CCL_WU, cv2.CCL_DEFAULT, cv2.CCL_BOLELLI, and cv2.CCL_GRANA for different algorithm choices.'
        },
        'output': {
            'number': '4',
            'description': 'Returns the number of connected components found, the labeled output image, statistics for each label, and centroids for each label.'
        }
    }
    return info

def get_contourArea_params_info():
    info = {
        'contour': {
            'format': 'numpy.ndarray',
            'type': 'float32',
            'description': 'Input vector of 2D points (contour vertices), which can be stored in a numpy.ndarray. The contour points should be passed in a shape of (n, 1, 2) for n points, or a simpler (n, 2) shape.'
        },
        'oriented': {
            'type': 'bool',
            'default': 'False',
            'description': 'Oriented area flag. If True, the function returns a signed area value, depending on the contour orientation (clockwise or counter-clockwise). This can be used to determine the orientation of a contour by observing the sign of the returned area value. The default value is False, meaning the absolute value of the area is returned.'
        },
        'output': {
            'number': '1',
            'description': 'The function returns the calculated area of the contour. The area is computed using the Green formula, and thus may differ from the number of non-zero pixels for drawn contours. The function may produce incorrect results for contours with self-intersections.'
        }
    }
    return info

def get_convertFp16_params_info():
    info = {
        'src': {
            'format': 'numpy.ndarray',
            'type': 'float32',
            'description': 'Input array. For forward conversion, the type should be float32 (CV_32F), representing single precision floating points. For backward conversion, the type should be int16 (CV_16S), representing half precision floating points stored in a 16-bit format.'
        },
        'output': {
            'number': '1',
            'description': 'The function returns the converted array. For forward conversion (CV_32F -> CV_16S), the output array will have half precision floating points represented in a 16-bit format (CV_16S). For backward conversion (CV_16S -> CV_32F), the output array will have single precision floating points (CV_32F).'
        }
    }
    return info

def get_convertMaps_params_info():
    info = {
        'map1': {
            'format': 'numpy.ndarray',
            'type': 'float32',
            'description': 'The first input map. It can be of type CV_16SC2, CV_32FC1, or CV_32FC2.'
        },
        'map2': {
            'format': 'numpy.ndarray',
            'type': 'float32',
            'description': 'The second input map. It is of type CV_16UC1 for CV_16SC2 map1 type, CV_32FC1 for CV_32FC1 map1 type, or none (empty matrix) for CV_32FC2 map1 type.'
        },
        'dstmap1type': {
            'type': 'int',
            'default': 'cv2.CV_16SC2',
            'flag': 'None',
            'description': 'Type of the first output map. It should be CV_16SC2, CV_32FC1, or CV_32FC2.'
        },
        'nninterpolation': {
            'type': 'bool',
            'default': 'False',
            'description': 'Flag indicating whether the fixed-point maps are used for the nearest-neighbor or for a more complex interpolation.'
        },
        'output': {
            'number': '2',
            'description': 'The function returns two output maps. The first output map (dstmap1) contains the rounded coordinates or the interpolation coefficients, depending on the specified output map type. The second output map (dstmap2), created only when nninterpolation=False and map1 type is CV_32FC1 or CV_32FC2, contains indices in the interpolation tables.'
        }
    }
    return info

def get_convertPointsFromHomogeneous_params_info():
    info = {
        'src': {
            'format': 'numpy.ndarray',
            'type': 'float32 or float64',
            'description': '1xN/Nx1 3-channel'
        },
        'output': {
            'number': '1',
            'description': 'The function returns an output vector of N-1-dimensional points in Euclidean space. Each point in the output vector is obtained by dividing the first N-1 coordinates of the input point by the last coordinate (perspective division).'
        }
    }
    return info

def get_convertPointsToHomogeneous_params_info():
    info = {
        'src': {
            'format': 'numpy.ndarray',
            'type': 'float32 or float64',
            'description': '1xN/Nx1 3-channel'
        },
        'output': {
            'number': '1',
            'description': 'The function returns an output vector of N-1-dimensional points in Euclidean space. Each point in the output vector is obtained by dividing the first N-1 coordinates of the input point by the last coordinate (perspective division).'
        }
    }
    return info

def get_convertScaleAbs_params_info():
    info = {
        'src': {
            'format': 'numpy.ndarray',
            'type': 'float32 or float64',
            'description': 'Input array that can be of any depth.'
        },
        'alpha': {
            'type': 'float',
            'default': '1.0',
            'description': 'Optional scale factor that multiplies the input array elements.'
        },
        'beta': {
            'type': 'float',
            'default': '0',
            'description': 'Optional delta value added to the scaled values before converting them to absolute values.'
        },
        'output': {
            'number': '1',
            'description': 'The function returns the output array that has the same size as the src. Each element of the output array is the absolute value of the corresponding element in the scaled input array, converted to 8-bit.'
        }
    }
    return info

def get_convexHull_params_info():
    info = {
        'points': {
            'format': 'numpy.ndarray',
            'type': 'float32',
            'description': 'Input 2D point set, stored in a numpy array of shape (N, 2), where N is the number of points.'
        },
        'clockwise': {
            'type': 'bool',
            'default': 'False',
            'description': 'Orientation flag. If true, the output convex hull is oriented clockwise. Otherwise, it is oriented counter-clockwise.'
        },
        'returnPoints': {
            'type': 'bool',
            'default': 'True',
            'description': 'Operation flag. When true, the function returns convex hull points. Otherwise, it returns indices of the convex hull points.'
        },
        'output': {
            'number': '1',
            'description': 'Output convex hull, which is either an integer numpy array of indices or a numpy array of points depending on the returnPoints flag.'
        }
    }
    return info

def get_copyMakeBorder_params_info():
    info = {
        'src': {
            'format': 'numpy.ndarray',
            'type': 'uint8 or float32',
            'description': 'Source image, a numpy array of any depth.'
        },
        'top': {
            'type': 'int',
            'description': 'Number of pixels to add to the top of the image.'
        },
        'bottom': {
            'type': 'int',
            'description': 'Number of pixels to add to the bottom of the image.'
        },
        'left': {
            'type': 'int',
            'description': 'Number of pixels to add to the left of the image.'
        },
        'right': {
            'type': 'int',
            'description': 'Number of pixels to add to the right of the image.'
        },
        'borderType': {
            'type': 'int',
            'default': 'cv2.BORDER_CONSTANT',
            'description': 'Flag representing the border type. Can be cv2.BORDER_CONSTANT, cv2.BORDER_REFLECT, cv2.BORDER_WRAP, etc.'
        },
        'value': {
            'format': 'tuple',
            'type': 'int',
            'default': '(0, 0, 0)',
            'flag': 'None',
            'description': 'Color of border if border type is cv2.BORDER_CONSTANT. It is ignored for other border types.'
        },
        'output': {
            'number': '1',
            'description': 'Destination image with the border added around the source image.'
        }
    }
    return info

def get_copyTo_params_info():
    info = {
        'src': {
            'format': 'numpy.ndarray',
            'type': 'uint8',
            'description': 'Source image, a numpy array of any depth.'
        },
        'mask': {
            'format': 'numpy.ndarray',
            'type': 'uint8 as same as',
            'description': 'Source image, a numpy array of any depth.'
        },
        'output': {
            'number': '1',
            'description': 'Destination image with the border added around the source image.'
        }
    }
    return info

def get_cornerEigenValsAndVecs_params_info():
    info = {
        'src': {
            'format': 'numpy.ndarray',
            'type': 'uint8',
            'description': 'Input 8-bit single-channel image, either or floating-point.'
        },
        'blockSize': {
            'type': 'int',
            'description': 'The size of the neighborhood considered for each pixel.'
        },
        'ksize': {
            'type': 'int',
            'description': 'Aperture parameter for the Sobel operator.'
        },
        'borderType': {
            'type': 'int',
            'default': 'cv2.BORDER_DEFAULT',
            'description': 'Pixel extrapolation method. BORDER_WRAP is not supported.'
        },
        'output': {
            'number': '1',
            'description': 'The destination image that stores the results. It is of the same size as the source image and has a type of CV_32FC(6).'
        }
    }
    return info

def get_cornerHarris_params_info():
    info = {
        'src': {
            'format': 'numpy.ndarray',
            'type': 'uint8',
            'description': 'Input 8-bit single-channel image, either or floating-point.'
        },
        'blockSize': {
            'type': 'int',
            'description': 'Neighborhood size for each pixel.'
        },
        'ksize': {
            'type': 'int',
            'description': 'Aperture parameter for the Sobel operator.'
        },
        'k': {
            'type': 'float',
            'description': 'Harris detector free parameter.'
        },
        'borderType': {
            'type': 'int',
            'default': 'cv2.BORDER_DEFAULT',
            'description': 'Pixel extrapolation method. BORDER_WRAP is not supported.'
        },
        'output': {
            'number': '1',
            'description': 'The destination image that stores the Harris detector responses. It is of the same size as src and has a type of CV_32FC1.'
        }
    }
    return info

def get_cornerMinEigenVal_params_info():
    info = {
        'src': {
            'format': 'numpy.ndarray',
            'type': 'uint8',
            'description': 'Input 8-bit single-channel image, either or floating-point.'
        },
        'blockSize': {
            'type': 'int',
            'description': 'Neighborhood size for each pixel.'
        },
        'ksize': {
            'type': 'int',
            'default': '3',
            'description': 'Aperture parameter for the Sobel operator. The default value is 3.'
        },
        'borderType': {
            'type': 'int',
            'default': 'cv2.BORDER_DEFAULT',
            'description': 'Pixel extrapolation method. BORDER_WRAP is not supported. The default value is BORDER_DEFAULT.'
        },
        'output': {
            'number': '1',
            'description': 'The destination image that stores the minimal eigenvalues of the gradient matrices. It is of the same size as src and has a type of CV_32FC1.'
        }
    }
    return info

def get_correctMatches_params_info():
    info = {
        'F': {
            'format': 'numpy.ndarray',
            'type': 'float64',
            'description': '3x3 fundamental matrix.'
        },
        'p1': {
            'format': 'numpy.ndarray',
            'type': 'float64',
            'description': '2x3 array containing the first set of points.'
        },
        'p2': {
            'format': 'numpy.ndarray',
            'type': 'float64 as same as',
            'description': '2x3 array containing the second set of points.'
        },
        'output': {
            'number': '2',
            'description': 'Two output arrays that store the optimized points. newPoints1 and newPoints2 are of the same format and type as the input points arrays.'
        }
    }
    return info


def get_countNonZero_params_info():
    info = {
        'src': {
            'format': 'numpy.ndarray',
            'type': 'uint8',
            'description': '8-bit single-channel array.'
        },
        'output': {
            'number': '1',
            'description': 'The number of non-zero elements in the input array.'
        }
    }
    return info

def get_cubeRoot_params_info():
    info = {
        'val': {
            'type': 'float',
            'description': 'A function argument.'
        },
        'output': {
            'number': '1',
            'description': 'The cube root of the input value. Negative arguments are handled correctly.'
        }
    }
    return info

def get_cvtColorTwoPlane_params_info():
    info = {
        'src1': {
            'format': 'numpy.ndarray',
            'type': 'uint8',
            'description': '8-bit image of the Y plane.'
        },
        'uv_plane': {
            'format': 'numpy.ndarray',
            'type': 'uint8',
            'description': '2-channel Image containing interleaved U/V plane.'
        },
        'code': {
            'type': 'int',
            'default': 'cv2.COLOR_YUV2BGR_NV21',
            'flag': 'None',
            'description': 'Specifies the type of conversion. Supported values are COLOR_YUV2BGR_NV12, COLOR_YUV2RGB_NV12, COLOR_YUV2BGRA_NV12, COLOR_YUV2RGBA_NV12, COLOR_YUV2BGR_NV21, COLOR_YUV2RGB_NV21, COLOR_YUV2BGRA_NV21, and COLOR_YUV2RGBA_NV21.'
        },
        'output': {
            'number': '1',
            'description': 'The output image after conversion.'
        }
    }
    return info

def get_dct_params_info():
    info = {
        'src': {
            'format': 'numpy.ndarray',
            'type': 'float32 or float64',
            'description': 'Input floating-point array. Can be 1D or 2D.'
        },
        'flags': {
            'type': 'int',
            'default': '0',
            'flag': 'None',
            'description': 'Transformation flags. DCT_INVERSE for inverse transform. DCT_ROWS to perform a 1D transform of each row. By default, it performs a forward 2D transform.'
        },
        'output': {
            'number': '1',
            'description': 'Output array of the same size and type as src. Contains the result of the DCT.'
        }
    }
    return info

def get_decomposeEssentialMat_params_info():
    info = {
        'E': {
            'format': 'numpy.ndarray',
            'type': 'float64',
            'description': 'The input essential matrix, a 3x3 numpy.ndarray.'
        },
        'output': {
            'number': '3',
            'description': 'The function returns three main outputs: two possible rotation matrices (R1, R2) and a normalized translation vector (t).'
        }
    }
    return info

def get_decomposeHomographyMat_params_info():
    info = {
        'H': {
            'format': 'numpy.ndarray',
            'type': 'float64',
            'description': 'The input homography matrix, a 3x3 numpy.ndarray, between two images.'
        },
        'K': {
            'format': 'numpy.ndarray',
            'type': 'float64',
            'description': 'The camera intrinsic matrix, a 3x3 numpy.ndarray.'
        },
        'output': {
            'number': '3',
            'description': 'The function returns three main outputs: arrays of possible rotations, translations, and plane normals.'
        }
    }
    return info

def get_decomposeProjectionMatrix_params_info():
    info = {
        'projMatrix': {
            'format': 'numpy.ndarray',
            'type': 'float64',
            'description': '4x3 matrix P.'
        },
        'output': {
            'number': '7',
            'description': 'The function returns seven main outputs: camera intrinsic matrix, external rotation matrix, translation vector, three optional rotation matrices around x, y, and z axes, and optional three Euler angles.'
        }
    }
    return info

def get_demosaicing_params_info():
    info = {
        'probImage': {
            'format': 'numpy.ndarray',
            'type': 'uint8',
            'description': '8-bit single-channel'
        },
        'c': {
            'type': 'int',
            'default':'cv2.COLOR_BayerRG2BGR',
            'flag': 'None',
            'description': 'Color space conversion code. It defines the type of demosaicing algorithm to use.'
        },
        'output': {
            'number': '1',
            'description': 'The function returns the demosaiced image as output.'
        }
    }
    return info

def get_detailEnhance_params_info():
    info = {
        'src': {
            'format': 'numpy.ndarray',
            'type': 'uint8',
            'description': 'Input 8-bit 3-channel image.'
        },
        'sigma_s': {
            'type': 'float',
            'default': '10',
            'flag': 'None',
            'description': 'Range between 0 to 200. Affects the scale of detail enhancement.'
        },
        'sigma_r': {
            'type': 'float',
            'default': '0.15',
            'flag': 'None',
            'description': 'Range between 0 to 1. Affects the degree of detail enhancement.'
        },
        'output': {
            'number': '1',
            'description': 'The function returns the enhanced image as output.'
        }
    }
    return info

def get_dft_params_info():
    info = {
        'src': {
            'format': 'numpy.ndarray',
            'type': 'float32 or float64',
            'description': 'Input array that could be real or complex.'
        },
        'flags': {
            'type': 'int',
            'default': 'cv2.DFT_COMPLEX_OUTPUT',
            'flag': 'None',
            'description': 'Transformation flags, representing a combination of the DftFlags. For example, DFT_INVERSE for inverse transform, DFT_SCALE to scale the result by 1/N, DFT_ROWS to perform a transform on each row individually.'
        },
        'output': {
            'number': '1',
            'description': 'The function returns the Discrete Fourier Transform of the input array.'
        }
    }
    return info

def get_distanceTransform_params_info():
    info = {
        'src': {
            'format': 'numpy.ndarray',
            'type': 'uint8',
            'description': '8-bit single-channel (binary) source image.'
        },
        'distType': {
            'type': 'int',
            'description': 'Type of distance. It can be one of DistanceTypes: DIST_L1, DIST_L2, etc.'
        },
        'n': {
            'type': 'int',
            'default': '3',
            'flag': 'None',
            'description': 'Size of the distance transform mask. It can be one of DistanceTransformMasks: DIST_MASK_3, DIST_MASK_5, DIST_MASK_PRECISE, etc. For DIST_L1 or DIST_C, it is forced to 3.'
        },
        'output': {
            'number': '1',
            'description': 'The function returns an image with calculated distances.'
        }
    }
    return info

def get_distanceTransformWithLabels_params_info():
    info = {
        'src': {
            'format': 'numpy.ndarray',
            'type': 'uint8',
            'description': '8-bit single-channel (binary) source image.'
        },
        'distType': {
            'type': 'int',
            'description': 'Type of distance. It can be one of DistanceTypes: DIST_L1, DIST_L2, etc.'
        },
        'ksize': {
            'type': 'int',
            'description': 'Size of the distance transform mask. It can be one of DistanceTransformMasks: DIST_MASK_3, DIST_MASK_5, etc. For DIST_L1 or DIST_C, it is forced to 3.'
        },
        'output': {
            'number': '2',
            'description': 'The function returns an image with calculated distances and a label map.'
        }
    }
    return info

def get_divSpectrums_params_info():
    info = {
        'a': {
            'format': 'numpy.ndarray',
            'type': 'float32',
            'description': 'First input array, result of a real or complex Fourier transform.'
        },
        'b': {
            'format': 'numpy.ndarray',
            'type': 'float32 as same as a',
            'description': 'Second input array, should be of the same size and type as the first.'
        },
        'flags': {
            'type': 'int',
            'default': '0',
            'flag': 'None',
            'description': 'Operation flags. DFT_ROWS indicates that each row of a and b is an independent 1D Fourier spectrum. If you do not want to use this flag, then simply add a `0` as value.'
        },
        'conjB': {
            'type': 'bool',
            'default': 'False',
            'description': 'Optional flag that conjugates the second input array before the division (True) or not (False).'
        },
        'output': {
            'number': '1',
            'description': 'The function returns the per-element division of two Fourier spectrums.'
        }
    }
    return info

def get_edgePreservingFilter_params_info():
    info = {
        'src': {
            'format': 'numpy.ndarray',
            'type': 'uint8, 3-channel',
            'description': 'Input 8-bit 3-channel image.'
        },
        'FILTER_flags': {
            'type': 'int',
            'default': 'cv2.RECURS_FILTER',
            'description': 'Edge preserving filters: cv2.RECURS_FILTER or cv2.NORMCONV_FILTER'
        },
        'sigma_s': {
            'type': 'int',
            'default': '60',
            'description': 'Range between 0 to 200. Spatial extent of the kernel, affects the amount of smoothing.'
        },
        'sigma_r': {
            'type': 'float',
            'default': '0.4',
            'flag': 'None',
            'description': 'Range between 0 to 1. Color space filtering sigma, smaller values result in sharper edges.'
        },
        'output': {
            'number': '1',
            'description': 'The function returns an output image with edge-preserving smoothing applied.'
        }
    }
    return info

def get_eigen_params_info():
    info = {
        'src': {
            'format': 'numpy.ndarray',
            'type': 'float32 or float64',
            'description': 'Input symmetric square matrix.'
        },
        'output': {
            'number': '3',
            'description': 'The function returns two main outputs: a vector of eigenvalues and a matrix of eigenvectors.'
        }
    }
    return info

def get_eigenNonSymmetric_params_info():
    info = {
        'src': {
            'format': 'numpy.ndarray',
            'type': 'float32 or float64',
            'description': 'Input non-symmetric square matrix.'
        },
        'output': {
            'number': '2',
            'description': 'The function returns two main outputs: a vector of eigenvalues and a matrix of eigenvectors.'
        }
    }
    return info

def get_ellipse_params_info():
    info = {
        'img': {
            'format': 'numpy.ndarray',
            'type': 'uint8 or float32 or float64',
            'description': 'Image on which the ellipse is drawn.'
        },
        'center': {
            'format': 'tuple',
            'type': 'int',
            'description': 'Center of the ellipse (x, y).'
        },
        'axes': {
            'format': 'tuple',
            'type': 'int',
            'description': 'Half of the size of the ellipse main axes.'
        },
        'angle': {
            'type': 'float',
            'description': 'angle Ellipse rotation angle in degrees.'
        },
        'startAngle': {
            'type': 'float',
            'default': '0',
            'flag': 'None',
            'description': 'Starting angle of the elliptic arc in degrees.'
        },
        'endAngle': {
            'type': 'float',
            'default': '300',
            'flag': 'None',
            'description': 'Ending angle of the elliptic arc in degrees.'
        },
        'color': {
            'format': 'tuple',
            'type': 'int',
            'description': 'Ellipse color (B, G, R).'
        },
        'thickness': {
            'type': 'int',
            'default': '1',
            'flag': 'None',
            'description': 'Thickness of the ellipse arc outline. Negative values, like cv2.FILLED, mean that a filled ellipse sector is to be drawn.'
        },
        'output': {
            'number': '1',
            'description': 'The function modifies the input image to draw the specified ellipse.'
        }
    }
    return info

def get_ellipse2Poly_params_info():
    info = {
        'center': {
            'format': 'tuple',
            'type': 'int',
            'description': 'Center of the arc (x, y).'
        },
        'axes': {
            'format': 'tuple',
            'type': 'int',
            'description': 'Half of the size of the ellipse main axes (major axis length, minor axis length).'
        },
        'angle': {
            'type': 'int',
            'description': 'Rotation angle of the ellipse in degrees.'
        },
        'startAngle': {
            'type': 'float',
            'default': '0',
            'flag': 'None',
            'description': 'Starting angle of the elliptic arc in degrees.'
        },
        'endAngle': {
            'type': 'float',
            'default': '300',
            'flag': 'None',
            'description': 'Ending angle of the elliptic arc in degrees.'
        },
        'delta': {
            'type': 'int',
            'flag': 'None',
            'default': '5',
            'description': 'Angle between the subsequent polyline vertices in degrees. It defines the approximation accuracy.'
        },
        'output': {
            'number': '1',
            'description': 'The function returns a list of points representing the approximated elliptic arc.'
        }
    }
    return info

def get_equalizeHist_params_info():
    info = {
        'src': {
            'format': 'numpy.ndarray',
            'type': 'uint8',
            'description': 'Source 8-bit single-channel image.'
        },
        'output': {
            'number': '1',
            'description': 'The function returns an image with equalized histogram.'
        }
    }
    return info

def get_estimateAffine2D_params_info():
    info = {
        'from_': {
            'format': 'numpy.ndarray',
            'type': 'float32',
            'description': 'First input 2D point set containing (X,Y).'
        },
        'to': {
            'format': 'numpy.ndarray',
            'type': 'float32 as samae as',
            'description': 'Second input 2D point set containing (x,y).'
        },
        'method': {
            'type': 'int',
            'default': 'cv2.RANSAC',
            'flag': 'None',
            'description': 'Robust method used to compute transformation. Options are cv2.RANSAC or cv2.LMEDS. cv2.RANSAC is the default method.'
        },
        'output': {
            'number': '2',
            'description': 'Returns a 2x3 affine transformation matrix and a vector of inliers.'
        }
    }
    return info

def get_estimateAffine3D_params_info():
    info = {
        'src': {
            'format': 'numpy.ndarray',
            'type': 'float32',
            'description': '1xN/Nx1 3-channel (X,Y,Z).'
        },
        'dst': {
            'format': 'numpy.ndarray',
            'type': 'float32',
            'description': '1xN/Nx1 3-channel (X,Y,Z).'
        },
        'output': {
            'number': '2',
            'description': 'Returns a 3x4 affine transformation matrix, a vector of inliers, and optionally the scale if not null.'
        }
    }
    return info

def get_fastNlMeansDenoising_params_info():
    info = {
        'src': {
            'format': 'numpy.ndarray',
            'type': 'uint8',
            'description': 'Input 1-channel, 2-channel, 3-channel, or 4-channel image'
        },
        'h': {
            'type': 'float',
            'description': 'Parameter regulating filter strength. For a single float, it applies to all channels. A list of floats applies to each channel respectively. Big h values perfectly remove noise but also remove image details, whereas smaller h values preserve details but also preserve some noise.'
        },
        'ksize': {
            'type': 'int',
            'default': '7',
            'flag': 'None',
            'description': 'Size in pixels of the template patch used to compute weights. Should be odd. Affects the performance linearly: greater value - greater denoising time. Recommended value is 7 pixels.'
        },
        'searchWindowSize': {
            'type': 'int',
            'default': '21',
            'flag': 'None',
            'description': 'Size in pixels of the window used to compute the weighted average for the given pixel. Should be odd. Affects the performance linearly: greater value - greater denoising time. Recommended value is 21 pixels.'
        },
        'output': {
            'number': '1',
            'description': 'The function returns the denoised image.'
        }
    }
    return info

def get_fastNlMeansDenoisingColored_params_info():
    info = {
        'src': {
            'format': 'numpy.ndarray',
            'type': 'uint8',
            'description': 'Input 8-bit 3-channel image.'
        },
        'h': {
            'type': 'float',
            'description': 'Parameter regulating filter strength for the luminance component. Larger h values perfectly remove noise but also remove image details, while smaller h values preserve details but also some noise.'
        },
        'hColor': {
            'type': 'float',
            'default': '10',
            'flag': 'None',
            'description': 'Parameter regulating filter strength for color components. It works similarly to h for the luminance component. A value of 10 is usually sufficient to remove colored noise without distorting colors.'
        },
        'ksize': {
            'type': 'int',
            'default': '7',
            'description': 'Size in pixels of the template patch used to compute weights. Should be odd. Affects performance linearly: greater value - greater denoising time. Recommended value is 7 pixels.'
        },
        'searchWindowSize': {
            'type': 'int',
            'default': '21',
            'flag': 'None',
            'description': 'Size in pixels of the window used to compute the weighted average for the given pixel. Should be odd. Affects performance linearly: greater value - greater denoising time. Recommended value is 21 pixels.'
        },
        'output': {
            'number': '1',
            'description': 'The function returns the denoised colored image.'
        }
    }
    return info

def get_fastNlMeansDenoisingColoredMulti_params_info():
    info = {
        'srcImgs': {
            'format': 'numpy.ndarray',
            'type': 'uint8',
            'description': 'Input sequence of 8-bit 3-channel images. All images should have the same type and size.'
        },
        'imgToDenoiseIndex': {
            'type': 'int',
            'default': '1',
            'flag': 'None',
            'description': 'Index of the target image to denoise in the srcImgs sequence.'
        },
        'size': {
            'type': 'int',
            'default': '3',
            'flag': 'None',
            'description': 'Number of surrounding images to use for target image denoising. Should be odd. The function uses images from imgToDenoiseIndex - temporalWindowSize / 2 to imgToDenoiseIndex + temporalWindowSize / 2.'
        },
        'h': {
            'type': 'float',
            'default': '2',
            'flag': 'None',
            'description': 'Parameter regulating filter strength for color components. It works similarly to h for the luminance component. A value of 10 is usually sufficient to remove colored noise without distorting colors.'
        },

        'output': {
            'number': '1',
            'description': 'The function returns the denoised colored image based on the input sequence.'
        }
    }
    return info

def get_fastNlMeansDenoisingMulti_params_info():
    info = {
        'srcImgs': {
            'format': 'numpy.ndarray',
            'type': 'uint8',
            'description': 'sequence of 8-bit 1-channel'
        },
        'imgToDenoiseIndex': {
            'type': 'int',
            'default': '1',
            'flag': 'None',
            'description': 'Index of the target image to denoise in the srcImgs sequence.'
        },
        'templateWindowSize': {
            'type': 'int',
            'default': '3',
            'flag': 'None',
            'description': 'Size in pixels of the template patch used to compute weights. Should be odd. Affects performance linearly: greater value - greater denoising time. Recommended value is 7 pixels.'
        },
        'h': {
            'type': 'float',
            'default': '10',
            'flag': 'None',
            'description': 'Parameter regulating filter strength for color components. It works similarly to h for the luminance component. A value of 10 is usually sufficient to remove colored noise without distorting colors.'
        },
        'output': {
            'number': '1',
            'description': 'The function returns the denoised grayscale or manually manipulated colorspace image based on the input sequence.'
        }
    }
    return info

def get_fillConvexPoly_params_info():
    info = {
        'img': {
            'format': 'numpy.ndarray',
            'type': 'uint8 or float32',
            'description': 'Input/output image.'
        },
        'points': {
            'format': 'numpy.ndarray',
            'type': 'int32',
            'description': 'Array of polygon vertices.'
        },
        'color': {
            'format': 'tuple',
            'type': 'uint8',
            'description': 'Polygon color.'
        },
        'output': {
            'number': '1',
            'description': 'The function draws a filled convex polygon on the input image.'
        }
    }
    return info

def get_find4QuadCornerSubpix_params_info():
    info = {
        'img': {
            'format': 'numpy.ndarray',
            'type': 'uint8',
            'description': 'Input image. It must be 8-bit single-channel.'
        },
        'cc': {
            'format': 'numpy.ndarray',
            'type': 'float32',
            'description': '2x2'
        },
        'ksize': {
            'format': 'tuple',
            'type': 'int',
            'description': 'Half of the side length of the search window. For example, if region_size=(5,5), then a 11x11 search window is used.'
        },
        'output': {
            'number': '2',
            'description': 'The function returns a boolean value indicating success or failure, and the refined coordinates of the corners.'
        }
    }
    return info

def get_filterSpeckles_params_info():
    info = {
        'img': {
            'format': 'numpy.ndarray',
            'type': 'uint8',
            'description': 'The input 16-bit signed disparity image.'
        },
        'newVal': {
            'type': 'int',
            'default': '0',
            'flag': 'None',
            'description': 'The disparity value used to paint-off the speckles.'
        },
        'ksize': {
            'type': 'int',
            'description': 'The maximum speckle size to consider it a speckle. Larger blobs are not affected by the algorithm.'
        },
        'maxDiff': {
            'type': 'int',
            'description': 'Maximum difference between neighbor disparity pixels to put them into the same blob. Note that since StereoBM, StereoSGBM, and may be other algorithms return a fixed-point disparity map, where disparity values are multiplied by 16, this scale factor should be taken into account when specifying this parameter value.'
        },
        'output': {
            'number': '2',
            'description': 'The function modifies the input image to filter out the speckles and optionally returns the buffer used for the operation.'
        }
    }
    return info

def get_findChessboardCorners_params_info():
    info = {
        'image': {
            'format': 'numpy.ndarray',
            'type': 'uint8',
            'description': 'Source chessboard view. It must be an 8-bit single-channel or color image.'
        },
        'patternSize': {
            'format': 'tuple',
            'type': 'int',
            'description': 'Number of inner corners per a chessboard row and column (patternSize = cv::Size(points_per_row,points_per_column) = cv::Size(columns,rows)).'
        },
        'CALIBflags': {
            'type': 'int',
            'default': 'cv2.CALIB_CB_ADAPTIVE_THRESH',
            'description': 'Various operation flags that can be zero or a combination of specific values: CALIB_CB_ADAPTIVE_THRESH, CALIB_CB_NORMALIZE_IMAGE, CALIB_CB_FILTER_QUADS, CALIB_CB_FAST_CHECK, CALIB_CB_PLAIN.'
        },
        'output': {
            'number': '2',
            'description': 'The function returns a boolean value indicating success or failure, and the detected corners in the output array.'
        }
    }
    return info

def get_findChessboardCornersSB_params_info():
    info = {
        'image': {
            'format': 'numpy.ndarray',
            'type': 'uint8',
            'description': 'Source chessboard view. It must be an 8-bit single-channel or color image.'
        },
        'patternSize': {
            'format': 'tuple',
            'type': 'int',
            'description': 'Number of inner corners per a chessboard row and column (patternSize = cv::Size(points_per_row,points_per_column) = cv::Size(columns,rows)).'
        },
        'CALIBflags': {
            'type': 'int',
            'default': 'cv2.CALIB_CB_ADAPTIVE_THRESH',
            'description': 'Various operation flags that can be zero or a combination of specific values: CALIB_CB_ADAPTIVE_THRESH, CALIB_CB_NORMALIZE_IMAGE, CALIB_CB_FILTER_QUADS, CALIB_CB_FAST_CHECK, CALIB_CB_PLAIN.'
        },
        'output': {
            'number': '2',
            'description': 'The function returns a boolean value indicating success or failure, and the detected corners in the output array.'
        }
    }
    return info

def get_findChessboardCornersSBWithMeta_params_info():
    info = {
        'image': {
            'format': 'numpy.ndarray',
            'type': 'uint8',
            'description': 'Source chessboard view. It must be an 8-bit single-channel or color image.'
        },
        'patternSize': {
            'format': 'tuple',
            'type': 'int',
            'default': '(3,3)',
            'flag':'None',
            'description': 'Number of inner corners per a chessboard row and column (patternSize = cv::Size(points_per_row,points_per_column) = cv::Size(columns,rows)).'
        },
        'CALIBflags': {
            'type': 'int',
            'default': 'cv2.CALIB_CB_NORMALIZE_IMAGE',
            'description': ' These influence the detection process and corner localization.'
        },
        'output': {
            'number': '3',
            'description': 'The function returns a boolean value indicating success or failure, the detected corners, and optional metadata array.'
        }
    }
    return info

def get_findCirclesGrid_params_info():
    info = {
        'image': {
            'format': 'numpy.ndarray',
            'type': 'uint8',
            'description': 'Grid view of input circles; it must be an 8-bit grayscale or color image.'
        },
        'patternSize': {
            'format': 'tuple',
            'type': 'int',
            'description': 'Number of circles per row and column (patternSize = Size(points_per_row, points_per_column)).'
        },
        'flags': {
            'type': 'int',
            'default': 'cv2.CALIB_CB_SYMMETRIC_GRID',
            'flag': 'None',
            'description': 'Operation flags, can be one of CALIB_CB_SYMMETRIC_GRID, CALIB_CB_ASYMMETRIC_GRID, CALIB_CB_CLUSTERING. These influence the detection process and grid type.'
        },
        'output': {
            'number': '2',
            'description': 'The function returns a boolean value indicating success or failure, and the detected centers.'
        }
    }
    return info

def get_findContours_params_info():
    info = {
        'image': {
            'format': 'numpy.ndarray',
            'type': 'uint8',
            'description': 'Source, an 8-bit single-channel image. Non-zero pixels are treated as 1s. Zero pixels remain 0s, treated as binary. Can also be a 32-bit integer image of labels (CV_32SC1) if mode is RETR_CCOMP or RETR_FLOODFILL.'
        },
        'Contoursmode': {
            'type': 'int',
            'default': 'cv2.RETR_CCOMP',
            'flag': 'None',
            'description': 'Contour retrieval mode, influencing the hierarchy of contours.'
        },
        'Contoursmethod': {
            'type': 'int',
            'default': 'cv2.CHAIN_APPROX_SIMPLE',
            'description': 'Contour approximation method, defining the contour shape approximation.'
        },
        'output': {
            'number': '2',
            'description': 'The function returns the detected contours and optionally the hierarchy of contours.'
        }
    }
    return info

def get_findEssentialMat_params_info():
    info = {
        'points1': {
            'format': 'numpy.ndarray',
            'type': 'float32 | float64',
            'description': 'Array of N (N >= 5) 2D points from the first image. The point coordinates should be floating-point.'
        },
        'points2': {
            'format': 'numpy.ndarray',
            'type': 'float32 | float64 as same as',
            'description': '2D points Array of the second image points of the same size and format as points1.'
        },
        'cameraMatrix': {
            'format': 'numpy.ndarray',
            'type': 'float32 | float64 as same as',
            'description': 'Camera intrinsic matrix.'
        },
        'method': {
            'type': 'int',
            'default': 'cv2.RANSAC',
            'description': 'Method for computing an essential matrix. Can be RANSAC or LMEDS.'
        },
        'prob': {
            'type': 'float',
            'default': '0.8',
            'flag': 'None',
            'description': 'Desired level of confidence (probability) that the estimated matrix is correct.'
        },
        'threshold': {
            'type': 'float',
            'description': 'Maximum distance from a point to an epipolar line in pixels, beyond which the point is considered an outlier.'
        },
        'maxIters': {
            'type': 'int',
            'description': 'Maximum number of robust method iterations.'
        },
        'output': {
            'number': '2',
            'description': 'The function returns the essential matrix and optionally the mask.'
        }
    }
    return info

def get_findFundamentalMat_params_info():
    info = {
        'points1': {
            'format': 'numpy.ndarray',
            'type': 'float32 | float64',
            'description': 'Array of N (N >= 5) 2D points from the first image. The point coordinates should be floating-point.'
        },
        'points2': {
            'format': 'numpy.ndarray',
            'type': 'float32 | float64 as same as',
            'description': '2D points Array of the second image points of the same size and format as points1.'
        },
        'method': {
            'type': 'int',
            'default': 'cv2.RANSAC',
            'description': 'Method for computing an essential matrix. Can be RANSAC or LMEDS.'
        },
        'ransacReprojThreshold': {
            'type': 'float',
            'description': 'Maximum distance from a point to an epipolar line in pixels, beyond which the point is considered an outlier. Used only for RANSAC.'
        },
        'prob': {
            'type': 'float',
            'default': '0.8',
            'flag': 'None',
            'description': 'Desired level of confidence (probability) that the estimated matrix is correct.'
        },
        'maxIters': {
            'type': 'int',
            'description': 'Maximum number of robust method iterations.'
        },
        'output': {
            'number': '2',
            'description': 'The function returns the fundamental matrix and optionally the mask.'
        }
    }
    return info

def get_findHomography_params_info():
    info = {
        'points1': {
            'format': 'numpy.ndarray',
            'type': 'float32',
            'description': 'Array of N (N >= 5) 2D points from the first image. The point coordinates should be floating-point.'
        },
        'points2': {
            'format': 'numpy.ndarray',
            'type': 'float32 as same as',
            'description': '2D points Array of the second image points of the same size and format as points1.'
        },
        'output': {
            'number': '2',
            'description': 'The function returns the homography matrix and optionally the mask.'
        }
    }
    return info

def get_findTransformECC_params_info():#bug
    info = {
        'templateImage': {
            'format': 'numpy.ndarray',
            'type': 'uint8 ',
            'description': '8-bit single-channel template image.'
        },
        'inputImage': {
            'format': 'numpy.ndarray',
            'type': 'uint8',
            'description': '8-bit single-channel input image which should be warped with the final warpMatrix in order to provide an image similar to templateImage.'
        },
        'cameraMatrix': {
            'format': 'numpy.ndarray',
            'type': 'flaot32',
            'description': 'warp_matrix 2x3.'
        },
        'motionType': {
            'type': 'int',
            'default': 'cv2.MOTION_AFFINE',
            'flag': 'None',
            'description': 'Specifies the type of motion: MOTION_TRANSLATION, MOTION_EUCLIDEAN, MOTION_AFFINE, or MOTION_HOMOGRAPHY.'
        },
        'criteria': {
            'format': 'tuple',
            'type': 'int',
            'default': '(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10000, 1e-5)',
            'flag': 'None',
            'description': 'Specifies the termination criteria of the ECC algorithm; defines the threshold of the increment in the correlation coefficient between two iterations.'
        },
        'output': {
            'number': '2',
            'description': 'The function returns the final enhanced correlation coefficient and the optimum transformation (warpMatrix) with respect to ECC criterion.'
        }
    }
    return info

def get_fitLine_params_info():
    info = {
        'points': {
            'format': 'numpy.ndarray',
            'type': 'uint8 ',
            'description': 'Input vector of 2D or 3D points.'
        },
        'distType': {
            'type': 'int',
            'description': 'Distance used by the M-estimator.'
        },
        'param': {
            'type': 'int',
            'default': '0',
            'flag': 'None',
            'description': 'Numerical parameter (C) for some types of distances. If 0, an optimal value is chosen.'
        },
        'reps': {
            'type': 'float',
            'description': 'Sufficient accuracy for the radius (distance between the coordinate origin and the line).'
        },
        'aeps': {
            'type': 'float',
            'default': '0.01',
            'flag': 'None',
            'description': 'Sufficient accuracy for the angle. 0.01 would be a good default value for reps and aeps.'
        },
        'output': {
            'number': '1',
            'description': 'The function returns the fitted line parameters as specified in the line parameter.'
        }
    }
    return info

def get_goodFeaturesToTrack_params_info():
    info = {
        'image': {
            'format': 'numpy.ndarray',
            'type': 'uint8',
            'description': 'Input 8-bit single-channel image.'
        },
        'maxCorners': {
            'type': 'int',
            'description': 'Maximum number of corners to return. If there are more corners than are found, the strongest of them is returned. maxCorners <= 0 implies that no limit on the maximum is set and all detected corners are returned.'
        },
        'qualityLevel': {
            'type': 'float',
            'description': 'Parameter characterizing the minimal accepted quality of image corners.'
        },
        'minDistance': {
            'type': 'float',
            'description': 'Minimum possible Euclidean distance between the returned corners.'
        },
        'blockSize': {
            'type': 'int',
            'description': 'Size of an average block for computing a derivative covariation matrix over each pixel neighborhood.'
        },
        'output': {
            'number': '2',
            'description': 'Output vector of detected corners.'
        }
    }
    return info

def get_goodFeaturesToTrackWithQuality_params_info():
    info = {
        'image': {
            'format': 'numpy.ndarray',
            'type': 'uint8',
            'description': 'Input 8-bit single-channel image.'
        },
        'maxCorners': {
            'type': 'int',
            'description': 'Maximum number of corners to return. If there are more corners than are found, the strongest of them is returned. maxCorners <= 0 implies that no limit on the maximum is set and all detected corners are returned.'
        },
        'qualityLevel': {
            'type': 'float',
            'description': 'Parameter characterizing the minimal accepted quality of image corners.'
        },
        'minDistance': {
            'type': 'float',
            'description': 'Minimum possible Euclidean distance between the returned corners.'
        },
        'mask': {
            'format': 'numpy.ndarray',
            'type': 'uint8',
            'description': '8-bit single-channel image'
        },
        'blockSize': {
            'type': 'int',
            'description': 'Size of an average block for computing a derivative covariation matrix over each pixel neighborhood.'
        },
        'output': {
            'number': '2',
            'description': 'Output vector of detected corners.'
        }
    }
    return info

def get_grabCut_params_info():
    info = {
        'img': {
            'format': 'numpy.ndarray',
            'type': 'uint8',
            'description': 'Input 8-bit 3-channel image.'
        },
        'mask': {
            'format': 'numpy.ndarray',
            'type': 'uint8',
            'description': 'Input/output 8-bit single-channel mask. Initialized by the function when mode is GC_INIT_WITH_RECT. Elements may have one of GrabCutClasses.'
        },
        'imgRect': {
            'format': 'tuple',
            'type': 'uint8',
            'description': 'ROI containing a segmented object. Pixels outside the ROI are marked as "obvious background". Used only when mode==GC_INIT_WITH_RECT.'
        },
        'bgdModel': {
            'format': 'numpy.ndarray',
            'type': 'float64',
            'flag': 'None',
            'description': '1x13 Temporary array for the background model. Do not modify it while processing the same image.'
        },
        'fgdModel': {
            'format': 'numpy.ndarray',
            'type': 'float64',
            'flag': 'None',
            'description': '1x13 Temporary array for the foreground model. Do not modify it while processing the same image.'
        },
        'iterCount': {
            'type': 'int',
            'default': '5',
            'flag': 'None',
            'description': 'Number of iterations the algorithm should make before returning the result. The result can be refined with further calls.'
        },
        'mode': {
            'type': 'int',
            'default': 'cv2.GC_INIT_WITH_RECT',
            'flag': 'None',
            'description': 'Operation mode. One of the GrabCutModes.'
        },
        'output': {
            'number': '3',
            'description': 'Output vector of detected corners.'
        }
    }
    return info

def get_groupRectangles_params_info():
    info = {
        'rectList': {
            'format': 'tuple',
            'type': 'int',
            'description': '3x4 matrix Input/output vector of rectangles. Input rectangles and returns the ones that passed the grouping.'
        },
        'groupThreshold': {
            'type': 'int',
            'description': 'Minimum possible number of rectangles minus 1. The threshold is used in a group of rectangles to retain it.'
        },
        'eps': {
            'type': 'float',
            'default': '0.2',
            'flag': 'None',
            'description': 'Minimum possible number of rectangles minus 1. The threshold is used in a group of rectangles to retain it.'
        },
        'output': {
            'number': '2',
            'description': 'Output vector of detected corners.'
        }
    }
    return info

def get_idft_params_info():
    info = {
        'src': {
            'format': 'numpy.ndarray',
            'type': 'float32',
            'description': 'Input array that could be real or complex.'
        },
        'output': {
            'number': '1',
            'description': 'Output vector of detected corners.'
        }
    }
    return info

def get_imdecode_params_info():
    info = {
        'buf': {
            'format': 'numpy.ndarray',
            'type': 'uint8',
            'description': 'Input array or vector of bytes.'
        },
        'flags': {
            'type': 'int',
            'description': 'The same flags as in cv::imread, see cv::ImreadModes.'
        },
        'output': {
            'number': '1',
            'description': 'The decoded image as a Mat object. Returns an empty matrix if the buffer is too short, contains invalid data, or decoding fails.'
        }
    }
    return info

def get_imdecodemulti_params_info():
    info = {
        'buf': {
            'format': 'numpy.ndarray',
            'type': 'uint8',
            'description': 'Input array or vector of bytes.'
        },
        'flags': {
            'type': 'int',
            'description': 'The same flags as in cv::imread, see cv::ImreadModes.'
        },
        'output': {
            'number': '1',
            'description': 'The decoded image as a Mat object. Returns an empty matrix if the buffer is too short, contains invalid data, or decoding fails.'
        }
    }
    return info

def get_imencode_params_info():
    info = {
        'ext': {
            'type': 'str',
            'description': 'Input array or vector of bytes.'
        },
        'buf': {
            'format': 'numpy.ndarray',
            'type': 'uint8',
            'description': 'Input array or vector of bytes.'
        },
        'output': {
            'number': '2',
            'description': 'The decoded image as a Mat object. Returns an empty matrix if the buffer is too short, contains invalid data, or decoding fails.'
        }
    }
    return info

def get_inRange_params_info():
    info = {
        'src': {
            'format': 'numpy.ndarray',
            'type': 'uint8',
            'description': 'Input array or vector of bytes.'
        },
        'lowerb': {
            'format': 'numpy.ndarray',
            'type': 'uint8',
            'description': 'Input array or vector of bytes.'
        },
        'upperb': {
            'format': 'numpy.ndarray',
            'type': 'uint8',
            'description': 'Input array or vector of bytes.'
        },
        'output': {
            'number': '1',
            'description': 'The decoded image as a Mat object. Returns an empty matrix if the buffer is too short, contains invalid data, or decoding fails.'
        }
    }
    return info

def get_initCameraMatrix2D_params_info():#Bug
    info = {
        'objectPoints': {
            'format': 'numpy.ndarray',
            'type': 'float32',
            'description': '3xN/Nx3 1-channel or 1xN/Nx1 3-channel,'
        },
        'imagePoints': {
            'format': 'numpy.ndarray',
            'type': 'float32',
            'description': ' 2xN/Nx2 1-channel or 1xN/Nx1 2-channel.'
        },
        'imageSize': {
            'format': 'tuple',
            'type': 'int',
            'description': 'Size of the image used only to initialize the intrinsic camera matrix.'
        },
        'output': {
            'number': '1',
        }
    }
    return info

def get_initInverseRectificationMap_params_info():
    info = {
        'cameraMatrix': {
            'format': 'numpy.ndarray',
            'type': 'float32',
            'description': 'Input camera matrix A=[fx 0 cx; 0 fy cy; 0 0 1].'
        },
        'distCoeffs': {
            'format': 'numpy.ndarray',
            'type': 'float32',
            'flag': 'None'
,            'description': ''
        },
        'R': {
            'default': 'None',
            'type': 'float32',
            'description': 'Optional rectification transformation in the object space (3x3 matrix). If the matrix is empty, the identity transformation is assumed.'
        },
        'newCameraMatrix': {
            'format': 'numpy.ndarray',
            'type': 'float32',
            'description': 'New camera matrix'
        },
        'imageSize': {
            'format': 'tuple',
            'type':'int',
            'description': 'Distorted image size.'
        },
        'm1type': {
            'type': 'int',
            'default':'cv2.CV_32FC1',
            'flag': 'None',
            'description': 'Type of the first output map. Can be CV_32FC1, CV_32FC2, or CV_16SC2.'
        },
        'output': {
            'number': '2',
            'description': 'The function returns two maps for remap. These maps can be used for projection and inverse-rectification transformation.'
        }
    }
    return info

def get_initUndistortRectifyMap_params_info():
    info = {
        'cameraMatrix': {
            'format': 'numpy.ndarray',
            'type': 'float32',
            'description': 'Input camera matrix A=[fx 0 cx; 0 fy cy; 0 0 1].'
        },
        'distCoeffs': {
            'format': 'numpy.ndarray',
            'type': 'float32',
            'flag': 'None'
,            'description': ''
        },
        'R': {
            'default': 'None',
            'type': 'float32',
            'description': 'Optional rectification transformation in the object space (3x3 matrix). If the matrix is empty, the identity transformation is assumed.'
        },
        'newCameraMatrix': {
            'format': 'numpy.ndarray',
            'type': 'float32',
            'description': 'New camera matrix'
        },
        'imageSize': {
            'format': 'tuple',
            'type':'int',
            'description': 'Distorted image size.'
        },
        'm1type': {
            'type': 'int',
            'default':'cv2.CV_32FC1',
            'flag': 'None',
            'description': 'Type of the first output map. Can be CV_32FC1, CV_32FC2, or CV_16SC2.'
        },
        'output': {
            'number': '2',
            'description': 'The function returns two maps for remap. These maps can be used for projection and inverse-rectification transformation.'
        }
    }
    return info

def get_inpaint_params_info():
    info = {
        'src': {
            'format': 'numpy.ndarray',
            'type': 'uint8',
            'description': 'Input image. 8-bit 3-channel image.'
        },
        'inpaintMask': {
            'format': 'numpy.ndarray',
            'type': 'uint8',
            'description': 'Inpainting mask, an 8-bit single-channel image. Non-zero pixels indicate the area that needs to be inpainted.'
        },
        'inpaintRadius': {
            'type': 'float',
            'description': 'Radius of a circular neighborhood of each point inpainted that is considered by the algorithm.'
        },
        'Impaint_flags': {
            'type': 'int',
            'default': 'cv2.INPAINT_TELEA',
            'description': 'Inpainting method. It could be cv::INPAINT_NS for Navier-Stokes based method or cv::INPAINT_TELEA for the method by Alexandru Telea.'
        },
        'output': {
            'number': '1',
            'description': 'Output image with the same size and type as the input image.'
        }
    }
    return info

def get_invertAffineTransform_params_info():
    info = {
        'input': {
            'format': 'numpy.ndarray',
            'type': 'float32',
            'description': '3x2'
        },
        'output': {
            'number': '1',
            'description': 'The function returns the inverse of the affine transformation matrix.'
        }
    }
    return info

def get_line_params_info():
    info = {
        'img': {
            'format': 'numpy.ndarray',
            'type': 'uint8 or float32 or float64',
            'description': 'The image on which the line is to be drawn. It can be of any type.'
        },
        'pt1': {
            'format': 'tuple',
            'type': 'int',
            'description': 'The first point of the line segment, represented as a tuple of two integers (x, y).'
        },
        'pt2': {
            'format': 'tuple',
            'type': 'int',
            'description': 'The second point of the line segment, represented as a tuple of two integers (x, y).'
        },
        'color': {
            'format': 'tuple',
            'type': 'int',
            'description': 'The color of the line, represented as a tuple of three integers (B, G, R).'
        },
        'thickness': {
            'type': 'int',
            'default': '1',
            'description': 'The thickness of the line. The default value is 1.'
        },
        'lineType': {
            'type': 'int',
            'default': 'cv2.LINE_8',
            'flag': 'None',
            'description': 'The type of the line. The default is 8-connected (LineTypes.LINE_8). Other options include LineTypes.LINE_4 and LineTypes.LINE_AA for antialiased line.'
        },
        'output': {
            'number': '1',
            'description': 'The function returns the image with the line drawn on it.'
        }
    }
    return info

def get_magnitude_params_info():
    info = {
        'x': {
            'format': 'numpy.ndarray',
            'type': 'float32 or float64',
            'description': 'Floating-point array of x-coordinates of the vectors.'
        },
        'y': {
            'format': 'numpy.ndarray',
            'type': 'float32 or float64 as same as',
            'description': 'Floating-point array of y-coordinates of the vectors; it must have the same size as x.'
        },
        'output': {
            'number': '1',
            'description': 'Output array of the same size and type as x containing the magnitudes of the corresponding vectors.'
        }
    }
    return info

def get_matMulDeriv_params_info():
    info = {
        'A': {
            'format': 'numpy.ndarray',
            'type': 'float32 or float64',
            'description': 'First multiplied matrix.'
        },
        'B': {
            'format': 'numpy.ndarray',
            'type': 'float32 or float64 as same as',
            'description': 'Second multiplied matrix.'
        },
        'output': {
            'number': '2',
            'description': 'Two output derivative matrices. dABdA is the derivative of the matrix product with respect to the first matrix, having size A.rows*B.cols x A.rows*A.cols. dABdB is the derivative with respect to the second matrix, having size A.rows*B.cols x B.rows*B.cols.'
        }
    }
    return info

def get_matchShapes_params_info():
    info = {
        'contour1': {
            'format': 'numpy.ndarray',
            'type': 'float32',
            'description': 'First contour or grayscale image.'
        },
        'contour2': {
            'format': 'numpy.ndarray',
            'type': 'float32',
            'description': 'Second contour or grayscale image.'
        },
        'method': {
            'type': 'int',
            'default': '2',
            'flag':'None',
            'description': 'Comparison method. It can be one of #ShapeMatchModes.'
        },
        'parameter': {
            'type': 'double',
            'default': '0',
            'description': 'Method-specific parameter. Currently, this parameter is not supported and must be set to 0.'
        },
        'output': {
            'number': '1',
            'description': 'The function returns a measure of similarity between the two shapes. The lower the result, the more similar the two shapes are.'
        }
    }
    return info

def get_mixChannels_params_info():
    info = {
        'src1': {
            'format': 'numpy.ndarray',
            'type': 'uin8 or float32',
            'description': '8-bit 3-channel'
        },
        'src2': {
            'format': 'numpy.ndarray',
            'type': 'uin8 or float32 as same as',
            'description': '8-bit 3-channel'
        },
        'fromTo': {
            'type': 'int',
            'default': ' [0, 2, 1, 1, 2, 0]',
            'flag':'None',
            'description': 'Array of index pairs specifying which channels are copied and where; fromTo[k*2] is a 0-based index of the input channel in src, fromTo[k*2+1] is an index of the output channel in dst; the continuous channel numbering is used.'
        },
        'output': {
            'number': '1',
            'description': 'The function modifies the destination matrices based on the specified channel mappings.'
        }
    }
    return info

def get_moments_params_info():#bug
    info = {
        'array': {
            'format': 'numpy.ndarray',
            'type': 'uint8 or float32',
            'description': '8-bit single-channel'
        },
        'binaryImage': {
            'type': 'bool',
            'default': 'False',
            'description': 'If true, all non-zero image pixels are treated as 1s. Used for images only.'
        },
        'output': {
            'number': '1',
            'description': 'The calculated moments up to the third order of the input shape.'
        }
    }
    return info

def get_morphologyEx_params_info():
    info = {
        'src': {
            'format': 'numpy.ndarray',
            'type': 'uint8 or float32',
            'description': 'Source image. The number of channels can be arbitrary.'
        },
        'op': {
            'type': 'int',
            'default': 'cv2.MORPH_OPEN',
            'flag': 'None',
            'description': 'Type of a morphological operation, see #MorphTypes.'
        },
        'kernel': {
            'format': 'tuple',
            'type': 'int',
            'description': 'It can be created using #getStructuringElement.'
        },
        'output': {
            'number': '1',
            'description': 'The calculated moments up to the third order of the input shape.'
        }

    }
    return info

def get_mulSpectrums_params_info():
    info = {
        'a': {
            'format': 'numpy.ndarray',
            'type': 'float32 or float64',
            'description': 'First input array, result of a real or complex Fourier transform.'
        },
        'b': {
            'format': 'numpy.ndarray',
            'type': 'float32 or float64 as same as',
            'description': 'Second input array of the same size and type as the first. Also, result of a real or complex Fourier transform.'
        },
        'flags': {
            'type': 'int',
            'default': '0',
            'flag': 'None',
            'description': 'Operation flags. The only supported flag is cv::DFT_ROWS, which indicates that each row of the input arrays is an independent 1D Fourier spectrum. If not using this flag, set to `0`.'
        },
        'output': {
            'number': '1',
            'description': 'The function returns the element-wise multiplication result of the two input Fourier spectrums.'
        }
    }
    return info

def get_mulTransposed_params_info():
    info = {
        'src': {
            'format': 'numpy.ndarray',
            'type': 'uint8 or float32',
            'description': 'Input matrix. Unlike cv::gemm, it can handle matrices of any depth.'
        },
        'aTa': {
            'type': 'bool',
            'description': 'A flag indicating the multiplication order. If true, performs (src - delta)^T * (src - delta), otherwise (src - delta) * (src - delta)^T.'
        },
        'output': {
            'number': '1',
            'description': 'The function returns the matrix product, scaled and adjusted by the delta if provided.'
        }
    }
    return info

def get_norm_params_info():
    info = {
        'src1': {
            'format': 'numpy.ndarray',
            'type': 'uint8',
            'description': 'First input array. For single array operation, it calculates its absolute norm. For two arrays, it calculates the absolute difference norm or the relative difference norm with src2.'
        },
        'src2': {
            'format': 'numpy.ndarray',
            'type': 'uint8 as same as',
            'description': 'Second input array of the same size and the same type as src1. This parameter is used for calculating the norm between src1 and src2.',
        },
        'normType': {
            'type': 'int',
            'default': 'cv2.NORM_L1',
            'description': 'Type of the norm to calculate. It can be one of cv2.NormTypes. If not specified, NORM_L2 is used for a single array, and the absolute difference norm is used for two arrays.'
        },
        'output': {
            'number': '1',
            'description': 'The function returns the norm of src1 if only src1 is provided. If both src1 and src2 are provided, it returns the absolute difference norm or the relative difference norm between src1 and src2, depending on the normType.'
        }
    }
    return info

def get_normalize_params_info():
    info = {
        'src': {
            'format': 'numpy.ndarray',
            'type': 'uint8',
            'description': 'Input array to be normalized.'
        },
        'alpha': {
            'type': 'float',
            'description': 'Norm value to normalize to or the lower range boundary in case of range normalization.'
        },
        'beta': {
            'type': 'float',
            'description': 'Upper range boundary in case of range normalization; not used for norm normalization.'
        },
        'normType': {
            'type': 'int',
            'default': 'cv2.NORM_L2',
            'description': 'Type of the norm to calculate. It can be one of cv2.NormTypes. If not specified, NORM_L2 is used for a single array, and the absolute difference norm is used for two arrays.'
        },
        'output': {
            'number': '1',
            'description': 'The function returns the dst array, which contains the normalized values according to the specified norm or range.'
        }
    }
    return info

def get_phase_params_info():
    info = {
        'x': {
            'format': 'numpy.ndarray',
            'type': 'float32',
            'description': 'Input floating-point array of x-coordinates of 2D vectors.'
        },
        'y': {
            'format': 'numpy.ndarray',
            'type': 'float32',
            'description': 'Input floating-point array of y-coordinates of 2D vectors. It must have the same size and the same type as x.'
        },
        'output': {
            'number': '1',
            'description': 'The function returns the angle array, which contains the rotation angles of the 2D vectors formed by x and y.'
        }
    }
    return info

def get_phaseCorrelate_params_info():
    info = {
        'points1': {
            'format': 'numpy.ndarray',
            'type': 'float32 or float64',
            'description': 'Source floating point array (CV_32FC1 or CV_64FC1).'
        },
        'points2': {
            'format': 'numpy.ndarray',
            'type': 'float32 or float64 as same as',
            'description': 'Source floating point array (CV_32FC1 or CV_64FC1).'
        },
        'output': {
            'number': '1',
            'description': 'The function returns the detected sub-pixel phase shift between the two input arrays along with the response value indicating the strength of the correlation.'
        }
    }
    return info

def get_pointPolygonTest_params_info():
    info = {
        'contour': {
            'format': 'numpy.ndarray',
            'type': 'float32',
            'description': 'Input contour, represented as a 2D point array.'
        },
        'pt': {
            'format': 'tuple',
            'type': 'int',
            'description': 'The point to be tested against the contour.'
        },
        'measureDist': {
            'type': 'bool',
            'description': 'Flag indicating whether to compute the signed distance from the point to the nearest contour edge. If False, the function merely checks the point\'s relation to the contour.'
        },
        'output': {
            'number': '1',
            'description': 'The function returns a positive value if the point is inside the contour, a negative value if it is outside, and zero if it is on the contour edge. When measureDist is True, the return value is the signed distance to the nearest contour edge.'
        }
    }
    return info

def get_polarToCart_params_info():
    info = {
        'magnitude': {
            'format': 'numpy.ndarray',
            'type': 'float32 or float64',
            'description': '1xN/Nx1 2-channel'
        },
        'angle': {
            'format': 'numpy.ndarray',
            'type': 'float32 or float64 as same as',
            'description': '1xN/Nx1 2-channel'
        },
        'angleInDegrees': {
            'type': 'bool',
            'default': 'False',
            'description': 'A flag indicating whether the input angles are measured in degrees. When True, angles are in degrees; otherwise, they are in radians.'
        },
        'output': {
            'number': '2',
            'description': 'The function returns a positive value if the point is inside the contour, a negative value if it is outside, and zero if it is on the contour edge. When measureDist is True, the return value is the signed distance to the nearest contour edge.'
        }
    }
    return info

def get_pyrMeanShiftFiltering_params_info():
    info = {
        'src': {
            'format': 'numpy.ndarray',
            'type': 'uint8',
            'description': 'The source image, which must be an 8-bit 3-channel image.'
        },
        'sp': {
            'type': 'float',
            'description': 'The spatial window radius that defines the size of the neighborhood area used for the mean shift filter.'
        },
        'sr': {
            'type': 'float',
            'description': 'The color window radius that defines how far in color space the meanshift filtering algorithm searches for pixels to include in the mean calculation.'
        },
        'maxLevel': {
            'type': 'int',
            'default': '0',
            'flag': 'None',
            'description': 'Maximum level of the pyramid for the segmentation. When set to a value greater than 0, a Gaussian pyramid is built, and the filtering is performed first on the smallest layer.'
        },
        'output': {
            'number': '1',
            'description': 'The function returns a positive value if the point is inside the contour, a negative value if it is outside, and zero if it is on the contour edge. When measureDist is True, the return value is the signed distance to the nearest contour edge.'
        }
    }
    return info

def get_rectangleIntersectionArea_params_info():
    info = {
        'imgRect1': {
            'format': 'tuple',
            'type': 'int',
            'description': 'The rectangle representing the image boundaries. It should be specified as a tuple (x, y, width, height), where (x, y) is the top-left corner, and width and height are the dimensions of the rectangle.'
        },
        'imgRect2': {
            'format': 'tuple',
            'type': 'int',
            'description': 'The rectangle representing the image boundaries. It should be specified as a tuple (x, y, width, height), where (x, y) is the top-left corner, and width and height are the dimensions of the rectangle.'
        },
        'output': {
            'number': '1',
            'description': 'Returns a boolean value indicating whether the line segment is inside the image rectangle. If true, it also returns the clipped line points within the image rectangle.'
        }
    }
    return info

def get_rectify3Collinear_params_info():#bug
    info = {
        'cameraMatrix1': {
            'format': 'numpy.ndarray',
            'type': 'float64',
            'description': 'matrix'
        },
        'distCoeffs1': {
            'format': 'numpy.ndarray',
            'type': 'float64',
            'flag': 'None',
            'description': ''
        },
        'cameraMatrix2': {
            'format': 'numpy.ndarray',
            'type': 'float64',
            'description': 'matrix'
        },
        'distCoeffs2': {
            'format': 'numpy.ndarray',
            'type': 'float64',
            'flag': 'None',
            'description': ''
        },
        'cameraMatrix3': {
            'format': 'numpy.ndarray',
            'type': 'float64',
            'description': 'matrix'
        },
        'distCoeffs3': {
            'format': 'numpy.ndarray',
            'type': 'float64',
            'flag': 'None',
            'description': ''
        },
        'imgpt1': {
            'format': 'numpy.ndarray',
            'type': 'float64',
            'description': '2x3'
        },
        'imgpt3': {
            'format': 'numpy.ndarray',
            'type': 'float64',
            'description': '2x3'
        },
        'imageSize': {
            'format': 'tuple',
            'type': 'int',
            'description': 'Size of the image used for rectification.'
        },
        'R12': {
            'format': 'numpy.ndarray',
            'type': 'float64',
            'description': 'Rotation matrix between the first and the second camera coordinate systems.'
        },
        'T12': {
            'format': 'numpy.ndarray',
            'type': 'float64',
            'description': '3x1'
        },
        'R13': {
            'format': 'numpy.ndarray',
            'type': 'float64',
            'description': ' 3x3 matrix '
        },
        'T13': {
            'format': 'numpy.ndarray',
            'type': 'float64',
            'description': ' 3x1 '
        },
        'alpha': {
            'type': 'float',
            'description': 'Free scaling parameter.'
        },
        'newImgSize': {
            'format': 'tuple',
            'type': 'int',
            'description': 'Size of the image after rectification.'
        },
        'flags': {
            'type': 'int',
            'default': '0',
            'description': 'Operation flags. Pass cv2.SVD_FULL_UV to compute full-size U and V matrices, or cv2.SVD_MODIFY_A for an optimized computation if A is not needed after the operation. This is an optional parameter.'
        },
        'output': {
            'number': '8',
            'description': 'Outputs include rotation matrices (R1, R2, R3), projection matrices (P1, P2, P3), a 4x4 disparity-to-depth mapping matrix (Q), and valid regions of interest (roi1, roi2) for the first and second cameras.'
        }
    }
    return info

def get_reduceArgMax_params_info():
    info = {
        'src': {
            'format': 'numpy.ndarray',
            'type': 'uint8',
            'description': 'Input 8-bit single-channel array.'
        },
        'axis': {
            'type': 'int',
            'default':'0',
            'flag' : 'None',
            'description': 'Axis along which to reduce. For a 2D array, 0 means reducing rows and 1 means reducing columns.'
        },
        'output': {
            'number': '1',
            'description': 'Outputs include rotation matrices (R1, R2, R3), projection matrices (P1, P2, P3), a 4x4 disparity-to-depth mapping matrix (Q), and valid regions of interest (roi1, roi2) for the first and second cameras.'
        }
    }
    return info

def get_reduceArgMin_params_info():
    info = {
        'src': {
            'format': 'numpy.ndarray',
            'type': 'uint8',
            'description': 'Input 8-bit single-channel array.'
        },
        'axis': {
            'type': 'int',
            'default':'0',
            'flag' : 'None',
            'description': 'Axis along which to reduce. For a 2D array, 0 means reducing rows and 1 means reducing columns.'
        },
        'output': {
            'number': '1',
            'description': 'Outputs include rotation matrices (R1, R2, R3), projection matrices (P1, P2, P3), a 4x4 disparity-to-depth mapping matrix (Q), and valid regions of interest (roi1, roi2) for the first and second cameras.'
        }
    }
    return info

def get_remap_params_info():
    info = {
        'src': {
            'format': 'numpy.ndarray',
            'type': 'float32',
            'description': 'Input 8-bit single-channel array.'
        },
        'map1': {
            'format': 'numpy.ndarray',
            'type': 'float32',
            'description': 'The first map of either (x,y) points or just x values. Type can be CV_16SC2, CV_32FC1, or CV_32FC2. Refer to #convertMaps for details on converting a floating point representation to fixed-point for speed.'
        },
        'map2': {
            'format': 'numpy.ndarray',
            'type': 'float32',
            'description': 'The second map of y values having the type CV_16UC1, CV_32FC1, or none (empty map if map1 is (x,y) points), respectively.'
        },
        'interpolation': {
            'type': 'int',
            'default': 'cv2.INTER_AREA',
            'flag': 'None',
            'description': 'Interpolation method. INTER_AREA and INTER_LINEAR_EXACT are not supported.'
        },
        'borderType': {
            'type': 'int',
            'default': 'cv2.BORDER_CONSTANT',
            'description': 'Pixel extrapolation method. When BORDER_TRANSPARENT, it means that the pixels in the destination image corresponding to the "outliers" in the source image are not modified by the function.'
        },
        'borderValue': {
            'type': 'Scalar',
            'default': '0',
            'flag': 'None',
            'description': 'Value used in case of a constant border. By default, it is 0.'
        },
        'output': {
            'number': '1',
            'description': 'Outputs include rotation matrices (R1, R2, R3), projection matrices (P1, P2, P3), a 4x4 disparity-to-depth mapping matrix (Q), and valid regions of interest (roi1, roi2) for the first and second cameras.'
        }
    }
    return info

def get_reprojectImageTo3D_params_info():
    info = {
        'disparity': {
            'format': 'numpy.ndarray',
            'type': 'uint8 or float32',
            'description': 'Input single-channel disparity image. Values of 8-bit / 16-bit signed formats are assumed to have no fractional bits. If the disparity is 16-bit signed format, it should be divided by 16 and scaled to float before being used.'
        },
        'maxtr': {
            'format': 'numpy.ndarray',
            'type': 'float32',
            'description': '4x4 matrix obtained with stereoRectify.'
        },
        'handleMissingValues': {
            'type': 'bool',
            'default': 'false',
            'description': 'Indicates whether to handle missing values, i.e., points where the disparity was not computed. If true, pixels with the minimal disparity that corresponds to outliers are transformed to 3D points with a large Z value.'
        },
        'ddepth': {
            'type': 'int',
            'default': '-1',
            'flag':'None',
            'description': 'Optional output array depth. -1 for CV_32F, can also be CV_16S, CV_32S, CV_32F.'
        },
        'output': {
            'number': '1',
            'description': 'Outputs include rotation matrices (R1, R2, R3), projection matrices (P1, P2, P3), a 4x4 disparity-to-depth mapping matrix (Q), and valid regions of interest (roi1, roi2) for the first and second cameras.'
        }
    }
    return info

def get_undistortPointsIter_params_info():
    info = {
        'src': {
            'format': 'numpy.ndarray',
            'type': 'float32',
            'description': 'Input vector of 2D points (with single-channel two-column matrix, or two-channel 1-column matrix, or std::vector<Point2f>).'
        },
        'cameraMatrix': {
            'format': 'numpy.ndarray',
            'type': 'float32',
            'description': 'Camera matrix f$A=ecthreethree{fx}{0}{cx}{0}{fy}{cy}{0}{0}{1}$.'
        },
        'distCoeffs': {
            'format': 'numpy.ndarray',
            'type': 'float32',
            'flag': 'None',
            'description': 'Input of vector distortion coefficients. If empty, zero coefficients are assumed.',
        },
        'R': {
            'format': 'numpy.ndarray',
            'type': 'float32',
            'description': 'Rectification transformation in the object space (3x3 matrix). Optional.'
        },
        'P': {
            'format': 'numpy.ndarray',
            'type': 'float32',
            'description': 'New camera matrix (3x3) or new projection matrix (3x4). Optional.'
        },
        'criteria': {
            'format': 'tuple',
            'type': 'int',
            'default': '(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 0.001)',
            'flag': 'None',
            'description': 'Criteria for termination of the iterative process of corner refinement. The process stops either after criteria.maxCount iterations or when the corner position moves by less than criteria.epsilon on some iteration.'
        },
        'output': {
            'number': '1',
            'description': 'Outputs include rotation matrices (R1, R2, R3), projection matrices (P1, P2, P3), a 4x4 disparity-to-depth mapping matrix (Q), and valid regions of interest (roi1, roi2) for the first and second cameras.'
        }
    }
    return info

def get_validateDisparity_params_info():#crash
    info = {
        'disparity': {
            'format': 'numpy.ndarray',
            'type': 'int16',
            'description': 'Input disparity map. It is a 8-bit single-channel image, where each pixel corresponds to the disparity value at the respective location.'
        },
        'cost': {
            'format': 'numpy.ndarray',
            'type': 'int16 as same as ',
            'description': '8-bit single-channel image Input cost (or confidence) for each disparity value. It should have the same size as the disparity map. Each element represents the matching cost at the corresponding location.'
        },
        'minDisparity': {
            'type': 'int',
            'description': 'Minimum possible disparity value. Normally, it is zero but sometimes rectification algorithms can shift images, so this parameter needs to be adjusted accordingly.'
        },
        'numberOfDisparities': {
            'type': 'int',
            'description': 'Maximum disparity minus minimum disparity. The range of disparities to be searched. Larger values may improve accuracy but will also increase computation time.'
        },
        'disp12MaxDisp': {
            'type': 'int',
            'default': '1',
            'flag': 'None',
            'description': 'Maximum allowed difference (in integer pixel units) in the left-right disparity check. Set it to a non-positive value to disable the check.'
        },
        'output': {
            'number': '1',
            'description': 'The output disparity map with invalidated disparities set to the minDisparity-1.'
        }
    }
    return info

def get_undistortPoints_params_info():
    info = {
        'src': {
            'format': 'numpy.ndarray',
            'type': 'float32',
            'description': 'Input vector of 2D points (with single-channel two-column matrix, or two-channel 1-column matrix, or std::vector<Point2f>).'
        },
        'cameraMatrix': {
            'format': 'numpy.ndarray',
            'type': 'float32',
            'description': 'Camera matrix f$A=ecthreethree{fx}{0}{cx}{0}{fy}{cy}{0}{0}{1}$.'
        },
        'distCoeffs': {
            'format': 'numpy.ndarray',
            'type': 'float32',
            'flag': 'None',
            'description': 'Input of vector distortion coefficients. If empty, zero coefficients are assumed.',
        },
        'R': {
            'format': 'numpy.ndarray',
            'type': 'float32',
            'description': 'Rectification transformation in the object space (3x3 matrix). Optional.'
        },
        'P': {
            'format': 'numpy.ndarray',
            'type': 'float32',
            'description': 'New camera matrix (3x3) or new projection matrix (3x4). Optional.'
        },
        'output': {
            'number': '1',
            'description': 'Outputs include rotation matrices (R1, R2, R3), projection matrices (P1, P2, P3), a 4x4 disparity-to-depth mapping matrix (Q), and valid regions of interest (roi1, roi2) for the first and second cameras.'
        }
    }
    return info

def get_undistortImagePoints_params_info():
    info = {
        'src': {
            'format': 'numpy.ndarray',
            'type': 'float32',
            'description': 'Input vector of 2D points (with single-channel two-column matrix, or two-channel 1-column matrix, or std::vector<Point2f>).'
        },
        'cameraMatrix': {
            'format': 'numpy.ndarray',
            'type': 'float32',
            'description': 'Camera matrix f$A=ecthreethree{fx}{0}{cx}{0}{fy}{cy}{0}{0}{1}$.'
        },
        'distCoeffs': {
            'format': 'numpy.ndarray',
            'type': 'float32',
            'flag': 'None',
            'description': 'Input of vector distortion coefficients. If empty, zero coefficients are assumed.',
        },
        'output': {
            'number': '1',
            'description': 'Outputs include rotation matrices (R1, R2, R3), projection matrices (P1, P2, P3), a 4x4 disparity-to-depth mapping matrix (Q), and valid regions of interest (roi1, roi2) for the first and second cameras.'
        }
    }
    return info

def get_undistort_params_info():
    info = {
        'src': {
            'format': 'numpy.ndarray',
            'type': 'float32',
            'description': 'Input vector of 2D points (with single-channel two-column matrix, or two-channel 1-column matrix, or std::vector<Point2f>).'
        },
        'cameraMatrix': {
            'format': 'numpy.ndarray',
            'type': 'float32',
            'description': 'Camera matrix f$A=ecthreethree{fx}{0}{cx}{0}{fy}{cy}{0}{0}{1}$.'
        },
        'distCoeffs': {
            'format': 'numpy.ndarray',
            'type': 'float32',
            'flag': 'None',
            'description': 'Input of vector distortion coefficients. If empty, zero coefficients are assumed.',
        },
        'newCameraMatrix': {
            'format': 'numpy.ndarray',
            'type': 'float32',
            'description': 'New camera matrix'
        },
        'output': {
            'number': '1',
            'description': 'Outputs include rotation matrices (R1, R2, R3), projection matrices (P1, P2, P3), a 4x4 disparity-to-depth mapping matrix (Q), and valid regions of interest (roi1, roi2) for the first and second cameras.'
        }
    }
    return info

def get_getAffineTransform_params_info():
    info = {
        'src': {
            'format': 'numpy.ndarray',
            'type': 'float32',
            'description': '2x3 Coordinates of triangle vertices in the source image. It is an array of points or a matrix of shape (3,2) containing the coordinates of the vertices of the triangle in the source image.'
        },
        'dst': {
            'format': 'numpy.ndarray',
            'type': 'float32',
            'description': '2x3 Coordinates of the corresponding triangle vertices in the destination image. It is an array of points or a matrix of shape (3,2) containing the coordinates of the vertices of the triangle in the destination image.'
        },
        'output': {
            'number': '1',
            'description': 'The function returns a 2x3 affine transformation matrix that can be used to transform the coordinates of the vertices of the triangle in the source image to get the corresponding vertices in the destination image.'
        }
    }
    return info

def get_getDefaultNewCameraMatrix_params_info():
    info = {
        'cameraMatrix': {
            'format': 'numpy.ndarray',
            'type': 'float64',
            'description': 'Input camera matrix. It is a 3x3 matrix that represents the intrinsic camera parameters.'
        },
        'imgsize': {
            'format': 'tuple',
            'type': 'int',
            'description': 'Camera view image size in pixels. It is a tuple of two integers representing the width and height of the image.'
        },
        'centerPrincipalPoint': {
            'type': 'bool',
            'default': 'False',
            'description': 'Indicates whether the principal point should be moved to the center of the image or not.'
        },
        'output': {
            'number': '1',
            'description': 'The function returns a new camera matrix. If centerPrincipalPoint is False, it returns an exact copy of the input cameraMatrix. If True, it returns a modified camera matrix where the principal point is at the center of the image.'
        }
    }
    return info

def get_getDerivKernels_params_info():
    info = {
        'dx': {
            'type': 'int',
            'description': 'Derivative order in respect of x. Specifies how many times the image is differentiated in the horizontal direction.'
        },
        'dy': {
            'type': 'int',
            'description': 'Derivative order in respect of y. Specifies how many times the image is differentiated in the vertical direction.'
        },
        'ksize': {
            'type': 'int',
            'description': 'Aperture size. It can be FILTER_SCHARR (=-1), 1, 3, 5, or 7. Specifies the size of the filter.'
        },
        'normalize': {
            'type': 'bool',
            'default': 'False',
            'description': 'Flag indicating whether to normalize (scale down) the filter coefficients or not. Normalization is done to preserve the fractional bits when processing.'
        },
        'output': {
            'number': '2',
            'description': 'The function returns two matrices (kx, ky) containing the derivatives kernels for the x and y directions, respectively.'
        }
    }
    return info

def get_getFontScaleFromHeight_params_info():
    info = {
        'fontFace': {
            'type': 'int',
            'description': 'Font to use. The available options are defined by the cv::HersheyFonts enumeration. This parameter specifies the typeface of the font.'
        },
        'pixelHeight': {
            'type': 'int',
            'description': 'Pixel height to compute the fontScale for. This is the desired height of the text in pixels.'
        },
        'thickness': {
            'type': 'int',
            'default': '1',
            'description': 'Thickness of lines used to render the text. This parameter affects the weight of the font. A higher value means a thicker font. See cv::putText for details.'
        },
        'output': {
            'number': '1',
            'description': 'The fontSize (fontScale) to use with cv::putText to achieve the specified pixel height of the text.'
        }
    }
    return info


def get_getGaborKernel_params_info():
    info = {
        'ksize': {
            'format': 'tuple',
            'type': 'int',
            'description': 'Size of the filter to be returned. Represents the width and height of the Gabor kernel.'
        },
        'sigma': {
            'type': 'float',
            'description': 'Standard deviation of the Gaussian envelope. This parameter controls the spread of the envelope.'
        },
        'theta': {
            'type': 'float',
            'description': 'Orientation of the normal to the parallel stripes of the Gabor function. Measured in radians.'
        },
        'lambd': {
            'type': 'float',
            'description': 'Wavelength of the sinusoidal factor. Specifies the wavelength of the sinusoidal wave in the Gabor function.'
        },
        'gamma': {
            'type': 'float',
            'description': 'Spatial aspect ratio. Specifies the ellipticity of the support of the Gabor function.'
        },
        'psi': {
            'type': 'float',
            'flag': 'None',
            'description': 'Phase offset. Specifies the phase offset of the sinusoidal factor in the Gabor function.'
        },
        'ktype': {
            'type': 'int',
            'default': 'cv2.CV_64F',
            'flag': 'None',
            'description': 'Phase offset. Specifies the phase offset of the sinusoidal factor in the Gabor function.'
        },
        'output': {
            'number': '1',
            'description': 'The resulting Gabor kernel as a Mat object.'
        }
    }
    return info

def get_getGaussianKernel_params_info():
    info = {
        'ksize': {
            'type': 'int',
            'description': 'Aperture size. It should be odd (ksize % 2 == 1) and positive.'
        },
        'sigma': {
            'type': 'float',
            'description': 'Gaussian standard deviation. If it is non-positive, it is computed from ksize as sigma = 0.3*((ksize-1)*0.5 - 1) + 0.8.'
        },
        'ktype': {
            'type': 'int',
            'default': 'cv2.CV_32F',
            'flag':'None',
            'description': 'Type of filter coefficients. It can be CV_32F or CV_64F.'
        },
        'output': {
            'number': '1',
            'description': 'The resulting \$\exttt{ksize} \imes 1\$ matrix of Gaussian filter coefficients.'
        }
    }
    return info

def get_getHardwareFeatureName_params_info():
    info = {
        'feature': {
            'type': 'int',
            'description': 'The ID of the hardware feature.'
        },
        'output': {
            'number': '1',
            'description': 'The name of the hardware feature if it is defined, otherwise an empty string.'
        }
    }
    return info

def get_getOptimalDFTSize_params_info():
    info = {
        'vecsize': {
            'type': 'int',
            'description': 'The size of the vector for which to find the optimal DFT size.'
        },
        'output': {
            'number': '1',
            'description': 'The minimum number greater than or equal to vecsize that can be transformed efficiently.'
        }
    }
    return info

def get_getOptimalNewCameraMatrix_params_info():
    info = {
        'cameraMatrix': {
            'format': 'numpy.ndarray',
            'type': 'int',
            'description': 'Input camera intrinsic matrix.'
        },
        'distCoeffs': {
            'format': 'None',
            'type': 'int',
            'description': 'Input vector of distortion coefficients. If NULL/empty, zero distortion coefficients are assumed.'
        },
        'imageSize': {
            'format': 'tuple',
            'type': 'int',
            'description': 'Original image size.'
        },
        'alpha': {
            'type': 'float',
            'description': 'Free scaling parameter between 0 and 1. Controls the balance between all pixels in the undistorted image being valid (0) and all source image pixels being retained (1).'
        },
        'output': {
            'number': '2',
            'description': 'Returns a new camera intrinsic matrix and an optional output rectangle that outlines the all-good-pixels region in the undistorted image.'
        }
    }
    return info

def get_getPerspectiveTransform_params_info():
    info = {
        'src': {
            'format': 'numpy.ndarray',
            'type': 'float32',
            'description': '2x4 matrix Coordinates of quadrangle vertices in the source image.'
        },
        'dst': {
            'format': 'numpy.ndarray',
            'type': 'float32',
            'description': '2x4 matrix Coordinates of the corresponding quadrangle vertices in the destination image.'
        },
        'solveMethod': {
            'type': 'int',
            'default': 'cv2.DECOMP_LU',
            'flag':'None',
            'description': 'Method passed to cv::solve. It can be one of cv2.DECOMP_LU, cv2.DECOMP_SVD, etc. Default is cv2.DECOMP_LU.'
        },
        'output': {
            'number': '1',
            'description': 'The 3x3 perspective transformation matrix.'
        }
    }
    return info

def get_getRectSubPix_params_info():
    info = {
        'image': {
            'format': 'numpy.ndarray',
            'type': 'float32',
            'description': 'Source image.'
        },
        'patchSize': {
            'format': 'tuple',
            'type': 'int',
            'description': 'Size of the extracted patch.'
        },
        'center': {
            'format': 'tuple',
            'type': 'int',
            'description': 'Floating point coordinates of the center of the extracted rectangle within the source image. The center must be inside the image.'
        },
        'output': {
            'number': '1',
            'description': 'The extracted patch with sub-pixel accuracy.'
        }
    }
    return info

def get_getRotationMatrix2D_params_info():
    info = {
        'center': {
            'format': 'tuple',
            'type': 'int',
            'description': 'Center of the rotation in the source image.'
        },
        'angle': {
            'type': 'float',
            'description': 'Rotation angle in degrees. Positive values mean counter-clockwise rotation (the coordinate origin is assumed to be the top-left corner).'
        },
        'scale': {
            'type': 'float',
            'description': 'Isotropic scale factor.'
        },
        'output': {
            'number': '1',
            'description': 'The 2x3 affine transformation matrix for the rotation.'
        }
    }
    return info

def get_getStructuringElement_params_info():
    info = {
        'shape': {
            'type': 'int',
            'default': 'cv2.MORPH_RECT',
            'flag': 'None',
            'description': 'Element shape that could be one of the predefined shapes: MORPH_RECT, MORPH_CROSS, or MORPH_ELLIPSE.'
        },
        'ksize': {
            'format': 'tuple',
            'type': 'int',
            'description': 'Size of the structuring element.'
        },
        'anchor': {
            'format': 'tuple',
            'type': 'int',
            'default': '(-1, -1)',
            'flag': 'None',
            'description': 'Anchor position within the element. Default value (-1, -1) means that the anchor is at the center. Note that only the shape of a cross-shaped element depends on the anchor position. In other cases, the anchor affects the shift of the morphological operation result.'
        },
        'output': {
            'number': '1',
            'description': 'The structuring element used for morphological operations.'
        }
    }
    return info

def get_getTextSize_params_info():
    info = {
        'text': {
            'type': 'str',
            'description': 'Input text string to be measured.'
        },
        'fontFace': {
            'type': 'int',
            'default': 'cv2.FONT_HERSHEY_SIMPLEX',
            'description': 'Font type to use. See #HersheyFonts for possible values.'
        },
        'fontScale': {
            'type': 'int',
            'default': '2',
            'flag': 'None',
            'description': 'Font scale factor that is multiplied by the font-specific base size.'
        },
        'thickness': {
            'type': 'int',
            'default': '1',
            'flag': 'None',
            'description': 'Thickness of the lines used to draw the text.'
        },
        'output': {
            'number': '2',
            'description': 'The first output, retval, is the size of a box that contains the specified text. The second output, baseLine, is the y-coordinate of the baseline relative to the bottom-most text point.'
        }
    }
    return info

def get_getTrackbarPos_params_info():#unspport
    info = {
        'trackbarname': {
            'type': 'str',
            'description': 'The name of the trackbar whose position you want to retrieve.'
        },
        'winname': {
            'type': 'str',
            'description': 'The name of the window that contains the trackbar. Note that for Qt backend, this can be empty if the trackbar is attached to the control panel.'
        },
        'output': {
            'number': '1',
            'description': 'The current position of the specified trackbar as an integer.'
        }
    }
    return info

def get_waitKeyEx_params_info():
    info = {
        'delay': {
            'type': 'int',
            'default': '0',
            'description': 'Delay in milliseconds. 0 is the special value that means “forever”.'
        },
        'output': {
            'number': '1',
            'description': 'The full key code of the pressed key. The key code is implementation specific and depends on the backend used by OpenCV (QT, GTK, Win32, etc.). If no key is pressed, the function returns -1.'
        }
    }
    return info

def get_getNumThreads_params_info():
    info = {
        'output': {
            'number': '1',
            'description': 'The number of threads used by OpenCV for parallel regions. The interpretation of the return value depends on the threading framework used by the OpenCV build.'
        },
    }
    return info

def get_getNumberOfCPUs_params_info():
    info = {
        'output': {
            'number': '1',
            'description': 'The number of threads used by OpenCV for parallel regions. The interpretation of the return value depends on the threading framework used by the OpenCV build.'
        },
    }
    return info

def get_FlannBasedMatcher_create_params_info():
    info = {
        'output': {
            'number': '1',
            'description': 'The number of threads used by OpenCV for parallel regions. The interpretation of the return value depends on the threading framework used by the OpenCV build.'
        },
    }
    return info

def get_HOGDescriptor_getDaimlerPeopleDetector_params_info():
    info = {
        'output': {
            'number': '1',
            'description': 'The number of threads used by OpenCV for parallel regions. The interpretation of the return value depends on the threading framework used by the OpenCV build.'
        },
    }
    return info

def get_HOGDescriptor_getDefaultPeopleDetector_params_info():
    info = {
        'output': {
            'number': '1',
            'description': 'The number of threads used by OpenCV for parallel regions. The interpretation of the return value depends on the threading framework used by the OpenCV build.'
        },
    }
    return info

def get_UMat_context_params_info():
    info = {
        'output': {
            'number': '1',
            'description': 'The number of threads used by OpenCV for parallel regions. The interpretation of the return value depends on the threading framework used by the OpenCV build.'
        },
    }
    return info

def get_UMat_queue_params_info():
    info = {
        'output': {
            'number': '1',
            'description': 'The number of threads used by OpenCV for parallel regions. The interpretation of the return value depends on the threading framework used by the OpenCV build.'
        },
    }
    return info

def get_createGeneralizedHoughBallard_params_info():
    info = {
        'output': {
            'number': '1',
            'description': 'The number of threads used by OpenCV for parallel regions. The interpretation of the return value depends on the threading framework used by the OpenCV build.'
        },
    }
    return info

def get_createGeneralizedHoughGuil_params_info():
    info = {
        'output': {
            'number': '1',
            'description': 'The number of threads used by OpenCV for parallel regions. The interpretation of the return value depends on the threading framework used by the OpenCV build.'
        },
    }
    return info

def get_createMergeDebevec_params_info():
    info = {
        'output': {
            'number': '1',
            'description': 'The number of threads used by OpenCV for parallel regions. The interpretation of the return value depends on the threading framework used by the OpenCV build.'
        },
    }
    return info

def get_createMergeRobertson_params_info():
    info = {
        'output': {
            'number': '1',
            'description': 'The number of threads used by OpenCV for parallel regions. The interpretation of the return value depends on the threading framework used by the OpenCV build.'
        },
    }
    return info

def get_empty_array_desc_params_info():
    info = {
        'output': {
            'number': '1',
            'description': 'The number of threads used by OpenCV for parallel regions. The interpretation of the return value depends on the threading framework used by the OpenCV build.'
        },
    }
    return info

def get_empty_gopaque_desc_params_info():
    info = {
        'output': {
            'number': '1',
            'description': 'The number of threads used by OpenCV for parallel regions. The interpretation of the return value depends on the threading framework used by the OpenCV build.'
        },
    }
    return info

def get_createMergeDebevec_params_info():
    info = {
        'output': {
            'number': '1',
            'description': 'The number of threads used by OpenCV for parallel regions. The interpretation of the return value depends on the threading framework used by the OpenCV build.'
        },
    }
    return info

def get_empty_scalar_desc_params_info():
    info = {
        'output': {
            'number': '1',
            'description': 'The number of threads used by OpenCV for parallel regions. The interpretation of the return value depends on the threading framework used by the OpenCV build.'
        },
    }
    return info

def get_getBuildInformation_params_info():
    info = {
        'output': {
            'number': '1',
            'description': 'The number of threads used by OpenCV for parallel regions. The interpretation of the return value depends on the threading framework used by the OpenCV build.'
        },
    }
    return info

def get_getCPUTickCount_params_info():
    info = {
        'output': {
            'number': '1',
            'description': 'The number of threads used by OpenCV for parallel regions. The interpretation of the return value depends on the threading framework used by the OpenCV build.'
        },
    }
    return info

def get_getCPUFeaturesLine_params_info():
    info = {
        'output': {
            'number': '1',
            'description': 'The number of threads used by OpenCV for parallel regions. The interpretation of the return value depends on the threading framework used by the OpenCV build.'
        },
    }
    return info

def get_getLogLevel_params_info():
    info = {
        'output': {
            'number': '1',
            'description': 'The number of threads used by OpenCV for parallel regions. The interpretation of the return value depends on the threading framework used by the OpenCV build.'
        },
    }
    return info

def get_getThreadNum_params_info():
    info = {
        'output': {
            'number': '1',
            'description': 'The number of threads used by OpenCV for parallel regions. The interpretation of the return value depends on the threading framework used by the OpenCV build.'
        },
    }
    return info

def get_getTickCount_params_info():
    info = {
        'output': {
            'number': '1',
            'description': 'The number of threads used by OpenCV for parallel regions. The interpretation of the return value depends on the threading framework used by the OpenCV build.'
        },
    }
    return info

def get_getTickFrequency_params_info():
    info = {
        'output': {
            'number': '1',
            'description': 'The number of threads used by OpenCV for parallel regions. The interpretation of the return value depends on the threading framework used by the OpenCV build.'
        },
    }
    return info

def get_getVersionMajor_params_info():
    info = {
        'output': {
            'number': '1',
            'description': 'The number of threads used by OpenCV for parallel regions. The interpretation of the return value depends on the threading framework used by the OpenCV build.'
        },
    }
    return info

def get_getVersionMinor_params_info():
    info = {
        'output': {
            'number': '1',
            'description': 'The number of threads used by OpenCV for parallel regions. The interpretation of the return value depends on the threading framework used by the OpenCV build.'
        },
    }
    return info

def get_getVersionRevision_params_info():
    info = {
        'output': {
            'number': '1',
            'description': 'The number of threads used by OpenCV for parallel regions. The interpretation of the return value depends on the threading framework used by the OpenCV build.'
        },
    }
    return info

def get_getVersionString_params_info():
    info = {
        'output': {
            'number': '1',
            'description': 'The number of threads used by OpenCV for parallel regions. The interpretation of the return value depends on the threading framework used by the OpenCV build.'
        },
    }
    return info

def get_haveOpenVX_params_info():
    info = {
        'output': {
            'number': '1',
            'description': 'The number of threads used by OpenCV for parallel regions. The interpretation of the return value depends on the threading framework used by the OpenCV build.'
        },
    }
    return info

def get_pollKey_params_info():
    info = {
        'output': {
            'number': '1',
            'description': 'The number of threads used by OpenCV for parallel regions. The interpretation of the return value depends on the threading framework used by the OpenCV build.'
        },
    }
    return info


def get_startWindowThread_params_info():
    info = {
        'output': {
            'number': '1',
            'description': 'The number of threads used by OpenCV for parallel regions. The interpretation of the return value depends on the threading framework used by the OpenCV build.'
        },
    }
    return info

def get_ariationalRefinement_create_params_info():
    info = {
        'output': {
            'number': '1',
            'description': 'The number of threads used by OpenCV for parallel regions. The interpretation of the return value depends on the threading framework used by the OpenCV build.'
        },
    }
    return info

def get_useOptimized_params_info():
    info = {
        'output': {
            'number': '1',
            'description': 'The number of threads used by OpenCV for parallel regions. The interpretation of the return value depends on the threading framework used by the OpenCV build.'
        },
    }
    return info

def get_VariationalRefinement_create_params_info():
    info = {
        'output': {
            'number': '1',
            'description': 'The number of threads used by OpenCV for parallel regions. The interpretation of the return value depends on the threading framework used by the OpenCV build.'
        },
    }
    return info

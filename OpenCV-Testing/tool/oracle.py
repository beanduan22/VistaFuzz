import numpy as np
'''
def test_param_success(mutated_arg, func, mutation_history,n):
    try:
        result = func(*mutated_arg)

        if result is None:
            mutation_history.log_error(func.__name__, mutated_arg, "None Value",n)
            return False

        if isinstance(result, np.ndarray):
            if np.any(np.isnan(result)):
                error_type = "NaN Value" if np.any(np.isnan(result)) else "Inf Value"
                mutation_history.log_error(func.__name__,mutated_arg, error_type,n)
                return False

        elif isinstance(result, (float, int)):
            if np.isnan(result) or np.isinf(result):
                error_type = "NaN Value" if np.isnan(result) else "Inf Value"
                mutation_history.log_error(func.__name__,mutated_arg, error_type,n)
                return False

        return True
    except Exception as e:
        mutation_history.log_error(func.__name__, mutated_arg, str(e),n)
        return False
    '''
def test_param_success(mutated_arg, func, mutation_history, n):
    try:
        result = func(*mutated_arg)

        # Handling for None, NaN, or Inf values in the result
        if (isinstance(result, (np.ndarray, float, int)) and not np.all(np.isfinite(result))):
            error_type = "NaN/Inf Value"
            mutation_history.log_error(func.__name__, mutated_arg, error_type, n)
            return False

        return True
    except Exception as e:
        mutation_history.log_error(func.__name__, mutated_arg, str(e), n)
        return False
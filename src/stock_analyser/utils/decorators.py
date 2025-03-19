import time
from functools import wraps
from datetime import datetime
from stock_analyser.utils.constants import TIMESTAMP


def timeit(func):
    @wraps(func)
    def timeit_wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        total_time = end_time - start_time
        print(f'⏱️ Function {func.__name__} took {total_time:.4f} seconds to complete\n')
        with open(f"/home/j/ai/crewAI/finance/stock_analyser/timings/timeit_{TIMESTAMP}.txt", "a") as f:
            f.write(f'{datetime.now().strftime("%Y-%m-%d %H:%M:%S")} - Function {func.__name__}({args = }) ({kwargs = }) took {total_time:.4f} seconds\n')
        return result
    return timeit_wrapper


# def track_token_usage(func):
#     @wraps(func)
#     def track_token_usage_wrapper(self, *args, **kwargs):
#         result = func(self, *args, **kwargs)
#         # Check if result exists and has usage metrics
#         token_count = result.token_usage
            
#         print(f"🪙 Number of tokens used: {token_count:,}")
#         with open(f"/home/j/ai/crewAI/finance/stock_analyser/timings/timeit_{TIMESTAMP}.txt", "a") as f:
#             f.write(f"{func.__name__} used {token_count:,} tokens\n")
#         self.state.total_token_usage += token_count
#         return result
#     return track_token_usage_wrapper


# def track_token_usage(token_count: int):
#     print(f"🪙 Number of tokens used: {token_count:,}")
#     with open(f"/home/j/ai/crewAI/finance/stock_analyser/timings/timeit_{TIMESTAMP}.txt", "a") as f:
#         f.write(f"{func.__name__} used {token_count:,} tokens\n")
#     self.state.total_token_usage += token_count
#     return result
#     return wrapper
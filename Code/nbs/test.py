import time
import math
from multiprocessing import Pool

# CPU-heavy function
def compute_heavy_task(n):
    total = 0
    for i in range(10000000):
        total += math.sqrt(i % (n + 1))
    return total

def run_serial():
    start = time.time()
    results = list(map(compute_heavy_task, range(5)))
    end = time.time()
    print("Serial time:", end - start)
    return results

def run_parallel():
    start = time.time()
    with Pool() as pool:
        results = pool.map(compute_heavy_task, range(5))
        end = time.time()
        print("Parallel time:", end - start)
        return results

if __name__ == '__main__':
    print("Running serial version:")
    run_serial()

    print("\nRunning parallel version:")
    run_parallel()

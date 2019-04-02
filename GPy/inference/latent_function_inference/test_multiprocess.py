from functools import partial
from itertools import repeat
from multiprocessing import Pool, freeze_support

def func(a, b):
    return a + b, a-b

if __name__=="__main__":
    freeze_support()
    a_args = [1,2,3,4,5]
    second_arg = 1
    with Pool(4) as pool:
        #L = pool.starmap(func, [(1, 1), (2, 1), (3, 1)])
        M = pool.starmap(func, zip(a_args, repeat(second_arg)), chunksize=4)
        #N = pool.map(partial(func, b=second_arg), a_args)
        #assert L == M == N

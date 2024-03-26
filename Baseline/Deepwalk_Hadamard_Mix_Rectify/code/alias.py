import numpy as np
def create_alias_table(area_ratio):
    """
    :param area_ratio: sum(area_ratio)=1
    :return: accept,alias
    """
    l = len(area_ratio)
    accept, alias = [0] * l, [0] * l
    small, large = [], []
    for i, prob in enumerate(area_ratio):
        accept[i]=l*prob
        if prob < 1.0:
            small.append(i)
        else:
            large.append(i)
    while len(small)>0 and len(large)>0:
        small_idx, large_idx = small.pop(), large.pop()
        alias[small_idx] = large_idx
        accept[large_idx] = accept[large_idx]+accept[small_idx]-1
        if accept[large_idx] < 1.0:
            small.append(large_idx)
        else:
            large.append(large_idx)
    # print(accept)
    return accept, alias

def alias_sample(accept, alias):
    """
    :param accept:
    :param alias:
    :return: sample index
    """
    N = len(accept)
    i = int(np.random.random()*N)
    r = np.random.random()
    if r < accept[i]:
        return i
    else:
        return alias[i]

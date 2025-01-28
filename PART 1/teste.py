from itertools import product
import numpy as np
import matplotlib.pyplot as plt

def distribuir_AP(M: int) -> np.ndarray:        # feito
    dx = 1000/(2*np.sqrt(M))        # passo
    a = np.arange(dx, 1001-dx, 2*dx)
    x, y = np.meshgrid(a, a)
    ap_coord = np.column_stack((x.ravel(), y.ravel()))
    return ap_coord

def dist_AP_UE(x_ue, y_ue, M) -> float:        # analisar
    ap_coord = distribuir_AP(M)
    for x in ap_coord:
        pass

if __name__ == '__main__':
    M = [1, 4, 9, 16, 25, 36, 49, 64]
    x_coord = np.random.randint(0, 1001)
    y_coord = np.random.randint(0, 1001)
    print(dist_AP_UE(4))

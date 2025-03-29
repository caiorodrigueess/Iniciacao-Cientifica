import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


class AP:
    def __init__(self, x: float, y: float, id: int):
        self.x = x
        self.y = y
        self.id = id
        self.channel = None
        self.ues = []

    def __str__(self):
        return f'AP[{self.id}]({self.x}, {self.y})'

class UE:
    def __init__(self, N: int):
        self.x = np.random.randint(0, 1001)
        self.y = np.random.randint(0, 1001)
        self.ap = None
        self.channel = 1
        self.dist = 0
        self.sinr = 0

    def __str__(self):
        return f'UE({self.x}, {self.y})'

def distribuir_AP(M: int) -> np.ndarray:
    APs = []
    dx = 1000/(2*np.sqrt(M))
    a = np.arange(dx, 1001-dx, 2*dx)
    x, y = np.meshgrid(a, a)
    id = 1
    for xi, yi in zip(x.ravel(), y.ravel()):
        APs.append(AP(xi, yi, id))
        id += 1
    return APs

def dist_AP_UE(M: int, ue: UE) -> float:
    aps = distribuir_AP(M)
    dist = 1e4
    for ap in aps:
        x = np.linalg.norm(np.array([ue.x, ue.y]) - np.array([ap.x, ap.y]))
        if x < dist:
            dist = x
            ue.ap = ap
            ap.channel = ue.channel
            ue.dist = dist
    return dist

# Função para calcular o pr dos UEs interferentes
def pr_interferente(ue1: UE, ue2: UE) -> float:
    if ue1.channel == ue2.channel:
        dist = np.linalg.norm(np.array([ue1.ap.x, ue1.ap.y]) - np.array([ue2.x, ue2.y]))
        return 1*1e-4/(dist**4)
    return 0

def SINR(ue: UE, ues: list ,N: int) -> float:
    pt=1        # transmited power
    k=1e-4      # constant
    n=4         # path gain
    bt=1e8      # avaiable bandwidth
    k0=1e-20    # constant for the noise power

    pn = k0*bt/N
    pr = pt*k/(ue.dist**n)
    pr_int = np.sum([pr_interferente(ue, u) for u in ues if u!=ue])
    
    return pr/(pn + pr_int)

def channel_capacity(SINR: float, N: int) -> float:
    bt=100      # avaiable bandwidth
    return np.around((bt/N)*np.log2(1+SINR), 4)

def simular_experimento(M: int, N: int, sim: int) -> tuple:
    sinr = []
    cap_canal = []
    ues = np.zeros(5, dtype=UE)
    break_flag = False          # Flag para interromper o laço externo
    ue_small = None             #UE para cenários extremos com valores baixos
    ue_large = None             #UE para cenários extremos com valores altos
    for _ in range(sim):
        for i in range(5):
            ue = UE(N)
            ue.dist = dist_AP_UE(M, ue)
            ues[i] = ue

        for ue in ues:
            s = SINR(ue, ues, N)
            cap = channel_capacity(s, N)
            sinr.append(s)
            cap_canal.append(cap)

            if s<1e-7:
                ue_small = ue
                break_flag = True
                break

            if s>1e20:
                ue_large = ue
                break_flag = True
                break
        
        if break_flag:
            break

    return sinr, ues, ue_small, ue_large

def plot_APs_UEs(M: int, ues: list, ue_small: UE, ue_large: UE) -> None:
    plt.figure(figsize=(10, 6))
    aps = distribuir_AP(M)

    for ap in aps:
        plt.scatter(ap.x, ap.y, marker='^', color='blue', label='AP' if ap == aps[0] else "")
    for ue in ues:
        plt.scatter(ue.x, ue.y, marker='s', color='red', label='UE')
        plt.plot([ue.ap.x, ue.x], [ue.ap.y, ue.y], linestyle='-', color='k', linewidth=0.25, )
    
    yellow_patch = mpatches.Patch(color='yellow', label='Very low SINR')
    green_patch = mpatches.Patch(color='green', label='Very large SINR')
        
    if ue_small is not None:
        plt.scatter(ue_small.x, ue_small.y, marker='s', color='yellow')
        plt.legend(handles=[yellow_patch])
    elif ue_large is not None:
        plt.scatter(ue_large.x, ue_large.y, marker='s', color='green', label='UE (very large value)')
        plt.legend(handles=[green_patch])



    plt.xlim(0, 1000)
    plt.ylim(0, 1000)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('APs e UEs')
    dx = 1000/(np.sqrt(M))
    plt.xticks(np.arange(0, 1001, dx))
    plt.yticks(np.arange(0, 1001, dx))
    plt.grid(True, linestyle='-', alpha=0.5)
    plt.show()
    
def plot_cdf(cdf: np.ndarray, tipo: str = 'capacity') -> None:
    cdf.sort()
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(cdf) + 1), cdf, label='CDF')
    '''plt.axhline(y=np.percentile(cdf, 1), color='r', linestyle='--', label='1th percentil')
    plt.axhline(y=np.percentile(cdf, 99), color='g', linestyle='--', label='99th percentil')'''

    if tipo == 'sinr':
        plt.title('Estimativa do SINR pelo número de simulações (Método Monte Carlo)')
        plt.ylabel('SINR')
    else:
        plt.title('Estimativa da Capacidade do Canal pelo número de simulações (Método Monte Carlo)')
        plt.ylabel('Capacidade do Canal (Mbps)')
    plt.xlabel('Número de Simulações')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_cdfs(cdf: list, tipo: str = 'capacity') -> None:
    plt.figure(figsize=(10, 6))
    for i in range(len(cdf)):
        cdf[i].sort()
        plt.plot(range(1, len(cdf[i]) + 1), cdf[i], label=f'{i+1} canal(is)')

    if tipo == 'sinr':
        plt.title('Estimativa do SINR pelo número de simulações (Método Monte Carlo)')
        plt.ylabel('SINR')
    else:
        plt.title('Estimativa da Capacidade do Canal pelo número de simulações (Método Monte Carlo)')
        plt.ylabel('Capacidade do Canal (Mbps)')
    plt.xlabel('Número de Simulações')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == '__main__':
    M = 36
    N = 1
    sim = 10000

    sinr, ues, ue_small, ue_large = simular_experimento(M, N, sim)
    print(f'Cenário: {M} APs e {N} canal(is)')
    print('1th:', np.percentile(sinr, 1))
    print('99th:', np.percentile(sinr, 99))
    print(len(sinr))
    plot_APs_UEs(M, ues, ue_small, ue_large)
    plot_cdf(sinr, 'sinr')

import numpy as np
import matplotlib.pyplot as plt

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
        self.channel = np.random.randint(1, N+1)
        self.dist = 0

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
    aps = distribuir_AP(M)  # Mantém a lista fixa de APs
    dist = 1e4
    ap_mais_proximo = None

    # Encontra o AP mais próximo inicialmente
    for ap in aps:
        x = np.linalg.norm(np.array([ue.x, ue.y]) - np.array([ap.x, ap.y]))
        if x < dist:
            dist = x
            ap_mais_proximo = ap

    # Se a distância for menor que 1m, reposiciona a UE e recalcula para todos os APs
    while dist < 1:
        ue.x = np.random.randint(0, 1001)
        ue.y = np.random.randint(0, 1001)
        dist = 1e4
        ap_mais_proximo = None

        for ap in aps:
            x = np.linalg.norm(np.array([ue.x, ue.y]) - np.array([ap.x, ap.y]))
            if x < dist:
                dist = x
                ap_mais_proximo = ap

    # Atualiza as informações da UE e do AP correspondente
    ue.dist = dist
    ue.ap = ap_mais_proximo
    ap_mais_proximo.channel = ue.channel

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

def simular_experimento(N: int, sim: int) -> tuple:
    sinr = []
    cap_canal = []
    ues = []
    av_sum_cap = []
    for _ in range(sim):
        ues = [UE(N) for i in range(13)]
        for ue in ues:
            ue.dist = dist_AP_UE(64, ue)
            s = SINR(ue, ues, N)
            cap = channel_capacity(s, N)
            sinr.append(s)
            cap_canal.append(cap)

    return sinr, cap_canal, ues

if __name__ == '__main__':
    sim = 10000
    N = 1
    
    sinr, cap_canal, ues = simular_experimento(N, sim)
    print(f'Outage (5%): {np.percentile(cap_canal, 5):.2f} Mbps')
import numpy as np
import matplotlib.pyplot as plt

class AP:
    def __init__(self, x: float, y: float, id: int):
        self.x = x
        self.y = y
        self.id = id
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

def distribuir_AP(M: int) -> list:
    APs = []
    dx = 1000/(2*np.sqrt(M))
    a = np.arange(dx, 1001-dx, 2*dx)
    x, y = np.meshgrid(a, a)
    id = 1
    for xi, yi in zip(x.ravel(), y.ravel()):
        APs.append(AP(xi, yi, id))
        id += 1
    return APs

def attach_AP_UE(ue: UE, aps: list) -> float:
    dist = np.inf
    ap_escolhida = None
    best_gain = -np.inf

    # Encontra o AP mais próximo inicialmente
    for ap in aps:
        d = np.linalg.norm(np.array([ue.x, ue.y]) - np.array([ap.x, ap.y]))
        while d < 1:    # Se a distância for menor que 1m, reposiciona a UE e recalcula para todos os APs
            ue.x = np.random.randint(0, 1001)
            ue.y = np.random.randint(0, 1001)
            d = np.linalg.norm(np.array([ue.x, ue.y]) - np.array([ap.x, ap.y]))

        # Simula o efeito lognormal na distância
        x = np.random.lognormal(0, 2)
        gain = x*1e-4/(d**4)
        if gain > best_gain:
            best_gain = gain
            dist = d
            ap_escolhida = ap

    # Atualiza as informações da UE e do AP correspondente
    ue.ap = ap_escolhida
    ap_escolhida.ues.append(ue)
    return dist

def pr_interferente(ue1: UE, ue2: UE) -> float:
    if ue1.channel == ue2.channel:
        dist = np.linalg.norm(np.array([ue1.ap.x, ue1.ap.y]) - np.array([ue2.x, ue2.y]))
        x = np.random.lognormal(0, 2)
        return x*1e-4/(dist**4)
    return 0

def SINR(ue: UE, ues: list ,N: int) -> float:
    pt=1        # transmited power
    k=1e-4      # constant
    n=4         # path gain
    bt=1e8      # avaiable bandwidth
    k0=1e-20    # constant for the noise power


    x = np.random.lognormal(0, 2)
    pn = k0*bt/N
    pr = x*pt*k/(ue.dist**n)
    pr_int = np.sum([pr_interferente(ue, u) for u in ues if u!=ue])
    
    return pr/(pn + pr_int)

def channel_capacity(SINR: float, N: int) -> float:
    bt=100      # avaiable bandwidth
    return np.around((bt/N)*np.log2(1+SINR), 4)

def simular_experimento(M: int, N: int, sim: int) -> tuple:
    aps = distribuir_AP(M)  # Mantém a lista fixa de APs
    sinr = []
    cap_canal = []
    ues = []
    av_sum_cap = []

    for _ in range(sim):
        ues = [UE(2) for i in range(13)]
        sum_cap = 0
        for ue in ues:
            ue.dist = attach_AP_UE(ue, aps)
            s = SINR(ue, ues, N)
            cap = channel_capacity(s, N)

            sinr.append(s)
            cap_canal.append(cap)
            sum_cap += cap

        av_sum_cap.append(sum_cap)

    return sinr, cap_canal, av_sum_cap

def plot_cdfs(cdf: list, tipo: str = 'capacity') -> None:
    plt.figure(figsize=(10, 6))

    if tipo == 'sinr':
        # Ordena os valores da CDF
        for i in range(len(cdf)):
            # Atualizando todos os valores para log2(1 + x)
            cdf[i] = [np.log2(1 + a) for a in cdf[i]]

            cdf[i].sort()
            
            # Calcula os percentis de 0 a 1
            percentis = np.linspace(0, 1, len(cdf[i]))

            # Plota usando os percentis no eixo X
            plt.plot(cdf[i], percentis, label=f' {i+1} UE')

        plt.title(f'CDF do SINR por UE')
        plt.xlabel('SINR')

    else:
        for i in range(len(cdf)):
            # Ordena os valores da CDF
            cdf[i].sort()
            percentis = np.linspace(0, 1, len(cdf[i]))

            plt.plot(cdf[i], percentis, label=f'{[i+1]} UE')

        plt.title(f'CDF da Capacidade do Canal por UE')
        plt.xlabel('Capacidade do Canal (Mbps)')
    
    plt.ylabel('Percentil')
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == '__main__':
    M = 64
    N = 2
    sim = 10000

    sinr, cap_canal, sinr_ues, cap_ues = simular_experimento(M, N, sim)



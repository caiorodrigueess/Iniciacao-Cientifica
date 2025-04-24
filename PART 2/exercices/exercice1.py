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

def simular_experimento(M: int, sim: int) -> tuple:
    sinr = []
    cap_canal = []
    ues = []
    av_sum_cap = []
    for _ in range(sim):
        ues = [UE(2) for i in range(13)]
        sum_cap = []
        for ue in ues:
            ue.dist = dist_AP_UE(M, ue)
            s = SINR(ue, ues, 2)
            cap = channel_capacity(s, 2)
            sinr.append(s)
            cap_canal.append(cap)
            sum_cap.append(cap)
        av_sum_cap.append(np.mean(sum_cap))

    return sinr, cap_canal, ues, av_sum_cap

def plot_cdfs(cdf: list, m: int, x: list, tipo: str = 'capacity') -> None:
    plt.figure(figsize=(10, 6))

    if tipo == 'sinr':
        # Ordena os valores da CDF
        for i in range(len(cdf)):
            # Atualizando todos os valores para log2(1 + x)
            cdf[i] = [np.log2(1 + a) for a in np.ravel(cdf[i])]

            cdf[i].sort()
            
            # Calcula os percentis de 0 a 1
            percentis = np.linspace(0, 1, len(cdf[i]))

            # Plota usando os percentis no eixo X
            plt.plot(cdf[i], percentis, label=f'{x[i]} canais')

        plt.yticks([0, 0.1, 0.3, 0.5, 0.7, 0.9, 1])
        plt.title(f'Estimativa do SINR pelo número de simulações ({m} APs)')
        plt.xlabel('SINR')

    else:
        for i in range(len(cdf)):
            # Ordena os valores da CDF
            cdf[i].sort()
            percentis = np.linspace(0, 1, len(cdf[i]))

            plt.plot(cdf[i], percentis, label=f'{x[i]} canais')

        plt.yticks([0, 0.1, 0.3, 0.5, 0.7, 0.9, 1])
        plt.axvline(x=100, color='r', linewidth=0.7, linestyle='--', label = '100 Mbps')
        plt.title(f'Estimativa da Capacidade do Canal pelo número de simulações ({m} APs)')
        plt.xlabel('Capacidade do Canal (Mbps)')
    
    plt.ylabel('Percentil')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == '__main__':
    M = [64, 49, 81]
    sim = 10000
    
    for m in M:
        sinr = []
        cap_canal = []
        x = []
        a, b, c, d = simular_experimento(m, sim)
        sinr.append(a)
        cap_canal.append(b)
        print(f'Cenário: {m} APs e 2 canais')

        
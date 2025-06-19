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

def dist_AP_UE(M: int, ue: UE):
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

def pr_interferente(ue1: UE, ue2: UE, shadowing:int = '') -> float:
    if ue1.channel == ue2.channel:
        dist = np.linalg.norm(np.array([ue1.ap.x, ue1.ap.y]) - np.array([ue2.x, ue2.y]))
        x = np.random.lognormal(0, 2) if shadowing == 'shadowing' else 1
        return x*1e-4/(dist**4)
    return 0

def SINR(ue: UE, ues: list ,N: int, shadowing: str='') -> float:
    pt=1        # transmited power
    k=1e-4      # constant
    n=4         # path gain
    bt=1e8      # avaiable bandwidth
    k0=1e-20    # constant for the noise power


    x = np.random.lognormal(0, 2) if shadowing == 'shadowing' else 1
    pn = k0*bt/N
    pr = x*pt*k/(ue.dist**n)
    pr_int = np.sum([pr_interferente(ue, u, shadowing) for u in ues if u!=ue])
    
    return pr/(pn + pr_int)

def channel_capacity(SINR: float, N: int) -> float:
    bt=100      # avaiable bandwidth
    return np.around((bt/N)*np.log2(1+SINR), 4)

def simular_experimento(M: int, N: int, sim: int, shadowing: str = '') -> tuple:
    sinr = []
    cap_canal = []
    ues = []
    av_sum_cap = []
    for _ in range(sim):
        ues = [UE(2) for i in range(13)]
        sum_cap = 0
        for ue in ues:
            ue.dist = dist_AP_UE(M, ue)
            s = SINR(ue, ues, N, shadowing)
            cap = channel_capacity(s, N)
            sinr.append(s)
            cap_canal.append(cap)
            sum_cap += cap
        av_sum_cap.append(sum_cap)

    return sinr, cap_canal, np.mean(av_sum_cap)

def plot_cdfs(cdf: list, m: int, tipo: str = 'capacity', shadowing: str = '') -> None:
    plt.figure(figsize=(10, 6))

    if tipo == 'sinr':
        cdf = [np.log2(1+a) for a in cdf]
        cdf.sort()
        percentis = np.linspace(0, 1, len(cdf))
        
        if shadowing == 'shadowing':
            plt.plot(cdf, percentis, label=f'With shadowing')
            plt.title(f'Estimativa do SINR ({m} APs, 2 canais)')
            plt.xlabel('SINR')

        else:
            plt.plot(cdf, percentis, label=f'Without shadowing')
            plt.title(f'Estimativa do SINR ({m} APs, 2 canais)')
            plt.xlabel('SINR')


    else:
        cdf.sort()
        percentis = np.linspace(0, 1, len(cdf))

        plt.plot(cdf, percentis, label=f'Capacidade do Canal')
        plt.title(f'Estimativa da Capacidade do Canal ({m} APs)')
        plt.xlabel('Capacidade do Canal (Mbps)')
    
    plt.ylabel('Percentil')
    plt.legend()
    plt.grid(True)


if __name__ == '__main__':
    M = 64
    N = 2
    sim = 10000

    sinr, cap_canal, sum_cap = simular_experimento(M, N, sim, 'shadowing')
    sinr1, a, b = simular_experimento(M, N, sim)

    print(f'Cenário: {M} APs e {N} canais)')
    print(f'a) SINR de cada UE:')
    print('10th:', np.percentile(sinr, 10))
    print('50th:', np.percentile(sinr, 50))

    print(f'b) Capacidade do Canal de cada UE:')
    print('10th:', np.percentile(cap_canal, 10), 'Mbps')
    print('50th:', np.percentile(cap_canal, 50), 'Mbps')

    print(f'c) Capacidade total do Canal: {sum_cap} Mbps')

#    plot_cdfs(cap_canal, M)
    plot_cdfs(sinr, M, 'sinr', 'shadowing')
    plot_cdfs(sinr1, M, 'sinr')
    plt.show()


''' #######
- Adicionar os valores de SINR e Capacidade do Canal sem o efeito de shadowing para comparação.
- Finalizar o exercicio 5
'''
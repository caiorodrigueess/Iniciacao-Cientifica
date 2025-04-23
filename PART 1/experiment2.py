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

def simular_experimento(M: int, N: int, sim: int) -> tuple:
    sinr = []
    cap_canal = []
    ues = []
    for _ in range(sim):
        ues = [UE(N) for i in range(13)]
        for ue in ues:
            ue.dist = dist_AP_UE(M, ue)
            s = SINR(ue, ues, N)
            cap = channel_capacity(s, N)
            sinr.append(s)
            cap_canal.append(cap)

    return sinr, cap_canal, ues

def plot_APs_UEs(M: int, ues: list) -> None:
    plt.figure(figsize=(10, 6))
    aps = distribuir_AP(M)

    for ap in aps:
        plt.scatter(ap.x, ap.y, marker='^', color='blue', label='AP' if ap == aps[0] else "")
    for ue in ues:
        plt.scatter(ue.x, ue.y, marker='s', color='red', label='UE')
        plt.plot([ue.ap.x, ue.x], [ue.ap.y, ue.y], linestyle='-', color='k', linewidth=0.25, )

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

def plot_cdfs(cdf: list, n: int, m: list, tipo: str = 'capacity') -> None:
    plt.figure(figsize=(10, 6))

    if tipo == 'sinr':
        # Ordena os valores da CDF
        for i in range(len(cdf)):
            # Atualizando todos os valores para log2(1 + x)
            cdf[i] = [np.log2(1 + a) for a in np.ravel(cdf[i])]

            x = np.sort(cdf[i])
            y = np.arange(1, len(cdf[i]) + 1) / len(cdf[i])

            # Encontrar o 10th percentil
            percentil_10 = np.percentile(cdf[i], 10)
            # Filtrar os dados até o 10th percentil
            mask = x <= percentil_10
            x_filtrado = x[mask]
            y_filtrado = y[mask]

            # Plotar a CDF até o 10th percentil
            plt.plot(x_filtrado, y_filtrado, label=f'{m[i]} APs')

        plt.title(f'Estimativa do SINR pelo número de simulações ({n} canais)')
        plt.xlabel('SINR')

    else:
        for i in range(len(cdf)):
            # Ordena os valores da CDF
            x = np.sort(cdf[i])
            y = np.arange(1, len(cdf[i]) + 1) / len(cdf[i])

            # Encontrar o 10th percentil
            percentil_10 = np.percentile(cdf[i], 10)
            # Filtrar os dados até o 10th percentil
            mask = x <= percentil_10
            x_filtrado = x[mask]
            y_filtrado = y[mask]

            # Plotar a CDF até o 10th percentil
            plt.plot(x_filtrado, y_filtrado, label=f'{m[i]} APs')

        plt.title(f'Estimativa da Capacidade do Canal pelo número de simulações ({n} canais)')
        plt.xlabel('Capacidade do Canal (Mbps)')
    
    plt.ylabel('Percentil')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == '__main__':
    M = [1, 9, 36, 64]
    N = [1, 2, 3]
    sim = 10000
    
    for n in N:
        sinr = []
        cap_canal = []
        x = []
        for m in M:
            a, b, c = simular_experimento(m, n, sim)
            sinr.append(a)
            cap_canal.append(b)
            print(f'Cenário: {m} APs e {n} canal(is)')
            print('CDF Channel Capacity:')
            print('10th:', np.percentile(b, 10))
            print('50th:', np.percentile(b, 50))
            print('90th:', np.percentile(b, 90))
            x.append(m)
        plot_cdfs(sinr, n, x, 'sinr')
        plot_cdfs(cap_canal, n, x)

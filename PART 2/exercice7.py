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
    id_counter = 0
    def __init__(self):
        self.id = UE.id_counter
        UE.id_counter += 1
        self.x = np.random.randint(0, 1001)
        self.y = np.random.randint(0, 1001)
        self.ap = None
        self.channel = 0
        self.dist = 0
        self.gain = 0
        self.ca = False
        self.sinr = 0

    def __str__(self):
        return f'UE({self.x}, {self.y})'
    
    def __eq__(self, other):
        if isinstance(other, UE):
            return self.id == other.id
        return False

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
    ap_coords = np.array([[ap.x, ap.y] for ap in aps])

    while True:
        ue_coord = np.array([ue.x, ue.y])
        distances = np.linalg.norm(ap_coords - ue_coord, axis=1)
        if np.all(distances >= 1):
            break
        ue.x = np.random.randint(0, 1001)
        ue.y = np.random.randint(0, 1001)

    # Cálculo vetorizado do ganho
    x = np.random.lognormal(0, 2, len(aps))
    distances = np.linalg.norm(ap_coords - np.array([ue.x, ue.y]), axis=1)
    distances = np.maximum(distances, 1)  # Evita divisão por 0
    gains = x * 1e-4 / (distances**4)
    best_index = np.argmax(gains)

    ue.ap = aps[best_index]
    ue.gain = gains[best_index]
    ue.dist = distances[best_index]
    aps[best_index].ues.append(ue)

    todos_os_canais = {0, 1}
    canais_em_uso = {u.channel for u in aps[best_index].ues[:-1]}
    canais_livres = list(todos_os_canais - canais_em_uso)
    if canais_livres:
        ue.channel = np.random.choice(canais_livres)
    else:
        canais = [u.channel for u in aps[best_index].ues[:-1]]
        canal_menos_usado = np.argmin(np.bincount(canais))
        ue.channel = canal_menos_usado

    # Cálculo da potência transmitida com controle de potência
    pt_max=1    # max transmitted power
    pt=1        # transmitted power
    bt=1e8      # available bandwidth
    k0=1e-20    # constant for the noise power
    pn = k0*bt/N
    pt_pc = pn/ue.gain
    pt = min(pt_pc, pt_max)
    ue.pt = pt

    return ue.dist

def SINR(ue: UE, ues: list, N: int, ca: bool) -> float | list:
    bt=1e8      # avaiable bandwidth
    k0=1e-20    # constant for the noise power

    pn = k0*bt/N
    pr = ue.gain * ue.pt

    interferentes = [u for u in ues if u != ue and u.channel == ue.channel]
    if not interferentes:
        return pr / pn

    pr_int = np.sum([u.gain * u.pt for u in interferentes])

    sinr = pr / (pn + pr_int)

    ue.sinr = sinr
    return sinr

def channel_capacity(ue: UE, N: int, ca: bool) -> float:
    bt=100      # 100 MHz de largura de banda
    
    # Se for o modelo sem agregação de canal, calcula a capacidade normalmente
    return np.around((bt/N)*np.log2(1+ue.sinr), 4)

def simular_experimento(M: int, N: int, sim: int, ca: bool = True) -> tuple:
    aps = distribuir_AP(M)  # Mantém a lista fixa de APs
    sinr = []
    cap_canal = []
    ues = []
    av_sum_cap = []

    for _ in range(sim):
        ues = [UE() for i in range(13)]
        sum_cap = 0
        for ue in ues:
            ue.dist = attach_AP_UE(ue, aps)

        if ca:
            for ue in ues:
                if len(ue.ap.ues) == 1:
                    ue.ca = True
            
        for ue in ues:
            s = SINR(ue, ues, N, ca)
            cap = channel_capacity(ue, N, ca)
            sum_cap += cap

            sinr.extend(x for x in s) if isinstance(s, list) else sinr.append(s)
            cap_canal.append(cap)
        av_sum_cap.append(sum_cap)

        for ap in aps:
            ap.ues = []  # Limpa a lista de UEs para o próximo experimento


    return sinr, cap_canal, np.mean(av_sum_cap)

def plot_cdfs(cdf: list, tipo: str = 'capacity') -> None:
    plt.figure(figsize=(10, 6))
    percentis = []
    if tipo == 'sinr':
        # Ordena os valores da CDF
        for i in range(len(cdf)):
            # Atualizando todos os valores para log2(1 + x)
            cdf[i] = [np.log2(1 + a) for a in cdf[i]]

            cdf[i].sort()
            
            # Calcula os percentis de 0 a 1
            percentis.append(np.linspace(0, 1, len(cdf[i])))

        # Plota usando os percentis no eixo X
        plt.plot(cdf[0], percentis[0], label=f'Without channel agregation')
        plt.plot(cdf[1], percentis[1], label=f'With channel agregation')

        plt.title(f'CDF do SINR por UE')
        plt.xlabel('SINR')

    else:
        for i in range(len(cdf)):
            # Ordena os valores da CDF
            cdf[i].sort()
            percentis.append(np.linspace(0, 1, len(cdf[i])))

        plt.plot(cdf[0], percentis[0], label=f'Without channel agregation')
        plt.plot(cdf[1], percentis[1], label=f'With channel agregation')
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

    
    sinr1, cap_canal1, av_sum_cap1 = simular_experimento(M, N, sim, False)
    print(f'Average sum capacity without channel aggregation: {av_sum_cap1:.2f} Mbps')
    print(f'10th percentile of SINR without channel aggregation: {np.percentile(sinr1, 10):.2f} dB')
    print(f'50th percentile of SINR without channel aggregation: {np.percentile(sinr1, 50):.2f} dB')
    print(f'10th percentile of channel capacity without channel aggregation: {np.percentile(cap_canal1, 10):.2f} Mbps')
    print(f'50th percentile of channel capacity without channel aggregation: {np.percentile(cap_canal1, 50):.2f} Mbps')


    sinr2, cap_canal2, av_sum_cap2 = simular_experimento(M, N, sim, True)
    print(f'Average sum capacity with channel aggregation: {av_sum_cap2:.2f} Mbps')
    print(f'10th percentile of SINR with channel aggregation: {np.percentile(sinr2, 10):.2f} dB')
    print(f'50th percentile of SINR with channel aggregation: {np.percentile(sinr2, 50):.2f} dB')
    print(f'10th percentile of channel capacity with channel aggregation: {np.percentile(cap_canal2, 10):.2f} Mbps')
    print(f'50th percentile of channel capacity with channel aggregation: {np.percentile(cap_canal2, 50):.2f} Mbps')

    plot_cdfs([sinr1, sinr2], 'sinr')
    plot_cdfs([cap_canal1, cap_canal2], 'capacity')

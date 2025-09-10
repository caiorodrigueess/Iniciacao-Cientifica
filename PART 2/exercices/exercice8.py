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
        self.pt = 1

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

def attach_AP_UE(ue: UE, aps: list, pc: bool) -> None:
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

    # Cálculo da potência transmitida com controle de potência
    pt_max=1    # max transmitted power
    pt=1        # transmitted power
    bt=1e8      # available bandwidth
    k0=1e-20    # constant for the noise power
    pn = k0*bt/N
    
    if pc:
        pt_pc = pn/ue.gain
        pt = min(pt_pc, pt_max)
    
    ue.pt = pt
    return None

def SINR(ue: UE, ues: list, N: int) -> float | list:
    bt=1e8      # available bandwidth
    k0=1e-20    # constant for the noise power

    # Cálculo da potência do ruído
    pn = k0*bt/N
    pr = ue.gain * ue.pt
    pt = np.array([u.pt for u in ues if u != ue])

    # Definindo as distancias dos interferentes
    interfer_coords = np.array([[u.x, u.y] for u in ues if u != ue])
    ap_coord = np.array([ue.ap.x, ue.ap.y])
    distances = np.linalg.norm(interfer_coords - ap_coord, axis=1)
    distances = np.maximum(distances, 1)

    # Cálculo vetorizado da potência recebida dos interferentes
    x = np.random.lognormal(0, 2, len(interfer_coords))
    pr_int = np.sum(x * 1e-4 * pt / (distances ** 4))

    # Cálculo do SINR
    sinr = pr / (pn + pr_int)
    ue.sinr = sinr
    return sinr

def channel_capacity(ue: UE, N: int) -> float:
    bt=100      # 100 MHz de largura de banda
    # Se for o modelo sem agregação de canal, calcula a capacidade normalmente
    return np.around((bt/N)*np.log2(1+ue.sinr), 4)

def simular_experimento(M: int, N: int, sim: int, pc: bool = True) -> tuple:
    aps = distribuir_AP(M)  # Mantém a lista fixa de APs
    sinr = []
    cap_canal = []
    ues = []
    av_sum_cap = []

    for _ in range(sim):
        ues = [UE() for i in range(15)]
        sum_cap = 0
        for ue in ues:
            attach_AP_UE(ue, aps, pc)
            
        for ue in ues:
            s = SINR(ue, ues, N)
            cap = channel_capacity(ue, N)
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
            # Atualizando todos os valores para 10*log10(x)
            cdf[i] = [10*np.log10(a) for a in cdf[i]]

            cdf[i].sort()
            
            # Calcula os percentis de 0 a 1
            percentis.append(np.linspace(0, 1, len(cdf[i])))

        # Plota usando os percentis no eixo X
        plt.plot(cdf[0], percentis[0], label=f'Without channel agregation')
        plt.plot(cdf[1], percentis[1], label=f'With channel agregation')

        plt.title(f'CDF do sinr por UE')
        plt.xlabel('sinr')

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
    M = 36
    N = 1
    sim = 10000
    
    sinr1, cap_canal1, av_sum_cap1 = simular_experimento(M, N, sim, pc=False)
    print(f'Average sum capacity without power control: {av_sum_cap1:.2f} Mbps')
    print(f'10th percentile of sinr without power control: {np.percentile(sinr1, 10):.2f} dB')
    print(f'50th percentile of sinr without power control: {np.percentile(sinr1, 50):.2f} dB')
    print(f'10th percentile of channel capacity without power control: {np.percentile(cap_canal1, 10):.2f} Mbps')
    print(f'50th percentile of channel capacity without power control: {np.percentile(cap_canal1, 50):.2f} Mbps')

    sinr2, cap_canal2, av_sum_cap2 = simular_experimento(M, N, sim, pc=True)
    print(f'Average sum capacity with power control: {av_sum_cap2:.2f} Mbps')
    print(f'10th percentile of sinr with power control: {np.percentile(sinr2, 10):.2f} dB')
    print(f'50th percentile of sinr with power control: {np.percentile(sinr2, 50):.2f} dB')
    print(f'10th percentile of channel capacity with power control: {np.percentile(cap_canal2, 10):.2f} Mbps')
    print(f'50th percentile of channel capacity with power control: {np.percentile(cap_canal2, 50):.2f} Mbps')

    plot_cdfs([sinr1, sinr2], 'sinr')
    plot_cdfs([cap_canal1, cap_canal2], 'capacity')

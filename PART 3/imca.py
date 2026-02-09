import random
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

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
    def __init__(self, N: int):
        self.id = UE.id_counter
        UE.id_counter += 1
        self.x = np.random.randint(0, 1001)
        self.y = np.random.randint(0, 1001)
        self.ap = None
        self.channel = 0
        self.dist = 0
        self.gain = 0
        self.power = 1  # Transmit power

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
    id = 0
    for xi, yi in zip(x.ravel(), y.ravel()):
        APs.append(AP(xi, yi, id))
        id += 1
    return APs

def gain_matrix(ues: list, aps: list) -> np.ndarray:
    ''' Inicializa a matriz'''
    G = np.zeros((len(ues), len(aps)))
    ap_coords = np.array([[ap.x, ap.y] for ap in aps])

    for i, ue in enumerate(ues):
        # garante que o UE não está a menos de 1 m de nenhum AP
        while True:
            ue_coord = np.array([ue.x, ue.y])
            d_all = np.linalg.norm(ap_coords - ue_coord, axis=1)
            if np.all(d_all >= 1):
                break
            ue.x = np.random.randint(0, 1001)
            ue.y = np.random.randint(0, 1001)

        # calcula ganhos para todos APs
        for j in range(len(aps)):
            d_ij = np.linalg.norm(ue_coord - ap_coords[j])
            d = max(d_ij, 1.0)
            G[i, j] = np.random.lognormal(0, 2) * 1e-4 / (d**4)
    return G

def alocar_canais_ortogonal(access_points, ues, number_channels, allocation):
    """
    Aloca canais aos UEs de forma ortogonal dentro de cada célula (AP).

    Esta função é uma adaptação do método 'channel_allocation'

    Argumentos:
    access_points (list): A lista de objetos AccessPoint.
    user_equipaments (list): A lista de objetos UserEquipament.
    number_channels (int): O número total de canais ortogonais (N).

    Modifica:
    O atributo 'channel' de cada objeto na lista user_equipaments é
    atualizado "in-place" (no próprio objeto).
    """

    # 1. Zera os canais de todos os UEs
    # (Necessário para a lógica de "preenchimento" funcionar)
    for ue in ues:
        ue.channel = None

    if allocation == 'random':
        for ue in ues:
            ue.channel = np.random.randint(1, number_channels + 1)
        return None

    # 2. Inicializa o contador global de uso de canal
    channels = {i: 0 for i in range(1, number_channels + 1)}

    # 3. Itera por cada AP para alocação intra-célula
    for ap in access_points:
        available_channels = list(channels.keys())
        random.shuffle(available_channels)

        # 4. Itera pelos UEs (por índice) conectados a este AP
        for ue0 in ap.ues:
            if not available_channels:
                # Para se o AP tiver mais UEs do que canais
                break

            # 5. Atribui um canal disponível e o remove da lista do AP
            channel = available_channels.pop()

            # 6. Atribui o canal ao UE e incrementa o contador global
            index_ue = ue0.id
            ues[index_ue].channel = channel
            channels[channel] += 1

    # 7. Etapa de Preenchimento (Cleanup)
    for ue in ues:
        if ue.channel is None:
            # Se o UE não recebeu um canal (passo 4),
            # atribui o canal menos usado globalmente.
            least_used_channel = min(channels, key=channels.get)
            ue.channel = least_used_channel
            channels[least_used_channel] += 1

def attach_AP_UE(ues: list, aps: list, G: np.ndarray) -> float:
    for i, ue in enumerate(ues):
        # Encontra o AP com o maior ganho para este UE
        best_index = np.argmax(G[i, :])
        ue.ap = aps[best_index]
        ue.gain = G[i, best_index]
        ue.dist = np.linalg.norm(np.array([ue.x, ue.y]) - np.array([ue.ap.x, ue.ap.y]))
        aps[best_index].ues.append(ue)


def SINR(ues: list, N: int, G: np.ndarray) -> float:
    pt=1        # transmited power
    bt=1e8      # avaiable bandwidth
    k0=1e-20    # constant for the noise power

    pn = k0*bt/N

    num_ues = len(ues)

    # Coleta os vetores de estado do sistema (extraídos dos objetos)
    # Vetor de Potência (P)
    P = np.array([ue.power for ue in ues])

    # Vetor de Associação de AP (A)
    # A[k] = índice do AP que serve o UE k
    A = np.array([ue.ap.id for ue in ues])

    # Vetor de Alocação de Canal (C)
    # C[k] = canal (ex: 1, 2, 3...) usado pelo UE k
    C = np.array([ue.channel for ue in ues])

    sinr_list = []

    # Itera por cada UE 'k' para calcular seu SINR
    for k in range(num_ues):

        # 1. Encontrar o AP de serviço para este UE 'k'
        m = A[k] # m é o índice do AP de serviço

        # 2. Calcular a Potência do Sinal (S)
        S = G[k, m] * P[k]

        # 3. Calcular a Potência de Interferência (I)
        I = 0.0
        for i in range(num_ues):
            if i != k and C[i] == C[k]:
                # I += G[i, m] * P[i]
                I += G[i, m] * P[i]

        # 5. Calcular SINR = S / (I + N)
        sinr = S / (I + pn)
        sinr_list.append(sinr)

    return sinr_list

def channel_capacity(sinr_valores: list, N: int) -> list:
    # largura de banda por canal (B_canal)
    B_per_channel = 1e8 / N

    # converte a(s) entrada(s) SINR para um array numpy
    sinr_array = np.asarray(sinr_valores)

    # C = B * log2(1 + SINR)
    capacity_mbps = B_per_channel * np.log2(1 + sinr_array) / 1e6  # Converte para Mbps

    return capacity_mbps

def simular_experimento(M: int, N: int, sim: int, allocation: str = '') -> tuple:
    aps = distribuir_AP(M)
    sinr = []
    cap_canal = []
    av_sum_cap = []

    for _ in range(sim):
        UE.id_counter = 0 
        ues = [UE(N) for i in range(13)] # Mantém K=13 conforme seu código
        
        # calculando Ganho Lento (Distância + Shadowing)
        G_slow = gain_matrix(ues, aps)
        
        # attachment (Baseado apenas no Ganho Lento)
        attach_AP_UE(ues, aps, G_slow)
        
        # Fast Fading (Rayleigh) para TODOS os canais
        # Shape: [Num_UEs, Num_APs, Num_Canais]
        sigma_r = 1/np.sqrt(2)

        # Gerando componentes normal (0, sigma)
        h_real = np.random.normal(0, sigma_r, (len(ues), len(aps), N))
        h_imag = np.random.normal(0, sigma_r, (len(ues), len(aps), N))

        G_rayleigh = h_real**2 + h_imag**2 # |h|^2
        
        # alocação de canais (Passando as matrizes necessárias para o IMCA)
        alocar_canais_ortogonal(aps, ues, N, allocation, G_slow, G_rayleigh)

        # construção da Matriz de Ganho Efetiva para cálculo da SINR Real
        # a função SINR() usa uma matriz G[k,m]. Precisamos que essa matriz contenha o ganho real do canal que o UE *efetivamente usou*.
        G_effective = np.zeros_like(G_slow)
        for i, ue in enumerate(ues):
            # O ganho efetivo do UE 'i' para qualquer AP 'j' é o ganho no canal que ele escolheu.
            c_idx = ue.channel - 1 # 0-based
            G_effective[i, :] = G_slow[i, :] * G_rayleigh[i, :, c_idx]

        # cálculo dos KPIs usando o ganho efetivo
        s = SINR(ues, N, G_effective)
        cap = channel_capacity(s, N)
        sum_cap = np.sum(cap)

        # salvando as amostras para plotar as CDFs depois
        sinr.extend(s)
        cap_canal.extend(cap)
        av_sum_cap.append(sum_cap)

        # limpando a lista de UEs em cada AP para o próximo experimento
        for ap in aps:
            ap.ues = []

    return sinr, cap_canal, np.mean(av_sum_cap)

def plot_cdfs(cdf_sinr: list, cdf_capacity: list) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Plot SINR CDF
    for i in range(len(cdf_sinr)):
        cdf_sinr[i] = [10*np.log10(a) for a in cdf_sinr[i]]
        cdf_sinr[i].sort()
        percentis = np.linspace(0, 1, len(cdf_sinr[i]))
        axes[0].plot(cdf_sinr[i], percentis, label=f'SINR {"Per-AP" if i==1 else "Random"} channel allocation')

    axes[0].axhline(y=0.10, color='r', linewidth=0.7, linestyle='--', label = f'10th Percentil')
    axes[0].axhline(y=0.50, color='b', linewidth=0.7, linestyle='--', label = f'50th Percentil')
    axes[0].set_title('CDF do SINR por UE')
    axes[0].set_xlabel('SINR (dB)')
    axes[0].set_ylabel('Percentil')
    axes[0].legend()
    axes[0].grid(True)

    # Plot Capacity CDF
    for i in range(len(cdf_capacity)):
        cdf_capacity[i].sort()
        percentis = np.linspace(0, 1, len(cdf_capacity[i]))
        axes[1].plot(cdf_capacity[i], percentis, label=f'Cap. do Canal {"Per-AP" if i==1 else "Random"} channel allocation')

    axes[1].axhline(y=0.10, color='r', linewidth=0.7, linestyle='--', label = f'10th Percentil')
    axes[1].axhline(y=0.50, color='b', linewidth=0.7, linestyle='--', label = f'50th Percentil')
    axes[1].set_title('CDF da Capacidade do Canal por UE')
    axes[1].set_xlabel('Capacidade do Canal (Mbps)')
    axes[1].set_ylabel('Percentil')
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plt.show()
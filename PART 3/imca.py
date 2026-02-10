import random
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

GAIN_MATRIX_CACHE = None


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

def alocar_canais(aps: list, ues: list, G: np.ndarray, N: int, allocation: str):
    # zera os canais de todos os UEs
    for ue in ues:
        ue.channel = None

    # ======================
    # 1. Random
    # ======================
    if allocation == 'random':
        for ue in ues:
            ue.channel = np.random.randint(1, N + 1)
        return None

    # ======================
    # 2. Per-AP Orthogonal
    # ======================

    elif allocation == 'papoa':
        # lista de canais disponíveis (1, 2, ..., N)
        canais = {i: 0 for i in range(1, N + 1)}

        # itera por cada AP 
        for ap in aps:
            canais_disponiveis = list(canais.keys())      # lista de canais disponíveis para este AP
            random.shuffle(canais_disponiveis)              # embaralha para evitar vieses

            # itera pelos UEs (por índice) conectados a este AP
            for ue in ap.ues:
                if not canais_disponiveis:
                    canais_disponiveis = list(canais.keys())  # se esgotar, reinicia a lista de canais
                    random.shuffle(canais_disponiveis)          # embaralha novamente

                # atribui um canal disponível e o remove da lista do AP
                channel = canais_disponiveis.pop()

                # atribui o canal ao UE e incrementa o contador global
                index_ue = ue.id
                ues[index_ue].channel = channel
                canais[channel] += 1

        return None
    
    # ======================
    # 3. IMCA
    # ======================

    elif allocation == 'imca':
        # embaralha a ordem dos UEs para evitar vieses
        indices_ues = list(range(len(ues)))
        random.shuffle(indices_ues)

        for idx in indices_ues:
            ue = ues[idx]
            ap = ue.ap
            id_ap = ap.id

            # variáveis auxiliares para encontrar o melhor canal para este UE
            melhor_canal = None
            menor_interferencia = float('inf')

            # itera por cada canal disponível
            for channel in range(1, N + 1):
                # variável para acumular a interferência total neste canal
                interferencia = 0.0
                
                # itera sobre todos os outros UEs do sistema
                for i, outro_ue in enumerate(ues):
                    # se outro UE estiver transmitindo no mesmo canal e for diferente do UE atual, acumula a interferência
                    if i != idx and outro_ue.channel == channel:
                        interferencia += G[i, id_ap] * outro_ue.power

                # atualiza o melhor canal se esta opção tiver menor interferência
                if interferencia < menor_interferencia:
                    menor_interferencia = interferencia
                    melhor_canal = channel

            # atribui o melhor canal encontrado ao UE
            ue.channel = melhor_canal
        return None


def attach_AP_UE(ues: list, aps: list, G: np.ndarray) -> float:
    for i, ue in enumerate(ues):
        # Encontra o AP com o maior ganho para este UE
        best_index = np.argmax(G[i, :])
        ue.ap = aps[best_index]     # associa o UE ao AP com maior ganho
        ue.gain = G[i, best_index]  # armazena o ganho para este UE
        ue.dist = np.linalg.norm(np.array([ue.x, ue.y]) - np.array([ue.ap.x, ue.ap.y]))     # armazena a distancia
        aps[best_index].ues.append(ue)      # adiciona o UE à lista de UEs do AP
    return None

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
        # (Sinal do UE k no AP m)
        # S = G[k, m] * P[k]
        S = G[k, m] * P[k]

        # 3. Calcular a Potência de Interferência (I)
        # (Soma da potência de todos os *outros* UEs 'i'
        #  que transmitem no mesmo canal 'C[k]'
        #  e são recebidos no *nosso* AP 'm')
        I = 0.0
        for i in range(num_ues):
            # A interferência ocorre se:
            # 1. Não é o próprio UE (i != k)
            # 2. O UE 'i' está no mesmo canal que o UE 'k' (C[i] == C[k])
            # (No seu código, a interferência inter-célula é considerada)
            if i != k and C[i] == C[k]:
                # I += G[i, m] * P[i]
                I += G[i, m] * P[i]

        # calcular SINR = S / (I + N)
        sinr = S / (I + pn)
        sinr_list.append(sinr)

    return sinr_list

def channel_capacity(sinr_valores: list, N: int) -> list:
    B_per_channel = 1e8 / N
    sinr_array = np.asarray(sinr_valores)
    capacity_mbps = B_per_channel * np.log2(1 + sinr_array) / 1e6  # Converte para Mbps

    return capacity_mbps

def simular_experimento(ues: list, aps: list, G:np.ndarray, N: int, sim: int, allocation: str = '') -> tuple:
    sinr = []
    cap_canal = []
    av_sum_cap = []

    for _ in range(sim):
        UE.id_counter = 0       # reseta a contagem em cada simulação
        sum_cap = 0
        attach_AP_UE(ues, aps, G)
        alocar_canais(aps, ues, G, N, allocation)

        s = SINR(ues, N, G)
        cap = channel_capacity(s, N)
        sum_cap = np.sum(cap)

        sinr.extend(s)
        cap_canal.extend(cap)
        av_sum_cap.append(sum_cap)

        for ap in aps:
            ap.ues = []  # Limpa a lista de UEs para o próximo experimento

    return sinr, cap_canal, np.mean(av_sum_cap)
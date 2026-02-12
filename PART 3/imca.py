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
            canais_disponiveis = list(canais.keys())        # lista de canais disponíveis para este AP
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
    bt=1e8      # avaiable bandwidth
    k0=1e-20    # constant for the noise power

    # noise power
    pn = k0*bt/N

    # multipath fading para cada canal
    h2 = (np.random.rayleigh(scale=1/np.sqrt(2), size=N))**2

    # vetor de potência
    P = np.array([ue.power for ue in ues])

    # variável para armazenar os valores de SINR
    sinr = []

    for i, ue in enumerate(ues):
        # para cada UE, calcula o sinal útil e a interferência total dos outros UEs no mesmo canal
        s = ue.gain * P[i] * h2[ue.channel - 1]  # sinal útil

        # interferência total de outros UEs no mesmo canal
        interference = 0.0

        # itera sobre todos os outros UEs do sistema
        for j, outro_ue in enumerate(ues):
            # se outro UE estiver transmitindo no mesmo canal e for diferente do UE atual, acumula a interferência
            if j != i and outro_ue.channel == ue.channel:
                interference += G[j, ue.ap.id] * P[j] * h2[ue.channel - 1]

        sinr.append(s / (interference + pn))

    return sinr

def channel_capacity(sinr: list, N: int) -> list:
    B_per_channel = 1e8 / N
    sinr_array = np.asarray(sinr)
    capacity_mbps = B_per_channel * np.log2(1 + sinr_array) / 1e6

    return capacity_mbps

def simular_experimento(ues: list, aps: list, G:np.ndarray, N: int, sim: int, allocation: str = '') -> tuple:
    # ----------------------
    # realiza uma simulação
    # ----------------------

    # associa os UEs aos APs e aloca os canais
    attach_AP_UE(ues, aps, G)
    alocar_canais(aps, ues, G, N, allocation)

    # calcula o SINR, a capacidade de canal e a capacidade total do sistema
    sinr = SINR(ues, N, G)
    cap = channel_capacity(sinr, N)
    sum_cap = np.sum(cap)

    for ap in aps:
        ap.ues = []  # Limpa a lista de UEs para o próximo experimento

    return sinr, cap, sum_cap

'''
if __name__ == '__main__':
    import copy
    # Configurações do Experimento
    M = 64                  # M = 64 APs
    canais = range(2, 11)   # N varia de 2 a 10
    num_simulacoes = 1000
    np.random.seed(42)

    # Estruturas para armazenar os KPIs finais
    resultados = {
        'random': {'sum_cap': [], 'sinr': [], 'cap': []},
        'papoa':  {'sum_cap': [], 'sinr': [], 'cap': []},
        'imca':  {'sum_cap': [], 'sinr': [], 'cap': []}
    }

    # --- Loop Principal ---
    for N in canais:
        # variaveis auxiliares para armazenar a média da capacidade de soma
        sum_cap = {'random': [], 'papoa': [], 'imca':[]}
        sinr = {'random': [], 'papoa': [], 'imca':[]}
        cap = {'random': [], 'papoa': [], 'imca':[]}
        
        for _ in range(num_simulacoes): 
            UE.id_counter = 0
            ues = [UE(N) for i in range(13)]
            aps = distribuir_AP(M)
            G = gain_matrix(ues, aps)

            for modo in ['random', 'papoa', 'imca']:

                # Criamos cópias dos APs e UEs para que um algoritmo não afete o outro
                aps_copy = copy.deepcopy(aps)
                ues_copy = copy.deepcopy(ues)
                    
                sinr_rd, cap_rd, sum_cap_rd = simular_experimento(ues_copy, aps_copy, G, N, num_simulacoes, modo)
                sinr_db = 10 * np.log10(sinr_rd)
                
                # Armazenamos os resultados brutos para calcular os percentis depois
                sinr[modo].extend(sinr_db)
                cap[modo].extend(cap_rd)
                sum_cap[modo].append(sum_cap_rd)

        # calculando a média da capacidade total do sistema em cada algoritmo
        for modo in ['random', 'papoa', 'imca']:
            resultados[modo]['sinr'].append(sinr[modo])
            resultados[modo]['cap'].append(cap[modo])
            resultados[modo]['sum_cap'].append(np.mean(sum_cap[modo]))

    plt.figure(figsize=(20, 5))

    # Mapeamento para garantir que usamos as chaves minúsculas do dicionário 'resultados'
    mapa = {'random': 'Random', 'papoa': 'PAPOA', 'imca': 'IMCA'}

    # Gráfico 1: Capacidade de Soma
    plt.subplot(1, 3, 1)
    for chave in ['random', 'papoa', 'imca']:
        plt.plot(canais, resultados[chave]['sum_cap'], 'o--', label=mapa[chave])
    plt.xlabel('Número de Canais (N)')
    plt.ylabel('Mbps')
    plt.title('Capacidade de Soma')
    plt.legend()
    plt.grid(True)

    # Gráfico 2: SINR (Percentil 10)
    plt.subplot(1, 3, 2)
    for chave in ['random', 'papoa', 'imca']:
        # Calcula o percentil 10 para cada valor de N na lista
        p10_sinr = [np.percentile(res, 10) for res in resultados[chave]['sinr']]
        plt.plot(canais, p10_sinr, 'o--', label=mapa[chave])
    plt.xlabel('Número de Canais (N)')
    plt.ylabel('dB')
    plt.title('Percentil 10 da SINR')
    plt.legend()
    plt.grid(True)

    # Gráfico 3: Capacidade de Canal (Percentil 10)
    plt.subplot(1, 3, 3)
    for chave in ['random', 'papoa', 'imca']:
        # Calcula o percentil 10 para cada valor de N na lista
        p10_cap = [np.percentile(res, 10) for res in resultados[chave]['cap']]
        plt.plot(canais, p10_cap, 'o--', label=mapa[chave])
    plt.xlabel('Número de Canais (N)')
    plt.ylabel('Mbps')
    plt.title('Percentil 10 da Capacidade de Canal')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()
'''
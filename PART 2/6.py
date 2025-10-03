import numpy as np
import matplotlib.pyplot as plt
np.random.seed(42)

class AP:
    def __init__(self, x: float, y: float, id: int):
        self.x = x
        self.y = y
        self.id = id
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
        # ALTERAÇÃO: UE agora tem uma lista de canais e de SINRs
        self.channels = []
        self.sinrs = []
        self.gain = 0
        self.ca = False # Flag para indicar se a CA está ativa para este UE

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

# ALTERAÇÃO: A função agora decide se ativa a CA e aloca os canais corretamente
def attach_ues_to_aps(ues: list, aps: list, G: np.ndarray, ca_ativada_simulacao: bool, N_canais: int) -> None:
    for i, ue in enumerate(ues):
        best_ap_index = np.argmax(G[i, :])   # AP com maior ganho
        best_ap = aps[best_ap_index]
        
        ue.ap = best_ap
        ue.gain = G[i, best_ap_index]        # ganho fixo do link desejado
        best_ap.ues.append(ue)

        # Lógica de ativação da CA movida para cá
        ue.ca = False # Começa como Falso
        if ca_ativada_simulacao and len(best_ap.ues) == 1:
            ue.ca = True

        # Algoritmo de alocação de canal
        todos_os_canais = set(range(N_canais))
        
        if ue.ca:
            # Se a CA estiver ativa para este UE, ele usa todos os canais
            ue.channels = list(todos_os_canais)
        else:
            # Se não, usa a lógica de alocação de canal único
            # Verifica canais de outros UEs que já estão no mesmo AP e alocados
            canais_em_uso = {u.channels[0] for u in best_ap.ues[:-1] if u.channels}
            canais_livres = list(todos_os_canais - canais_em_uso)
            
            if canais_livres:
                assigned_channel = np.random.choice(canais_livres)
            else:
                canais_usados_lista = [u.channels[0] for u in best_ap.ues[:-1] if u.channels]
                if not canais_usados_lista: # Se for o primeiro UE no AP
                    assigned_channel = 0
                else:
                    assigned_channel = np.argmin(np.bincount(canais_usados_lista))

            ue.channels.append(assigned_channel)
    return None

# ALTERAÇÃO: Função SINR completamente reescrita para ser consistente com a CA
def calcular_sinr(ues: list, aps: list, G: np.ndarray, p_vec: np.ndarray, pn: float) -> None:
    """ Calcula a SINR para cada canal alocado de cada UE e armazena no próprio objeto UE. """
    for i, ue in enumerate(ues):
        ue.sinrs = [] # Limpa SINRs antigas
        if not ue.channels or ue.ap is None:
            continue

        # CORREÇÃO: Bug de indexação do AP
        ap_index = aps.index(ue.ap)
        
        # Calcula a potência por canal para o UE atual
        potencia_por_canal_i = p_vec[i] / len(ue.channels)

        # Itera sobre cada canal que o UE 'i' está efetivamente usando
        for canal_c in ue.channels:
            # 1. Calcular o Sinal no canal c
            sinal = G[i, ap_index] * potencia_por_canal_i

            # 2. Calcular a Interferência no canal c
            interf = 0.0
            for k, outro_ue in enumerate(ues):
                if i == k:
                    continue # UE não interfere em si mesmo

                # Verifica se o outro UE está transmitindo no mesmo canal
                if canal_c in outro_ue.channels:
                    # CORREÇÃO: A potência do interferente também é dividida
                    potencia_por_canal_k = p_vec[k] / len(outro_ue.channels)
                    interf += G[k, ap_index] * potencia_por_canal_k
            
            # 3. Calcular e armazenar a SINR para este canal
            sinr_c = sinal / (interf + pn)
            ue.sinrs.append(sinr_c)

# ALTERAÇÃO: Função de capacidade reescrita para somar as taxas dos canais agregados
def channel_capacity(ue: UE, N: int, bt: float) -> float:
    """ Calcula a capacidade total de um UE somando a capacidade de cada um de seus canais. """
    if not ue.sinrs:
        return 0.0
    
    bw_por_canal_mhz = (bt / 1e6) / N
    capacidade_total = 0.0
    
    for sinr_val in ue.sinrs:
        # Garante que o valor da SINR não seja negativo para o log2
        if sinr_val > 0:
            capacidade_total += bw_por_canal_mhz * np.log2(1 + sinr_val)
            
    return np.around(capacidade_total, 4)

# ALTERAÇÃO: Loop de simulação ajustado para usar as novas funções
def simular_experimento(M: int, N: int, sim: int, ca: bool = True) -> tuple:
    aps = distribuir_AP(M)
    lista_sinr_geral = []
    lista_cap_canal_geral = []
    lista_soma_caps = []
    
    # Parâmetros da Simulação
    bt = 1e8      # Largura de banda total em Hz (100 MHz)
    k0 = 1e-20    # Constante de ruído
    pn = k0 * bt / N # Potência de ruído por canal

    for _ in range(sim):
        ues = [UE() for _ in range(13)]
        
        G = gain_matrix(ues, aps)
        # Passa o flag 'ca' para a função decidir sobre a alocação
        attach_ues_to_aps(ues, aps, G, ca, N)
        
        pt = np.full(len(ues), 1.0) # Vetor de potências (1W para todos)

        # Calcula as SINRs (resultados são armazenados nos objetos UE)
        calcular_sinr(ues, aps, G, pt, pn)
        
        # Calcula as capacidades com base nas SINRs armazenadas
        caps = [channel_capacity(ue, N, bt) for ue in ues]

        # Agrega os resultados da simulação
        for ue in ues:
            lista_sinr_geral.extend(ue.sinrs) # Adiciona todas as SINRs calculadas
        lista_cap_canal_geral.extend(caps)
        lista_soma_caps.append(np.sum(caps))

        # Limpa o estado para a próxima iteração
        for ap in aps:
            ap.ues = []

    return lista_sinr_geral, lista_cap_canal_geral, np.mean(lista_soma_caps)

def plot_cdfs(cdf: list, tipo: str = 'capacity') -> None:
    plt.figure(figsize=(10, 6))
    percentis = []
    if tipo == 'sinr':
        # Ordena os valores da CDF
        for i in range(len(cdf)):
            # Atualizando todos os valores para log2(1 + x)
            cdf[i] = 10*np.log10(cdf[i])

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
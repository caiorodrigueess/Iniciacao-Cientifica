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

def attach_AP_UE(ue: UE, aps: list, allocation:str) -> float:
    dist = np.inf
    ap_escolhida = None
    best_gain = -np.inf

    while True:
        posicao_valida = True
        
        # Laço interno para validar a posição atual contra todos os APs
        for ap in aps:
            d = np.linalg.norm(np.array([ue.x, ue.y]) - np.array([ap.x, ap.y]))
            if d < 1:
                posicao_valida = False
                break # Posição é inválida, não precisa checar outros APs            

        # Se a posição for válida para todos os APs, saímos do laço de reposicionamento
        if posicao_valida:
            break
        # Se não for, sorteamos uma nova posição e o laço 'while True' continua para revalidar
        else:
            ue.x = np.random.randint(0, 1001)
            ue.y = np.random.randint(0, 1001)

    for ap in aps:
        d = np.linalg.norm(np.array([ue.x, ue.y]) - np.array([ap.x, ap.y]))

        x = np.random.lognormal(0, 2)
        gain = x * 1e-4 / (d**4)
        
        if gain > best_gain:
            best_gain = gain
            dist = d
            ap_escolhida = ap

    # Atualiza as informações da UE e do AP correspondente
    ue.ap = ap_escolhida
    ue.gain = best_gain
    ap_escolhida.ues.append(ue)

    if allocation == 'random':
        ue.channel = np.random.randint(0, 2)

    elif allocation == 'per-AP':
        todos_os_canais = {0, 1} # Cria um conjunto com todos os canais possíveis, ex: {0, 1}

        # 1. Pega os canais já em uso pelos outros UEs neste AP
        # O '-1' é para excluir o UE atual que acabamos de adicionar
        canais_em_uso = {ue.channel for ue in ap_escolhida.ues[:-1]}

        # 2. Identifica os canais que estão livres nesta célula
        canais_livres = list(todos_os_canais - canais_em_uso)

        # 3. Decide qual canal alocar
        if canais_livres:
            # CASO 1: Existem canais livres. Sorteia um aleatoriamente.
            ue.channel = np.random.choice(canais_livres)
        else:
            # CASO 2: Não há canais livres, precisamos reutilizar.
            # Encontra o canal menos utilizado para balancear a carga.
            contagem_de_canais = Counter(c.channel for c in ap_escolhida.ues[:-1])
            # .most_common() lista do mais comum para o menos comum.
            # O [-1] pega o último item, que é o menos comum.
            # O [0] pega o número do canal desse item.
            canal_menos_usado = contagem_de_canais.most_common()[-1][0]
            ue.channel = canal_menos_usado

    return dist

def pr_interferente(ue1: UE, ue2: UE) -> float:
    if ue1.channel == ue2.channel:
        dist = np.linalg.norm(np.array([ue1.ap.x, ue1.ap.y]) - np.array([ue2.x, ue2.y]))

        if dist < 1:
            dist = 1

        x = np.random.lognormal(0, 2)
        return x*1e-4/(dist**4)
    return 0

def SINR(ue: UE, ues: list ,N: int) -> float:
    pt=1        # transmited power
    bt=1e8      # avaiable bandwidth
    k0=1e-20    # constant for the noise power

    pn = k0*bt/N
    pr = ue.gain * pt
    pr_int = np.sum([pr_interferente(ue, u) for u in ues if u!=ue])
    
    return pr/(pn + pr_int)

def channel_capacity(SINR: float, N: int) -> float:
    bt=100      # avaiable bandwidth
    return np.around((bt/N)*np.log2(1+SINR), 4)

def simular_experimento(M: int, N: int, sim: int, allocation: str = '') -> tuple:
    aps = distribuir_AP(M)  # Mantém a lista fixa de APs
    sinr = []
    cap_canal = []
    ues = []
    av_sum_cap = []

    for _ in range(sim):
        ues = [UE(N) for i in range(13)]
        sum_cap = 0
        for ue in ues:
            ue.dist = attach_AP_UE(ue, aps, allocation)
            
        for ue in ues:
            s = SINR(ue, ues, N)
            cap = channel_capacity(s, N)
            sum_cap += cap

            sinr.append(s)
            cap_canal.append(cap)
        av_sum_cap.append(sum_cap)

        for ap in aps:
            ap.ues = []  # Limpa a lista de UEs para o próximo experimento


    return sinr, cap_canal, np.mean(av_sum_cap)

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
        plt.plot(cdf[0], percentis, label=f'Random channel allocation')
        plt.plot(cdf[1], percentis, label=f'Per-AP channel allocation')

        plt.title(f'CDF do SINR por UE')
        plt.xlabel('SINR')

    else:
        for i in range(len(cdf)):
            # Ordena os valores da CDF
            cdf[i].sort()
            percentis = np.linspace(0, 1, len(cdf[i]))

        plt.plot(cdf[0], percentis, label=f'Random channel allocation')
        plt.plot(cdf[1], percentis, label=f'Per-AP channel allocation')
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

    sinr1, cap_canal1, av_sum_cap1 = simular_experimento(M, N, sim, 'random')
    print(f'Average sum capacity with random channel allocation: {av_sum_cap1:.2f} Mbps')
    print(f'10th percentile of SINR with random channel allocation: {np.percentile(sinr1, 10):.2f} dB')
    print(f'50th percentile of SINR with random channel allocation: {np.percentile(sinr1, 50):.2f} dB')
    print(f'10th percentile of channel capacity with random channel allocation: {np.percentile(cap_canal1, 10):.2f} Mbps')
    print(f'50th percentile of channel capacity with random channel allocation: {np.percentile(cap_canal1, 50):.2f} Mbps')

    sinr2, cap_canal2, av_sum_cap2 = simular_experimento(M, N, sim, 'per-AP')
    print(f'Average sum capacity with per-AP channel allocation: {av_sum_cap2:.2f} Mbps')
    print(f'10th percentile of SINR with per-AP channel allocation: {np.percentile(sinr2, 10):.2f} dB')
    print(f'50th percentile of SINR with per-AP channel allocation: {np.percentile(sinr2, 50):.2f} dB')
    print(f'10th percentile of channel capacity with per-AP channel allocation: {np.percentile(cap_canal2, 10):.2f} Mbps')
    print(f'50th percentile of channel capacity with per-AP channel allocation: {np.percentile(cap_canal2, 50):.2f} Mbps')

    plot_cdfs([sinr1, sinr2], 'sinr')
    plot_cdfs([cap_canal1, cap_canal2], 'capacity')

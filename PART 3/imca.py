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

def alocar_canais_ortogonal(access_points, ues, number_channels, allocation, G_slow=None, G_rayleigh=None):
    """
    Aloca canais. Agora suporta 'imca' se G_slow e G_rayleigh forem passados.
    """
    # Limpa alocações anteriores
    for ue in ues:
        ue.channel = None
    
    # Reinicia controle de uso dos APs
    for ap in access_points:
        ap.canais_usados = set()

    # --- MODO RANDOM ---
    if allocation == 'random':
        for ue in ues:
            ue.channel = np.random.randint(1, number_channels + 1)
        return

    # --- MODO IMCA (Exercício 7/8) ---
    if allocation == 'imca':
        # Lista embaralhada para justiça na alocação
        ues_shuffled = ues[:]
        random.shuffle(ues_shuffled)
        
        # Lista auxiliar para calcular interferência acumulada
        ues_ja_alocados = []
        
        bt = 1e8
        k0 = 1e-20
        pn = k0 * bt / number_channels

        for ue in ues_shuffled:
            ap = ue.ap
            
            # 1. Candidatos: Canais livres no AP (ou todos se estiver cheio)
            candidatos = [c for c in range(1, number_channels + 1) if c not in ap.canais_usados]
            if not candidatos:
                candidatos = list(range(1, number_channels + 1))
            
            melhor_sinr = -1
            melhor_canal = candidatos[0]
            
            # 2. Medição: Testa cada candidato
            for c in candidatos:
                c_idx = c - 1 # Índice para matriz (0 a N-1)
                
                # Sinal Útil Estimado (Slow * Fast do canal c)
                g_total = G_slow[ue.id, ap.id] * G_rayleigh[ue.id, ap.id, c_idx]
                S = ue.power * g_total
                
                # Interferência Medida (Soma dos UEs já alocados neste canal c)
                I = 0.0
                for outro in ues_ja_alocados:
                    if outro.channel == c:
                        # Ganho do interferente -> meu AP
                        g_int = G_slow[outro.id, ap.id] * G_rayleigh[outro.id, ap.id, c_idx]
                        I += outro.power * g_int
                
                sinr_est = S / (I + pn)
                
                if sinr_est > melhor_sinr:
                    melhor_sinr = sinr_est
                    melhor_canal = c
            
            # 3. Atribuição
            ue.channel = melhor_canal
            ap.canais_usados.add(melhor_canal)
            ues_ja_alocados.append(ue)
        return

    # --- MODO PAPOA (Default - Mantido igual) ---
    # (Seu código original do PAPOA aqui...)
    channels = {i: 0 for i in range(1, number_channels + 1)}
    for ap in access_points:
        available_channels = list(channels.keys())
        random.shuffle(available_channels)
        for ue0 in ap.ues:
            if not available_channels:
                break
            channel = available_channels.pop()
            ue0.channel = channel
            ap.canais_usados.add(channel) # Importante atualizar isso também
            channels[channel] += 1
            
    for ue in ues:
        if ue.channel is None:
            least = min(channels, key=channels.get)
            ue.channel = least
            channels[least] += 1

def attach_AP_UE(ues: list, aps: list, G: np.ndarray) -> float:
    for i, ue in enumerate(ues):
        # Encontra o AP com o maior ganho para este UE
        best_index = np.argmax(G[i, :])
        ue.ap = aps[best_index]
        ue.gain = G[i, best_index]
        ue.dist = np.linalg.norm(np.array([ue.x, ue.y]) - np.array([ue.ap.x, ue.ap.y]))
        aps[best_index].ues.append(ue)


def SINR(ues: list, N: int, G_effective: np.ndarray) -> list:
    bt=1e8
    k0=1e-20
    pn = k0*bt/N
    
    sinr_list = []
    # Cria mapas auxiliares para agilizar
    ue_power = np.array([u.power for u in ues])
    ue_channel = np.array([u.channel for u in ues])
    
    for k, ue_k in enumerate(ues):
        ap_servidor_id = ue_k.ap.id
        meu_canal = ue_k.channel
        
        # Sinal Útil
        S = G_effective[k, ap_servidor_id] * ue_k.power
        
        # Interferência: Soma de (G_effective[i, meu_ap] * P[i]) onde canal[i] == meu_canal
        # G_effective JÁ deve conter o ganho do canal que o usuário 'i' está usando.
        
        # Mascara: todos os UEs no mesmo canal, exceto eu
        mask = (ue_channel == meu_canal)
        mask[k] = False # Remove eu mesmo
        
        if np.any(mask):
            # Ganhos dos interferentes para o MEU AP
            gains_interferentes = G_effective[mask, ap_servidor_id]
            powers_interferentes = ue_power[mask]
            I = np.sum(gains_interferentes * powers_interferentes)
        else:
            I = 0.0
            
        sinr_list.append(S / (I + pn))
        
    return sinr_list

def channel_capacity(sinr_list: list, N: int) -> list:
    B_per_channel = 1e8 / N
    sinr_arr = np.array(sinr_list)
    # Shannon: B * log2(1 + SINR)
    return (B_per_channel * np.log2(1 + sinr_arr)) / 1e6 # Mbps

def simular_comparativo(M: int, N: int, sim: int):
    """
    Executa uma simulação comparativa justa (Shared Snapshot).
    Gera um cenário e testa os 3 algoritmos nele antes de passar para o próximo.
    """
    aps = distribuir_AP(M)
    
    # Dicionário para armazenar resultados de todos os métodos
    resultados = {
        'Random': {'sinr': [], 'cap': [], 'sum_cap': []},
        'PAPOA':  {'sinr': [], 'cap': [], 'sum_cap': []},
        'IMCA':   {'sinr': [], 'cap': [], 'sum_cap': []}
    }

    metodos = ['Random', 'PAPOA', 'IMCA']
    # Mapeamento para as strings que a função de alocação entende
    mapa_alloc = {'Random': 'random', 'PAPOA': '', 'IMCA': 'imca'}

    for _ in range(sim):
        # 1. GERAÇÃO DO CENÁRIO (ÚNICO PARA ESTA ITERAÇÃO)
        UE.id_counter = 0 
        ues = [UE(N) for i in range(13)] # Ajuste a quantidade de UEs aqui se necessário
        
        # Ganho Lento e Attachment (Comum a todos)
        G_slow = gain_matrix(ues, aps)
        attach_AP_UE(ues, aps, G_slow)
        
        # Ganho Rápido (Rayleigh) - Gerado UMA vez para este snapshot
        sigma_r = 1/np.sqrt(2)
        h_vals = np.random.normal(0, sigma_r, (len(ues), len(aps), N))**2 + \
                 np.random.normal(0, sigma_r, (len(ues), len(aps), N))**2
        G_rayleigh = h_vals

        # 2. RODADA DE TESTES (Competição)
        for metodo in metodos:
            # Importante: A função de alocação deve limpar o canal anterior
            # Passamos o MESMO G_slow e G_rayleigh para todos
            alocar_canais_ortogonal(aps, ues, N, mapa_alloc[metodo], G_slow, G_rayleigh)
            
            # 3. CÁLCULO DE MÉTRICAS (Baseado no canal escolhido)
            # Recria a matriz efetiva baseada na escolha do algoritmo atual
            G_effective = np.zeros_like(G_slow)
            for i, ue in enumerate(ues):
                c_idx = ue.channel - 1
                if c_idx < 0: c_idx = 0
                G_effective[i, :] = G_slow[i, :] * G_rayleigh[i, :, c_idx]
            
            # Calcula SINR e Capacidade
            s_linear = SINR(ues, N, G_effective)
            c_shannon = channel_capacity(s_linear, N)
            
            # Armazena os dados brutos (ou estatísticas)
            # Aqui estamos guardando TUDO para processar depois, ou você pode
            # guardar apenas médias/percentis para economizar memória.
            resultados[metodo]['sinr'].extend(s_linear)
            resultados[metodo]['cap'].extend(c_shannon)
            resultados[metodo]['sum_cap'].append(np.sum(c_shannon))

        # Limpeza dos APs para a próxima iteração do loop principal
        for ap in aps:
            ap.ues = [] 
            ap.canais_usados = set()

    return resultados

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
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
    Suporta: 'random', 'papoa' (default) e 'imca'.
    Para 'imca', G_slow e G_rayleigh são OBRIGATÓRIOS.
    """
    # 1. Limpa alocações anteriores
    for ue in ues:
        ue.channel = None
    for ap in access_points:
        ap.canais_usados = set() # Reinicia conjunto de canais usados

    # --- MODO RANDOM ---
    if allocation == 'random':
        for ue in ues:
            ue.channel = np.random.randint(1, number_channels + 1)
        return

    # --- MODO IMCA ---
    if allocation == 'imca':
        ues_shuffled = ues[:]
        random.shuffle(ues_shuffled)
        
        # Parâmetros de ruído (consistentes com a função SINR)
        bt = 1e8
        k0 = 1e-20
        pn = k0 * bt / number_channels # Ruído por canal

        ues_alocados = [] # Lista local para ajudar no cálculo de interferência

        for ue in ues_shuffled:
            ap = ue.ap
            
            # A. Candidatos (Ortogonalidade Local)
            candidatos = [c for c in range(1, number_channels + 1) if c not in ap.canais_usados]
            
            # Se lotou o AP, libera reuso total
            if not candidatos:
                candidatos = list(range(1, number_channels + 1))
            
            melhor_sinr = -1.0
            melhor_canal = candidatos[0] if candidatos else 1
            
            # B. Medição e Estimativa
            for c in candidatos:
                c_idx = c - 1 # 0-based para matriz
                
                # 1. Sinal Estimado (G_slow * G_rayleigh)
                # ue.id deve corresponder ao índice na matriz. 
                # Assumindo ue.id == índice i da lista original.
                g_total = G_slow[ue.id, ap.id] * G_rayleigh[ue.id, ap.id, c_idx]
                S = ue.power * g_total
                
                # 2. Interferência Medida (soma das potências dos UEs JÁ alocados neste canal c)
                I = 0.0
                for other_ue in ues_alocados:
                    if other_ue.channel == c:
                        # Ganho do interferente (other_ue) para o MEU AP (ap.id)
                        g_int = G_slow[other_ue.id, ap.id] * G_rayleigh[other_ue.id, ap.id, c_idx]
                        I += other_ue.power * g_int
                
                # 3. SINR Estimada
                sinr_est = S / (I + pn)
                
                if sinr_est > melhor_sinr:
                    melhor_sinr = sinr_est
                    melhor_canal = c
            
            # C. Atribuição Final
            ue.channel = melhor_canal
            ap.canais_usados.add(melhor_canal)
            ues_alocados.append(ue)
        return

    # --- MODO PAPOA (Default) ---
    # PAPOA Clássico: Ortogonalidade por AP apenas
    channels_global_count = {i: 0 for i in range(1, number_channels + 1)}
    
    for ap in access_points:
        available_channels = list(channels_global_count.keys())
        random.shuffle(available_channels)
        
        for ue in ap.ues:
            if not available_channels:
                break # Acabaram os canais neste AP (UEs excedentes ficam sem canal ou tratamos depois)
            
            channel = available_channels.pop()
            ue.channel = channel
            ap.canais_usados.add(channel)
            channels_global_count[channel] += 1
            
    # Cleanup (Preenchimento para UEs que sobraram, se houver)
    for ue in ues:
        if ue.channel is None:
            # Pega o canal menos usado no sistema (load balancing simples)
            least_used = min(channels_global_count, key=channels_global_count.get)
            ue.channel = least_used
            channels_global_count[least_used] += 1

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
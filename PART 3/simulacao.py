import numpy as np
import pandas as pd
import copy
import matplotlib.pyplot as plt
import seaborn as sns

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
    id_counter = 1
    def __init__(self, L: int):
        self.id = UE.id_counter
        UE.id_counter += 1
        self.x = np.random.randint(0, L+1)
        self.y = np.random.randint(0, L+1)
        self.L = L
        self.ap = None
        self.channel = 0
        self.dist = 0
        self.gain = 0
        self.power = 1  # Transmit power
        self.interference = []

    def __str__(self):
        return f'UE({self.x}, {self.y})'

    def __eq__(self, other):
        if isinstance(other, UE):
            return self.id == other.id
        return False

def distribuir_AP(M: int, L: int) -> list:
    APs = []
    dx = L/(2*np.sqrt(M))
    a = np.arange(dx, L+1-dx, 2*dx)
    x, y = np.meshgrid(a, a)
    id = 0
    for xi, yi in zip(x.ravel(), y.ravel()):
        APs.append(AP(xi, yi, id))
        id += 1
    return APs

def gain_matrix(ues: list, aps: list, L: int) -> np.ndarray:
    ''' Inicializa a matriz'''
    G = np.zeros((len(ues), len(aps)))
    ap_coords = np.array([[ap.x, ap.y] for ap in aps])

    for i, ue in enumerate(ues):
        # garante que o UE não está a menos de 1 m de nenhum AP
        while True:
            d_all = np.linalg.norm(ap_coords - np.array([ue.x, ue.y]), axis=1)
            if np.all(d_all >= 1):
                break
            ue.x = np.random.randint(0, L+1)
            ue.y = np.random.randint(0, L+1)

        # calcula ganhos para todos APs
        for j in range(len(aps)):
            ue_coord = np.array([ue.x, ue.y])
            d_ij = np.linalg.norm(ue_coord - ap_coords[j])
            d = max(d_ij, 1.0)
            G[i, j] = np.random.lognormal(0, 2) * 1e-4 / (d**4)
    return G

def attach_AP_UE(ues: list, aps: list, G: np.ndarray) -> float:
    for i, ue in enumerate(ues):
        # Encontra o AP com o maior ganho para este UE
        best_index = np.argmax(G[i, :])
        ue.ap = aps[best_index]     # associa o UE ao AP com maior ganho
        ue.gain = G[i, best_index]  # armazena o ganho para este UE
        ue.dist = np.linalg.norm(np.array([ue.x, ue.y]) - np.array([ue.ap.x, ue.ap.y]))     # armazena a distancia
        aps[best_index].ues.append(ue)      # adiciona o UE à lista de UEs do AP
    return None

def DPC(ues: list, N: int, t_max: int, G: np.ndarray, R: np.ndarray, y_tar: float, p_min: float, p_max: float, p_init: float) -> np.ndarray:
    # vetor de potencias
    p = np.ones((len(ues), t_max)) * p_init         # inicializa as potências com p_init para cada UE
    y = np.zeros((len(ues), t_max))                 # vetor para armazenar os SINRs ao longo das iterações
    pn = 1e8*1e-20/N
    iteracoes = 0

    for t in range(t_max):
        for i, ue in enumerate(ues):
            # Calcula a interferência total para este UE
            interference = sum([p[k][t] * G[k][ue.ap.id] * R[k][ue.ap.id] for k in range(len(ues)) if k != i and ues[k].channel == ue.channel])
            ue.interference.append(interference)

            # Calcula o SINR para este UE
            sinr = (p[i][t] * ue.gain * R[i][ue.ap.id]) / (interference + pn)
            y[i][t] = sinr

            # Atualiza a potência usando o algoritmo de controle de potência
            pt = min(max(p_min, p[i][t] * y_tar / sinr), p_max)

            if t < t_max - 1:  # Evita atualizar a potência na última iteração
                p[i][t+1] = pt

            #print(f'DPC: UE{i} - Iteração {t} - Potência: {p[i][t]:.4f} W - Ganho: {ue.gain} - SINR: {sinr:.4f} - Interferência: {interference:.4e}')

        iteracoes += 1
        if iteracoes > 5 and t < t_max - 1:
            soma_atual = np.sum(p[:, t+1])
            soma_anterior = np.sum(p[:, t])
            if abs(soma_atual - soma_anterior) < 1e-3:
                break
    return p[:, 0:iteracoes]

def maxsum(ues: list, passo: float, N: int, t_max: int, G: np.ndarray, R: np.ndarray, p_min: float, p_max: float, p_init: float, crit_parada: float) -> np.ndarray:
    # vetor de potencias
    p = np.ones((len(ues), t_max))          # inicializa as potências com 1W para cada UE
    p[:, 0] = p_init * np.ones(len(ues))    # define a potência inicial para cada UE
    y = np.zeros((len(ues), t_max))         # vetor para armazenar os SINRs ao longo das iterações
    interferences = np.zeros((len(ues), t_max))
    pn = 1e8*1e-20/N
    iteracoes = 0

    for t in range(t_max):
        for i in range(len(ues)):
            # calcula a interferência total para cada UE
            interferences[i][t] = sum([p[k][t] * G[k][ues[i].ap.id] * R[k][ues[i].ap.id] for k in range(len(ues)) if k != i]) + pn

        # calcula o SINR para cada UE
        for i, ue in enumerate(ues):
            S_i = p[i, t] * ue.gain * R[i, ue.ap.id]  # Sinal útil
            y[i, t] = S_i / interferences[i, t]

        # loop principal para calcular a potência
        for k, ue in enumerate(ues):
            # Calcula a interferência total para este UE
            I_k = interferences[k][t]
            ue.interference.append(I_k)

            # path loss e fading
            g_k = G[k][ue.ap.id]*R[k][ue.ap.id]

            # somatório de y_i/I_i para i != k
            somatorio = sum([y[i][t] / interferences[i][t] for i in range(len(ues)) if i != k])

            # Atualiza a potência usando o algoritmo de controle de potência
            pt = min(max(p_min, p[k][t] + passo*g_k*((1/I_k) - somatorio)), p_max)

            if t < t_max - 1:  # Evita atualizar a potência na última iteração
                p[k][t+1] = pt
        
            #print(f'MaxSum: UE{k} - Iteração {t} - Potência: {p[k][t]:.4f} W - Ganho: {ue.gain} - SINR: {y[k][t]:.4f} - Interferência: {interferences[k][t]:.4e}')

        iteracoes += 1
        if t>=5 and t<t_max-1:
            soma_atual = np.sum(p[:, t+1])
            soma_anterior = np.sum(p[:, t])

            if abs(soma_atual - soma_anterior) < crit_parada:
                break
    return p[:, 0:iteracoes]

def maxprod(ues: list, passo: float, N: int, t_max: int, G: np.ndarray, R: np.ndarray, p_min: float, p_max: float, p_init: float, crit_parada: float) -> np.ndarray:
    # vetor de potencias
    p = np.ones((len(ues), t_max))          # inicializa as potências com 1W para cada UE
    p[:, 0] = p_init * np.ones(len(ues))    # define a potência inicial para cada UE
    y = np.zeros((len(ues), t_max))         # vetor para armazenar os SINRs ao longo das iterações
    interferences = np.zeros((len(ues), t_max))
    pn = 1e8*1e-20/N
    iteracoes = 0

    for t in range(t_max):
        for i in range(len(ues)):
            interferences[i][t] = sum([p[k][t] * G[k][ues[i].ap.id] * R[k][ues[i].ap.id] for k in range(len(ues)) if k != i]) + pn

        for k, ue in enumerate(ues):
            # Calcula a interferência total para este UE
            I_k = interferences[k][t]
            ue.interference.append(I_k)

            # Calcula o SINR para este UE
            sinr = (p[k][t] * ue.gain * R[k][ue.ap.id]) / I_k
            y[k][t] = sinr

            # somatório de 1/I_i para i != k
            sum_I_j = 0
            for j in range(len(ues)):
                if j != k:
                    sum_I_j += G[j][ues[k].ap.id]*R[j][ues[k].ap.id]/interferences[j][t]

            # Atualiza a potência usando o algoritmo de controle de potência
            pt = min(max(p_min, p[k][t] + passo*((G[k][ue.ap.id]*R[k][ue.ap.id]/(sinr*I_k)) - sum_I_j)), p_max)

            if t < t_max - 1:  # Evita atualizar a potência na última iteração
                p[k][t+1] = pt

            #print(f'MaxProd: UE{k} - Iteração {t} - Potência: {p[k][t]:.4f} W - Ganho: {ue.gain} - SINR: {y[k][t]:.4f} - Interferência: {interferences[k][t]:.4e}')

        iteracoes += 1
        if t>5 and t<t_max-1:
            soma_atual = np.sum(p[:, t+1])
            soma_anterior = np.sum(p[:, t])

            if abs(soma_atual - soma_anterior) < crit_parada:
                break
    return p[:, 0:iteracoes]

def SINR(ues: list, N: int, G: np.ndarray, R: np.ndarray) -> float:
    bt=1e8      # avaiable bandwidth
    k0=1e-20    # constant for the noise power

    # noise power
    pn = k0*bt/N

    # vetor de potência
    P = np.array([ue.power for ue in ues])

    # variável para armazenar os valores de SINR
    sinr = []

    for i, ue in enumerate(ues):
        # para cada UE, calcula o sinal útil e a interferência total dos outros UEs no mesmo canal
        s = ue.gain * P[i] * R[i, ue.ap.id]  # sinal útil

        # interferência total de outros UEs no mesmo canal
        interference = 0.0

        # itera sobre todos os outros UEs do sistema
        for j, outro_ue in enumerate(ues):
            # se outro UE estiver transmitindo no mesmo canal e for diferente do UE atual, acumula a interferência
            if j != i and outro_ue.channel == ue.channel:
                interference += G[j, ue.ap.id] * P[j] * R[j, ue.ap.id]

        sinr.append(s / (interference + pn))

    return sinr

def channel_capacity(sinr: list, N: int) -> list:
    B_per_channel = 1e8 / N
    sinr_array = np.asarray(sinr)
    capacity_mbps = B_per_channel * np.log2(1 + sinr_array) / 1e6

    return capacity_mbps


def plotar_convergencia_potencia(df_potencias, id_simulacao=0):
    """
    Plota a evolução da potência de cada UE ao longo das iterações para uma simulação específica.
    Cria um subgráfico para cada algoritmo testado.
    """
    # 1. Filtra os dados apenas para a simulação desejada
    df_sim = df_potencias[df_potencias['ID_Simulacao'] == id_simulacao]

    # mudando o ID_UE para começar de 1 ao invés de 0
    #UE_labels = {1: 'UE1', 2: 'UE2', 3: 'UE3', 4: 'UE4'}
    #df_sim['UE'] = 'UE' + (df_sim['ID_UE'] + 1).astype(str)
    df_sim['ID_UE'] = df_sim['ID_UE'] + 1
    UE_labels = {ue_id: f'UE{ue_id}' for ue_id in sorted(df_sim['ID_UE'].unique())}
    df_sim['UE'] = df_sim['ID_UE'].map(UE_labels)
    
    # 2. Identifica quais algoritmos estão no DataFrame
    algoritmos = df_sim['Nome_Algoritmo'].unique()
    n_algs = len(algoritmos)
    
    # 3. Cria uma figura com gráficos lado a lado (um para cada algoritmo)
    fig, axes = plt.subplots(1, n_algs, figsize=(6 * n_algs, 5), sharey=True)
    
    # Ajuste técnico caso haja apenas 1 algoritmo (axes não será uma lista)
    if n_algs == 1:
        axes = [axes]
        
    # 4. Desenha as linhas de cada UE no gráfico correspondente ao algoritmo
    for ax, alg in zip(axes, algoritmos):
        df_alg = df_sim[df_sim['Nome_Algoritmo'] == alg]
        
        # O parâmetro 'hue' separa automaticamente uma linha de cor diferente para cada UE
        sns.lineplot(data=df_alg, x='Iteracao', y='Potencia', hue='ID_UE', palette='tab10', ax=ax)
        
        ax.set_title(f'Convergência - {alg}')
        ax.set_xlabel('Iteração')
        ax.set_ylabel('Potência (W)')
        ax.grid(True, linestyle='--', alpha=0.7)
        
    plt.tight_layout()
    plt.show()


def plotar_cdfs(df_metricas):
    """
    Gera três gráficos de CDF lado a lado comparando o desempenho final dos algoritmos.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # --- Gráfico 1: CDF do SINR (em dB) ---
    sns.ecdfplot(data=df_metricas, x='SINR_dB', hue='Nome_Algoritmo', linewidth=2, ax=axes[0])
    axes[0].set_title('CDF do SINR')
    axes[0].set_xlabel('SINR (dB)')
    axes[0].set_ylabel('Probabilidade Acumulada')
    axes[0].grid(True, linestyle='--', alpha=0.7)

    # --- Gráfico 2: CDF da Capacidade de Canal (por Usuário) ---
    sns.ecdfplot(data=df_metricas, x='Capacidade_Canal', hue='Nome_Algoritmo', linewidth=2, ax=axes[1])
    axes[1].set_title('CDF da Capacidade por Usuário')
    axes[1].set_xlabel('Capacidade (Mbps)')
    axes[1].set_ylabel('Probabilidade Acumulada')
    axes[1].grid(True, linestyle='--', alpha=0.7)

    # --- Gráfico 3: CDF da Capacidade de Soma (Total do Sistema) ---
    # Removemos as duplicatas porque a capacidade de soma é a mesma para todos os UEs da mesma simulação
    df_soma = df_metricas[['ID_Simulacao', 'Nome_Algoritmo', 'Capacidade_Soma']].drop_duplicates()
    
    sns.ecdfplot(data=df_soma, x='Capacidade_Soma', hue='Nome_Algoritmo', linewidth=2, ax=axes[2])
    axes[2].set_title('CDF da Capacidade Total do Sistema')
    axes[2].set_xlabel('Capacidade Total (Mbps)')
    axes[2].set_ylabel('Probabilidade Acumulada')
    axes[2].grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.show()

def comparar_10_percentil(df_metricas):
    """
    Calcula e plota o 10º percentil do SINR (dB), da Capacidade de Canal e da Capacidade de Soma
    para avaliar o desempenho dos piores cenários/usuários em cada algoritmo.
    """
    
    # 1. Agrupar por algoritmo e calcular o percentil 0.10 (10%) para as três métricas
    df_10th = df_metricas.groupby('Nome_Algoritmo')[['SINR_dB', 'Capacidade_Canal', 'Capacidade_Soma']].quantile(0.10).reset_index()
    
    # 2. Exibir os valores numéricos exatos no terminal para o seu registro
    print("=== Comparação 10th percentil ===")
    print(df_10th.to_string(index=False))
    print("================================================\n")

    # 3. Preparar a figura com 3 subgráficos (aumentei a largura para 18 para acomodar bem os 3)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Dicionário de cores padronizado
    cores = {'DPC': '#1f77b4', 'MaxSum': '#ff7f0e', 'MaxProd': '#2ca02c'}
    
    # --- Gráfico 1: 10th Percentil do SINR (dB) ---
    ordem_sinr = df_10th.sort_values('SINR_dB', ascending=False)['Nome_Algoritmo']
    
    sns.barplot(data=df_10th, x='Nome_Algoritmo', y='SINR_dB', 
                hue='Nome_Algoritmo', legend=False,
                order=ordem_sinr, palette=cores, ax=axes[0])
    axes[0].set_title('10º Percentil do SINR')
    axes[0].set_ylabel('SINR (dB)')
    axes[0].set_xlabel('Algoritmo')
    axes[0].grid(axis='y', linestyle='--', alpha=0.7)

    # --- Gráfico 2: 10th Percentil da Capacidade ---
    ordem_cap = df_10th.sort_values('Capacidade_Canal', ascending=False)['Nome_Algoritmo']
    
    sns.barplot(data=df_10th, x='Nome_Algoritmo', y='Capacidade_Canal', 
                hue='Nome_Algoritmo', legend=False,
                order=ordem_cap, palette=cores, ax=axes[1])
    axes[1].set_title('10º Percentil da Capacidade')
    axes[1].set_ylabel('Capacidade (Mbps)')
    axes[1].set_xlabel('Algoritmo')
    axes[1].grid(axis='y', linestyle='--', alpha=0.7)

    # --- Gráfico 3: 10th Percentil da Capacidade de Soma ---
    # Ordenando do maior (melhor) para o menor
    ordem_soma = df_10th.sort_values('Capacidade_Soma', ascending=False)['Nome_Algoritmo']
    
    sns.barplot(data=df_10th, x='Nome_Algoritmo', y='Capacidade_Soma', 
                hue='Nome_Algoritmo', legend=False,
                order=ordem_soma, palette=cores, ax=axes[2])
    axes[2].set_title('10º Percentil da Capacidade de Soma')
    axes[2].set_ylabel('Capacidade Total (Mbps)')
    axes[2].set_xlabel('Algoritmo')
    axes[2].grid(axis='y', linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.show()

def comparar_mediana(df_metricas):
    """
    Calcula e plota a mediana do SINR (dB), da Capacidade de Canal e da Capacidade de Soma
    para avaliar o desempenho dos piores usuários em cada algoritmo.
    """
    
    # 1. Agrupar por algoritmo e calcular o percentil 0.50 (50%) para as três métricas
    df_50th = df_metricas.groupby('Nome_Algoritmo')[['SINR_dB', 'Capacidade_Canal', 'Capacidade_Soma']].quantile(0.50).reset_index()
    
    # 2. Exibir os valores numéricos exatos no terminal para o seu registro
    print("=== Comparação mediana ===")
    print(df_50th.to_string(index=False))
    print("================================================\n")

    # 3. Preparar a figura com 3 subgráficos (aumentei a largura para 18 para acomodar bem os 3)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Dicionário de cores padronizado
    cores = {'DPC': '#1f77b4', 'MaxSum': '#ff7f0e', 'MaxProd': '#2ca02c'}
    
    # --- Gráfico 1: 50th Percentil do SINR (dB) ---
    ordem_sinr = df_50th.sort_values('SINR_dB', ascending=False)['Nome_Algoritmo']
    
    sns.barplot(data=df_50th, x='Nome_Algoritmo', y='SINR_dB', 
                hue='Nome_Algoritmo', legend=False,
                order=ordem_sinr, palette=cores, ax=axes[0])
    axes[0].set_title('Mediana do SINR')
    axes[0].set_ylabel('SINR (dB)')
    axes[0].set_xlabel('Algoritmo')
    axes[0].grid(axis='y', linestyle='--', alpha=0.7)

    # --- Gráfico 2: 50th Percentil da Capacidade ---
    ordem_cap = df_50th.sort_values('Capacidade_Canal', ascending=False)['Nome_Algoritmo']
    
    sns.barplot(data=df_50th, x='Nome_Algoritmo', y='Capacidade_Canal', 
                hue='Nome_Algoritmo', legend=False,
                order=ordem_cap, palette=cores, ax=axes[1])
    axes[1].set_title('Mediana da Capacidade')
    axes[1].set_ylabel('Capacidade (Mbps)')
    axes[1].set_xlabel('Algoritmo')
    axes[1].grid(axis='y', linestyle='--', alpha=0.7)

    # --- Gráfico 3: 50th Percentil da Capacidade de Soma ---
    # Ordenando do maior (melhor) para o menor
    ordem_soma = df_50th.sort_values('Capacidade_Soma', ascending=False)['Nome_Algoritmo']
    
    sns.barplot(data=df_50th, x='Nome_Algoritmo', y='Capacidade_Soma', 
                hue='Nome_Algoritmo', legend=False,
                order=ordem_soma, palette=cores, ax=axes[2])
    axes[2].set_title('Mediana da Capacidade de Soma')
    axes[2].set_ylabel('Capacidade Total (Mbps)')
    axes[2].set_xlabel('Algoritmo')
    axes[2].grid(axis='y', linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.show()

def plot_efiencia_energia(df_metricas):
    """
    Calcula a Eficiência Energética (Mbps/W) de cada UE e gera os gráficos 
    comparativos de CDF, 10º Percentil e Mediana para os algoritmos.
    """
    
    # 1. Calcula a Eficiência Energética individual (Capacidade / Potência)
    df_ee = df_metricas.copy()
    df_ee['Eficiencia_Energetica'] = df_ee['Capacidade_Canal'] / df_ee['Potencia']
    
    # 2. Calcula o 10º Percentil (0.10) e a Mediana (0.50)
    df_10th = df_ee.groupby('Nome_Algoritmo')['Eficiencia_Energetica'].quantile(0.10).reset_index()
    df_10th.rename(columns={'Eficiencia_Energetica': 'EE_10th'}, inplace=True)
    
    df_median = df_ee.groupby('Nome_Algoritmo')['Eficiencia_Energetica'].median().reset_index()
    df_median.rename(columns={'Eficiencia_Energetica': 'EE_Mediana'}, inplace=True)
    
    # Mescla as duas tabelas para exibir um resumo limpo no terminal
    df_resumo = pd.merge(df_10th, df_median, on='Nome_Algoritmo')
    print("=== Eficiência Energética (bits/Joule) ===")
    print(df_resumo.to_string(index=False))
    print("================================================\n")

    # 3. Prepara a figura com 3 subgráficos lado a lado
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Dicionário de cores para manter o padrão visual
    cores = {'DPC': '#1f77b4', 'MaxSum': '#ff7f0e', 'MaxProd': '#2ca02c'}
    
    # --- Gráfico 1: CDF da Eficiência Energética ---
    sns.ecdfplot(data=df_ee, x='Eficiencia_Energetica', hue='Nome_Algoritmo', palette=cores, linewidth=2, ax=axes[0])
    axes[0].set_title('CDF da Eficiência Energética')
    axes[0].set_xlabel('Eficiência Energética (Mbps/W)')
    axes[0].set_ylabel('Probabilidade Acumulada')
    axes[0].grid(True, linestyle='--', alpha=0.7)

    # --- Gráfico 2: 10th Percentil (Piores Usuários) ---
    ordem_10th = df_10th.sort_values('EE_10th', ascending=False)['Nome_Algoritmo']
    
    sns.barplot(data=df_10th, x='Nome_Algoritmo', y='EE_10th', 
                hue='Nome_Algoritmo', legend=False,
                order=ordem_10th, palette=cores, ax=axes[1])
    axes[1].set_title('10º Percentil da Eficiência Energética')
    axes[1].set_ylabel('Eficiência Energética (Mbps/W)')
    axes[1].set_xlabel('Algoritmo')
    axes[1].grid(axis='y', linestyle='--', alpha=0.7)

    # --- Gráfico 3: Mediana (Usuário Intermediário) ---
    ordem_median = df_median.sort_values('EE_Mediana', ascending=False)['Nome_Algoritmo']
    
    sns.barplot(data=df_median, x='Nome_Algoritmo', y='EE_Mediana', 
                hue='Nome_Algoritmo', legend=False,
                order=ordem_median, palette=cores, ax=axes[2])
    axes[2].set_title('Mediana da Eficiência Energética')
    axes[2].set_ylabel('Eficiência Energética (bits/Joule)')
    axes[2].set_xlabel('Algoritmo')
    axes[2].grid(axis='y', linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.show()

def simular_experimento(printar_convergencia: bool, cenario: str, num_simulacoes: int = 1000, max_iteracoes: int = 2000, crit_parada_maxsum: float = 1e-3, crit_parada_maxprod: float = 1e-3, p_init: float = 1.0, passo_maxsum: float = 0.1, passo_maxprod: float = 0.1, sinr_target: float = 1.0, M: int = 4, K: int = 4) -> pd.DataFrame:
    N = 1
    p_max, p_min = 1.0, 0.001
    t_limite_maximo = max_iteracoes
    
    if cenario == 'noise':
        L = 1000
    elif cenario == 'interference':
        L = 100

    nomes_algoritmos = ['DPC', 'MaxSum', 'MaxProd']
    
    # Listas globais para guardar TODOS os dados
    lista_historico_geral = []
    lista_metricas_finais = []

    for sim_idx in range(num_simulacoes):
        UE.id_counter = 1  # Reinicia o contador de IDs para cada simulação
        aps = distribuir_AP(M, L)
        ues = [UE(L) for _ in range(K)]
        G = gain_matrix(ues, aps, L)
        R = (np.random.rayleigh(scale=1/np.sqrt(2), size=(K, M)))**2
        attach_AP_UE(ues, aps, G)

        # Lista TEMPORÁRIA para guardar a potência apenas desta simulação
        lista_historico_atual = [] 

        for nome_alg in nomes_algoritmos:
            # Usando cópias para garantir que cada algoritmo começa com as mesmas condições iniciais
            ues_copia = copy.deepcopy(ues)
            # 1. Executa os algoritmos
            if nome_alg == 'DPC':
                p_historico = DPC(ues_copia, N, t_limite_maximo, G, R, sinr_target, p_min, p_max, p_init)
                print('\n')

            elif nome_alg == 'MaxSum':
                p_historico = maxsum(ues_copia, passo_maxsum, N, t_limite_maximo, G, R, p_min, p_max, p_init, crit_parada_maxsum)
                print('\n')
                
            elif nome_alg == 'MaxProd':
                p_historico = maxprod(ues_copia, passo_maxprod, N, t_limite_maximo, G, R, p_min, p_max, p_init, crit_parada_maxprod)
                print('\n')
            
            iteracoes_reais = p_historico.shape[1]
            
            # 2. Salva o histórico de potência desta simulação
            for ue_idx in range(K):
                for t in range(iteracoes_reais):
                    lista_historico_atual.append({
                        'ID_Simulacao': sim_idx,
                        'ID_UE': ue_idx,
                        'Iteracao': t,
                        'Nome_Algoritmo': nome_alg,
                        'Potencia': p_historico[ue_idx, t]
                    })
            
            # 3. Salva as métricas finais (normal)
            p_final = p_historico[:, iteracoes_reais - 1] 
            for i, ue in enumerate(ues_copia): ue.power = p_final[i]
            
            sinr_final = SINR(ues_copia, N, G, R)
            cap_final = channel_capacity(sinr_final, N)
            cap_soma = np.sum(cap_final)
            
            for i, ue in enumerate(ues_copia):
                lista_metricas_finais.append({
                    'ID_Simulacao': sim_idx,
                    'ID_UE': i,
                    'Nome_Algoritmo': nome_alg,
                    'SINR_Linear': float(sinr_final[i]),
                    'SINR_dB': float(10 * np.log10(sinr_final[i])),
                    'Capacidade_Canal': float(cap_final[i]),
                    'Capacidade_Soma': float(cap_soma),
                    'Potencia': float(p_final[i])
                })
                
        for ap in aps: ap.ues = []
        
        # A. Transfere os dados atuais para a lista geral
        lista_historico_geral.extend(lista_historico_atual)
        
        # B. Transforma a lista atual em DataFrame 
        df_atual = pd.DataFrame(lista_historico_atual)
        
        # Chama função de plotagem
        if printar_convergencia:
            plotar_convergencia_potencia(df_atual, id_simulacao=sim_idx)

    # ==========================================
    # FIM DE TODAS AS SIMULAÇÕES MONTE CARLO
    # ==========================================
    
    df_potencias = pd.DataFrame(lista_historico_geral)
    df_metricas = pd.DataFrame(lista_metricas_finais)
    
    # Chama a função para plotar as CDFs finais
    #plotar_cdfs(df_metricas)

    #comparar_10_percentil(df_metricas)
    
    return df_metricas

'''
if __name__ == "__main__":
    df_potencias, df_metricas = simular_experimento(cenario='interference', num_simulacoes=10, max_iteracoes=2000, crit_parada_maxsum=1e-3, crit_parada_maxprod=1e-3, p_init=1.0, passo_maxsum=0.1, passo_maxprod=0.1, sinr_target=1.0, M=4, K=4)
    plotar_cdfs(df_metricas)
    comparar_10_percentil(df_metricas)
    comparar_mediana(df_metricas)
    plot_efiencia_energia(df_metricas)'''
import numpy as np
import pandas as pd
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
            interference = sum([p[k][t] * G[k][ue.ap.id] * R[k][ue.ap.id] for k in range(len(ues)) if k != i])
            ue.interference.append(interference)

            # Calcula o SINR para este UE
            sinr = (p[i][t] * ue.gain * R[i][ue.ap.id]) / (interference + pn)
            y[i][t] = sinr

            # Atualiza a potência usando o algoritmo de controle de potência
            pt = min(max(p_min, p[i][t] * y_tar / sinr), p_max)

            if t < t_max - 1:  # Evita atualizar a potência na última iteração
                p[i][t+1] = pt

        soma_atual = np.sum(p[:, t+1])
        soma_anterior = np.sum(p[:, t])
        iteracoes += 1

        if iteracoes > 5:
            if abs(soma_atual - soma_anterior) < 1e-3:
                break
    return p[:, 0:iteracoes]

def maxsum(ues: list, passo: float, N: int, t_max: int, G: np.ndarray, R: np.ndarray, p_min: float, p_max: float, p_init: float) -> np.ndarray:
    # vetor de potencias
    t_max = 5000
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

        iteracoes += 1
        if t>=5:
            soma_atual = np.sum(p[:, t])
            soma_anterior = np.sum(p[:, t-1])

            if abs(soma_atual - soma_anterior) < 1e-3:
                break
    return p[:, 0:iteracoes]

def maxprod(ues: list, passo: float, N: int, t_max: int, G: np.ndarray, R: np.ndarray, p_min: float, p_max: float, p_init: float) -> np.ndarray:
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
                    sum_I_j += G[k][ues[j].ap.id]*R[k][ues[j].ap.id]/interferences[j][t]

            # Atualiza a potência usando o algoritmo de controle de potência
            pt = min(max(p_min, p[k][t] + passo*((G[k][ue.ap.id]*R[k][ue.ap.id]/(sinr*I_k)) - sum_I_j)), p_max)

            if t < t_max - 1:  # Evita atualizar a potência na última iteração
                p[k][t+1] = pt

        iteracoes += 1
        if t>5:
            soma_atual = np.sum(p[:, t])
            soma_anterior = np.sum(p[:, t-1])

            if abs(soma_atual - soma_anterior) < 1e-4:
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
                interference += G[j, ue.ap.id] * P[j] * R[i, ue.ap.id]

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
    Calcula e plota o 10º percentil do SINR (dB) e da Capacidade de Canal
    para avaliar o desempenho dos piores usuários em cada algoritmo.
    """
    
    # 1. Agrupar por algoritmo e calcular o percentil 0.10 (10%)
    df_10th = df_metricas.groupby('Nome_Algoritmo')[['SINR_dB', 'Capacidade_Canal']].quantile(0.10).reset_index()
    
    # 2. Exibir os valores numéricos exatos no terminal para o seu registro
    print(df_10th.to_string(index=False))

    # 3. Preparar a figura com 2 subgráficos
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Dicionário de cores padronizado
    cores = {'DPC': '#1f77b4', 'MaxSum': '#ff7f0e', 'MaxProd': '#2ca02c'}
    
    # --- Gráfico 1: 10th Percentil do SINR (dB) ---
    # Ordenando do maior (melhor) para o menor
    ordem_sinr = df_10th.sort_values('SINR_dB', ascending=False)['Nome_Algoritmo']
    
    sns.barplot(data=df_10th, x='Nome_Algoritmo', y='SINR_dB', 
                hue='Nome_Algoritmo', legend=False,
                order=ordem_sinr, palette=cores, ax=axes[0])
    axes[0].set_title('10º Percentil do SINR')
    axes[0].set_ylabel('SINR (dB)')
    axes[0].set_xlabel('Algoritmo')
    axes[0].grid(axis='y', linestyle='--', alpha=0.7)

    # --- Gráfico 2: 10th Percentil da Capacidade ---
    # Ordenando do maior (melhor) para o menor
    ordem_cap = df_10th.sort_values('Capacidade_Canal', ascending=False)['Nome_Algoritmo']
    
    sns.barplot(data=df_10th, x='Nome_Algoritmo', y='Capacidade_Canal', 
                hue='Nome_Algoritmo', legend=False,
                order=ordem_cap, palette=cores, ax=axes[1])
    axes[1].set_title('10º Percentil da Capacidade')
    axes[1].set_ylabel('Capacidade (Mbps)')
    axes[1].set_xlabel('Algoritmo')
    axes[1].grid(axis='y', linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.show()

def simular_experimento(cenario: str, num_simulacoes: int = 1000, p_init: float = 1.0, passo: float = 0.1, M: int = 4, K: int = 4) -> pd.DataFrame:
    N = 1
    p_max, p_min, y_tar = 1.0, 0.001, 1
    t_limite_maximo = 2000
    L = 1000 if cenario == 'noise' else 100

    nomes_algoritmos = ['DPC', 'MaxSum', 'MaxProd']
    
    # Listas globais para guardar TODOS os dados
    lista_historico_geral = []
    lista_metricas_finais = []

    for sim_idx in range(num_simulacoes):
        aps = distribuir_AP(M, L)
        ues = [UE() for _ in range(K)]
        G = gain_matrix(ues, aps)
        R = (np.random.rayleigh(scale=1/np.sqrt(2), size=(K, M)))**2
        attach_AP_UE(ues, aps, G)

        # Lista TEMPORÁRIA para guardar a potência apenas desta simulação
        lista_historico_atual = [] 

        for nome_alg in nomes_algoritmos:
            
            # 1. Executa os algoritmos (mantive os fictícios para o exemplo não quebrar)
            if nome_alg == 'DPC':
                p_historico = DPC(ues, N, t_limite_maximo, G, R, y_tar, p_min, p_max, p_init)
            elif nome_alg == 'MaxSum':
                p_historico = maxsum(ues, passo, N, t_limite_maximo, G, R, p_min, p_max, p_init)
            elif nome_alg == 'MaxProd':
                p_historico = maxprod(ues, passo, N, t_limite_maximo, G, R, p_min, p_max, p_init)
            
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
            for i, ue in enumerate(ues): ue.power = p_final[i]
                
            sinr_final = SINR(ues, N, G, R)
            cap_final = channel_capacity(sinr_final, N)
            cap_soma = np.sum(cap_final)
            
            for i, ue in enumerate(ues):
                lista_metricas_finais.append({
                    'ID_Simulacao': sim_idx,
                    'ID_UE': ue.id,
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
        plotar_convergencia_potencia(df_atual, id_simulacao=sim_idx)

    # ==========================================
    # FIM DE TODAS AS SIMULAÇÕES MONTE CARLO
    # ==========================================
    
    df_potencias = pd.DataFrame(lista_historico_geral)
    df_metricas = pd.DataFrame(lista_metricas_finais)
    
    # Chama a função para plotar as CDFs finais
    plotar_cdfs(df_metricas)

    comparar_10_percentil(df_metricas)
    
'''    return df_potencias, df_metricas

if __name__ == "__main__":
    # Roda a simulação completa
    df_potencias, df_metricas = simular_experimento(cenario='noise', num_simulacoes=5, p_init=1, passo=1e-3)'''
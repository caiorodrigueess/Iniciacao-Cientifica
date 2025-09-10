'''import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

class AP:
    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y
        self.id = id
        self.channel = None
        self.ues = []

    def __str__(self):
        return f'AP[{self.id}]({self.x}, {self.y})'

class UE:
    id_counter = 0
    def __init__(self, x: float, y: float):
        self.id = UE.id_counter
        UE.id_counter += 1
        self.x = x
        self.y = y
        self.ap = None
        self.channel = 0
        self.dist = 0
        self.gain = 0
        self.ca = False
        self.sinr = 0

    def __str__(self):
        return f'UE({self.x}, {self.y})'
    
    def __eq__(self, other):
        if isinstance(other, UE):
            return self.id == other.id
        return False

def attach_AP_UE(ue: UE, aps: list) -> float:
    ap_coords = np.array([[ap.x, ap.y] for ap in aps])

    # Cálculo vetorizado do ganho
    x = np.random.lognormal(0, 2, len(aps))
    distances = np.linalg.norm(ap_coords - np.array([ue.x, ue.y]), axis=1)
    gains = x * 1e-4 / (distances ** 4)
    
    return gains

def simular_experimento():
    ap1 = AP(0, 0)
    ap2 = AP(3100, 0)
    ue1 = UE(1, 0)
    gains = np.zeros((2, 3099))

    for n in range(3099):
        gains1, gains2 = attach_AP_UE(ue1, [ap1, ap2])
        ue1.x += 1
        gains[0, n] = gains1
        gains[1, n] = gains2

    return gains

def plot_gain(gains: np.ndarray):

    x = np.arange(len(gains[0]))
    points = np.array([x, gains[0]]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    cores = ["red" if (yi < 1e-17 or yj < 1e-17) else "blue" for yi, yj in zip(gains[0][:-1], gains[0][1:])]
    lc = LineCollection(segments, colors=cores, linewidths=0.35)

    plt.semilogy(x[~mask1], gains[0][~mask1], linewidth = 0.35, color='#1F4DE6', label='AP1')
    plt.semilogy(x[mask1], gains[0][mask1], linewidth = 0.35, color="#FF0000DF")

    mask2 = gains[1] <= 1e-17
    plt.semilogy(x[~mask2], gains[1][~mask2], linewidth = 0.35, color="#F16717", label='AP2')
    plt.semilogy(x[mask2], gains[1][mask2], linewidth = 0.35, color="#FF0000DF")

    fig, ax = plt.subplots()
    ax.set_yscale('log')
    ax.add_collection(lc)
    ax.axhline(1e-17, color='red', linestyle='--', linewidth=1, label='Threshold (10^-17)')
    ax.set_xlabel('Position (m)')
    ax.set_ylabel('Gain')
    ax.set_title('Gain of each AP for UE')
    ax.grid()
    ax.legend()
    plt.show()
    
if __name__ == "__main__":

    gains = simular_experimento()
    plot_gain(gains)'''

import numpy as np
import matplotlib.pyplot as plt

# --- Parâmetros da Simulação (baseado no Exercício 11) ---
DISTANCIA_APs = 3100.0  # Distância total entre APs em metros
VELOCIDADE_UE = 20.0     # Velocidade do UE em m/s
T_S = 1.0                # Período de amostragem em segundos (Ts)
P_MIN = 1e-17            # Limiar de potência para outage em Watts
TRAJETO_INICIO = 50.0    # UE começa a 50m do AP1
TRAJETO_FIM = DISTANCIA_APs - 50.0 # UE termina a 50m do AP2

# --- Parâmetros do Canal ---
K_CANAL = 1e-4           # Constante de propagação k
N_CANAL = 4              # Expoente de perda de percurso n
SIGMA_X = 2              # Desvio padrão do sombreamento (shadowing)

class AP:
    def __init__(self, x: float, y: float, id_ap: int):
        self.x = x
        self.y = y
        self.id = id_ap

    def __str__(self):
        return f'AP[{self.id}]({self.x}, {self.y})'

class UE:
    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y

    def __str__(self):
        return f'UE({self.x}, {self.y})'

def calcular_ganho_canal(ue: UE, aps: list) -> np.ndarray:
    """
    Calcula o ganho de canal do UE para uma lista de APs, incluindo o sombreamento.
    Retorna um array com os ganhos.
    """
    ap_coords = np.array([[ap.x, ap.y] for ap in aps])
    ue_coords = np.array([ue.x, ue.y])

    # Sorteia uma variável lognormal para o sombreamento para cada AP
    x_shadowing = np.random.lognormal(0, SIGMA_X, len(aps))
    
    # Calcula as distâncias do UE a todos os APs
    distances = np.linalg.norm(ap_coords - ue_coords, axis=1)
    
    # Calcula o ganho usando a fórmula do documento
    gains = x_shadowing * K_CANAL / (distances ** N_CANAL)
    
    return gains

def simular_experimento():
    """
    Executa uma única simulação da travessia do UE, aplicando as regras de 
    handover Soft e Hard a cada passo de tempo.
    """
    ap1 = AP(0, 0, id_ap=1)
    ap2 = AP(DISTANCIA_APs, 0, id_ap=2)
    aps = [ap1, ap2]
    
    # O UE começa sua jornada na posição inicial
    ue = UE(TRAJETO_INICIO, 0)

    # Calcula o número total de passos na simulação
    distancia_total_percurso = TRAJETO_FIM - TRAJETO_INICIO
    tempo_total = distancia_total_percurso / VELOCIDADE_UE
    num_passos = int(np.ceil(tempo_total / T_S))

    # Vetores para armazenar os resultados a cada passo
    tempos = np.arange(num_passos) * T_S
    posicoes = np.zeros(num_passos)
    
    conexoes_soft = np.zeros(num_passos, dtype=int)
    outages_soft = np.zeros(num_passos, dtype=bool)
    
    conexoes_hard = np.zeros(num_passos, dtype=int)
    outages_hard = np.zeros(num_passos, dtype=bool)

    # Condição inicial: UE começa conectado ao AP1 para ambos os algoritmos
    conexoes_hard[0] = ap1.id

    for t in range(num_passos):
        # Atualiza a posição do UE
        ue.x = TRAJETO_INICIO + t * VELOCIDADE_UE * T_S
        posicoes[t] = ue.x
        
        # Calcula o ganho para os dois APs na posição atual
        ganho_ap1, ganho_ap2 = calcular_ganho_canal(ue, aps)

        # --- Lógica do Soft Handover ---
        # Conecta-se sempre ao AP com maior ganho (potência)
        if ganho_ap1 > ganho_ap2:
            conexoes_soft[t] = ap1.id
            ganho_conectado = ganho_ap1
        else:
            conexoes_soft[t] = ap2.id
            ganho_conectado = ganho_ap2
        
        # Verifica se há outage no Soft Handover
        if ganho_conectado < P_MIN:
            outages_soft[t] = True

        # --- Lógica do Hard Handover ---
        # A decisão depende da conexão no passo anterior (se t > 0)
        ap_conectado_anterior = conexoes_hard[t-1] if t > 0 else conexoes_hard[0]
        
        ganho_do_ap_conectado = ganho_ap1 if ap_conectado_anterior == ap1.id else ganho_ap2
        
        # Verifica se há outage no Hard Handover
        if ganho_do_ap_conectado < P_MIN:
            outages_hard[t] = True
            # Se houve outage, a conexão no próximo passo mudará para o outro AP
            proximo_ap = ap2.id if ap_conectado_anterior == ap1.id else ap1.id
        else:
            # Se não houve outage, a conexão é mantida
            proximo_ap = ap_conectado_anterior

        # Atualiza a conexão para o passo atual (exceto o último)
        if t < num_passos -1:
            conexoes_hard[t] = proximo_ap
        else: # No último passo, a conexão é a mesma do anterior
             conexoes_hard[t] = ap_conectado_anterior
    

    return {
        "tempos": tempos,
        "posicoes": posicoes,
        "conexoes_soft": conexoes_soft,
        "outages_soft": outages_soft,
        "conexoes_hard": conexoes_hard,
        "outages_hard": outages_hard,
    }

def plot_handover_comparison(resultados: dict):
    """
    Plota o gráfico comparativo dos algoritmos de handover, conforme item b.
    """
    tempos = resultados["tempos"]
    
    fig, ax = plt.subplots(figsize=(14, 7))

    # --- Plotar linhas de conexão ---
    # Usamos 'step' para mostrar claramente as transições discretas
    ax.step(tempos, resultados["conexoes_soft"], where='post', 
            label='Conexão Soft Handover', color='blue', linewidth=2)
    ax.step(tempos, resultados["conexoes_hard"], where='post', 
            label='Conexão Hard Handover', color='red', linestyle='--', linewidth=2)

    # --- Marcar os outages ---
    # Encontra os pontos onde houve outage
    t_outage_soft = tempos[resultados["outages_soft"]]
    conn_outage_soft = resultados["conexoes_soft"][resultados["outages_soft"]]
    
    t_outage_hard = tempos[resultados["outages_hard"]]
    # A conexão durante o outage do hard handover é a do AP que falhou
    conn_outage_hard = resultados["conexoes_hard"][np.roll(resultados["outages_hard"], 1)]
    conn_outage_hard[0] = resultados["conexoes_hard"][0]


    if t_outage_soft.any():
        ax.plot(t_outage_soft, conn_outage_soft, 'x', color='cyan', 
                markersize=10, markeredgewidth=2, label='Outage Soft')
    if t_outage_hard.any():
        ax.plot(t_outage_hard, conn_outage_hard, 'x', color='magenta', 
                markersize=10, markeredgewidth=2, label='Outage Hard')
    
    # --- Configurações do Gráfico ---
    ax.set_xlabel('Tempo (s)')
    ax.set_ylabel('Ponto de Acesso (AP)')
    ax.set_title('Comparação de Algoritmos de Handover (item 11.b)')
    ax.set_yticks([1, 2])
    ax.set_yticklabels(['AP1', 'AP2'])
    ax.set_ylim(0.5, 2.5)
    ax.grid(True, which='both', linestyle=':', linewidth=0.6)
    ax.legend()
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Garante que os resultados aleatórios sejam os mesmos a cada execução
    # np.random.seed(42) 
    
    # Executa a simulação
    for i in range(10):
        resultados_simulacao = simular_experimento()
        print(i, 'simulação')
        print(resultados_simulacao)
        #print(f'Soft: {np.sum([1 for _ in resultados_simulacao["outages_soft"] if _ ==True])}')
        #print(f'Hard: {np.sum([1 for _ in resultados_simulacao["outages_hard"] if _ ==True])}')

    # Plota o gráfico comparativo
    plot_handover_comparison(resultados_simulacao)

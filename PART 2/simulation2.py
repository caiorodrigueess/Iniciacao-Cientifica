import numpy as np
import matplotlib.pyplot as plt

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

def attach_AP_UE(ue: UE, aps: list) -> np.ndarray:
    ap_coords = np.array([[ap.x, ap.y] for ap in aps])

    # Cálculo vetorizado do ganho
    x = np.random.lognormal(0, 2, len(aps))
    distances = np.linalg.norm(ap_coords - np.array([ue.x, ue.y]), axis=1)
    gains = x * 1e-4 / (distances ** 4)
    
    return gains

def simular_experimento():
    """
    Executa uma única simulação da travessia do UE, aplicando as regras de 
    handover Soft e Hard a cada passo de tempo.
    """
    ap1 = AP(333, 333, 1)
    ap2 = AP(666, 666, 2)
    aps = [ap1, ap2]
    
    # O UE começa sua na posição inicial
    ue = UE(383, 383)
    ganho_ap1 = []
    ganho_ap2 = []
    posicoes = []

    while(np.sqrt((ue.x - ap2.x)**2 + (ue.y - ap2.y)**2) > 50):
        posicoes.append([ue.x, ue.y])
        ganho_ap1.append(attach_AP_UE(ue, aps)[0])
        ganho_ap2.append(attach_AP_UE(ue, aps)[1])
        theta = np.random.uniform(0, np.pi/2)
        ue.x = ue.x + 20*np.cos(theta)
        ue.y = ue.y + 20*np.sin(theta)

    num_passos = len(ganho_ap1)

    # Vetores para armazenar os resultados a cada passo
    tempos = np.arange(num_passos)
    
    conexoes_soft = np.zeros(num_passos, dtype=int)
    outages_soft = np.zeros(num_passos, dtype=bool)
    
    conexoes_hard = np.zeros(num_passos, dtype=int)
    outages_hard = np.zeros(num_passos, dtype=bool)

    # Condição inicial: UE começa conectado ao AP1 para ambos os algoritmos
    conexoes_hard[0] = ap1.id
    p_min = 1e-17

    for t in range(num_passos):
        # --- Lógica do Soft Handover ---
        # Conecta-se sempre ao AP com maior ganho (potência)
        if ganho_ap1[t] > ganho_ap2[t]:
            conexoes_soft[t] = ap1.id
            ganho_conectado = ganho_ap1[t]
        else:
            conexoes_soft[t] = ap2.id
            ganho_conectado = ganho_ap2[t]

        # Verifica se há outage no Soft Handover
        if ganho_conectado < p_min:
            outages_soft[t] = True

        # --- Lógica do Hard Handover ---
        # A decisão depende da conexão no passo anterior (t-1)
        ap_conectado_anterior = conexoes_hard[t-1] if t > 0 else conexoes_hard[0]

        ganho_do_ap_conectado = ganho_ap1[t] if ap_conectado_anterior == ap1.id else ganho_ap2[t]

        # Verifica se há outage no Hard Handover
        if ganho_do_ap_conectado < p_min:
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
        "posicoes": posicoes,
        "tempos": tempos,
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
    
    _, ax = plt.subplots(figsize=(14, 7))

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
        resultados_simulacao = simular_experimento()
        print(f'Conexões: {resultados_simulacao['conexoes_soft']}, {resultados_simulacao['conexoes_soft']}')
        #print(f'Soft: {np.sum([1 for _ in resultados_simulacao["outages_soft"] if _ ==True])}')
        #print(f'Hard: {np.sum([1 for _ in resultados_simulacao["outages_hard"] if _ ==True])}')

    # Plota o gráfico comparativo
        plot_handover_comparison(resultados_simulacao)

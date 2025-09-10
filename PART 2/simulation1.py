import numpy as np
import matplotlib.pyplot as plt

# --- Parâmetros da Simulação (baseado no Exercício 11) ---
DISTANCIA_APs = 3100.0
VELOCIDADE_UE = 20.0
T_S = 1.0
P_MIN = 1e-17            # Limiar de potência para outage em Watts [cite: 520]
TRAJETO_INICIO = 50.0    # UE começa a 50m do AP1 [cite: 496, 502]
TRAJETO_FIM = DISTANCIA_APs - 50.0 # UE termina a 50m do AP2 [cite: 496, 503]
K_CANAL = 1e-4           # Constante de propagação k [cite: 513]
N_CANAL = 4              # Expoente de perda de percurso n [cite: 513]
SIGMA_X = 2              # Desvio padrão do sombreamento (shadowing) [cite: 513]

def simular_travessia():
    """
    Simula a travessia do UE e retorna os ganhos de canal ao longo do percurso.
    """
    # Define as posições dos APs
    ap_pos = np.array([[0, 0], [DISTANCIA_APs, 0]])

    # Calcula o número de passos da simulação
    distancia_percurso = TRAJETO_FIM - TRAJETO_INICIO
    tempo_total = distancia_percurso / VELOCIDADE_UE
    num_passos = int(np.ceil(tempo_total / T_S))
    
    # Prepara vetores para armazenar resultados
    posicoes = np.zeros(num_passos)
    ganhos_ap1 = np.zeros(num_passos)
    ganhos_ap2 = np.zeros(num_passos)

    # Loop principal da simulação
    for t in range(num_passos):
        pos_x = TRAJETO_INICIO + t * VELOCIDADE_UE * T_S
        posicoes[t] = pos_x
        ue_pos = np.array([pos_x, 0])
        
        # Calcula o ganho do canal para ambos os APs
        distancias = np.linalg.norm(ap_pos - ue_pos, axis=1)
        sombreamento = np.random.lognormal(0, SIGMA_X, 2)
        ganhos = sombreamento * K_CANAL / (distancias ** N_CANAL)
        
        ganhos_ap1[t] = ganhos[0]
        ganhos_ap2[t] = ganhos[1]
        
    return {
        "posicoes": posicoes,
        "ganhos_ap1": ganhos_ap1,
        "ganhos_ap2": ganhos_ap2
    }

def plot_ganho_vs_trajeto(resultados: dict):
    """
    Plota o gráfico de ganho vs. posição, destacando os períodos de outage,
    conforme solicitado no item 11.a.
    """
    posicoes = resultados["posicoes"]
    ganhos_ap1 = resultados["ganhos_ap1"]
    ganhos_ap2 = resultados["ganhos_ap2"]

    fig, ax = plt.subplots(figsize=(14, 7))

    # --- Plotagem para AP1 ---
    # Cria uma máscara para identificar quando o ganho está abaixo do limiar
    mask_ap1_outage = ganhos_ap1 < P_MIN
    
    # Prepara os dados: onde a condição não é atendida, o valor vira NaN para não ser plotado
    ap1_ok = np.where(mask_ap1_outage, np.nan, ganhos_ap1)
    ap1_outage = np.where(~mask_ap1_outage, np.nan, ganhos_ap1)

    ax.plot(posicoes, ap1_ok, color='blue', label='Ganho AP1 (Sinal OK)', linewidth=1.5)
    ax.plot(posicoes, ap1_outage, color='red', label='Ganho AP1 (Abaixo do Limiar)', linewidth=1.5)

    # --- Plotagem para AP2 ---
    mask_ap2_outage = ganhos_ap2 < P_MIN
    ap2_ok = np.where(mask_ap2_outage, np.nan, ganhos_ap2)
    ap2_outage = np.where(~mask_ap2_outage, np.nan, ganhos_ap2)
    
    ax.plot(posicoes, ap2_ok, color='green', label='Ganho AP2 (Sinal OK)', linewidth=1.5)
    ax.plot(posicoes, ap2_outage, color='orange', label='Ganho AP2 (Abaixo do Limiar)', linewidth=1.5)

    # --- Linha do Limiar e Configurações do Gráfico ---
    ax.axhline(P_MIN, color='black', linestyle='--', linewidth=1.2, label=f'Limiar P_min ({P_MIN:.0e} W)')
    
    ax.set_yscale('log') # Eixo Y em escala logarítmica para melhor visualização
    ax.set_title('Ganho de Canal vs. Posição do Usuário (item 11.a)')
    ax.set_xlabel('Posição do Usuário no Eixo X (metros)')
    ax.set_ylabel('Ganho de Canal (escala log)')
    ax.grid(True, which='both', linestyle=':')
    ax.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # np.random.seed(42) # Descomente para obter sempre o mesmo resultado aleatório
    
    # Executa a simulação para obter os dados
    resultados_da_simulacao = simular_travessia()
    
    # Plota o gráfico do item 11.a
    plot_ganho_vs_trajeto(resultados_da_simulacao)
import numpy as np
import random
import matplotlib.pyplot as plt

def distribuir_APs(M):
    if M not in [1, 4, 9, 16, 25, 36, 49, 64]:
        return None

    tamanho_quadrado = 1000
    lado_quadrado = int(np.sqrt(M))

    tamanho_celula = tamanho_quadrado // lado_quadrado

    # Criar coordenadas usando meshgrid
    x, y = np.meshgrid(np.arange(0.5 * tamanho_celula, tamanho_quadrado, tamanho_celula),
                      np.arange(0.5 * tamanho_celula, tamanho_quadrado, tamanho_celula))

    coordenadas_APs = np.column_stack((x.ravel(), y.ravel()))

    return coordenadas_APs

def dAPUE(x_coord, y_coord, M):
    dAPUE = np.linalg.norm(np.array([x_coord, y_coord]) - M)
    return dAPUE

def pot_rec(pot_trans, dist, d_0):
    c = 1e-4
    n = 4
    pot_rec_result = 0  # Inicializa a variável local

    if dist >= d_0:
        pot_rec_result = pot_trans * (c / ((dist) ** n))

    return pot_rec_result

def simular_experimento(B_t, p_t, d_0, K_0, M, N):
    x_coord = random.random() * 1000
    y_coord = random.random() * 1000

    distanciaAP_UE = np.zeros(M)
    coordAp = distribuir_APs(M)
    potencia_recebida = np.zeros(M)
    p_n = K_0*(B_t/N)
    SNR = np.zeros(M)
    for i in range(M):
        distanciaAP_UE[i] = dAPUE(x_coord, y_coord, coordAp[i])
        distan = distanciaAP_UE[i]
        potencia_recebida[i] = pot_rec(p_t, distan, d_0)
        SNR[i] = potencia_recebida[i]/p_n

    #Calculando a Capacidade
    B_c = B_t / N
    snr_2 = np.sum(SNR)
    Capacidade = np.zeros(K)
    for i in range(K):
      Capacidade[i] = B_c * np.log2(1+snr_2)

    return SNR

B_t, p_t, d_0, K_0 = 100e6, 1e3, 1, 1e-17 # Em MHz, mW, metros, mW/Hz respectivamente
M_1, K, N = 1, 1, 1 #Número de APs, UEs e Canais respectivamente
M_9, M_25, M_64 = 9, 25, 64

# Número de iterações
num_iteracoes = 1000

# Armazenar todas as capacidades
Capacidade_total_1 = []
Capacidade_total_9 = []
Capacidade_total_25 = []
Capacidade_total_64 = []

# Iteração
for _ in range(num_iteracoes):
    Capacidade_iteracao_1 = simular_experimento(B_t, p_t, d_0, K_0, M_1, N)
    Capacidade_total_1 = np.concatenate((Capacidade_total_1, Capacidade_iteracao_1))

for _ in range(num_iteracoes):
    Capacidade_iteracao_9 = simular_experimento(B_t, p_t, d_0, K_0, M_9, N)
    Capacidade_total_9 = np.concatenate((Capacidade_total_9, Capacidade_iteracao_9))

for _ in range(num_iteracoes):
    Capacidade_iteracao_25 = simular_experimento(B_t, p_t, d_0, K_0, M_25, N)
    Capacidade_total_25 = np.concatenate((Capacidade_total_25, Capacidade_iteracao_25))

for _ in range(num_iteracoes):
    Capacidade_iteracao_64 = simular_experimento(B_t, p_t, d_0, K_0, M_64, N)
    Capacidade_total_64 = np.concatenate((Capacidade_total_64, Capacidade_iteracao_64))

#Deixando em ordem crescente
x1 = np.sort(Capacidade_total_1)
x2 = np.sort(Capacidade_total_9)
x3 = np.sort(Capacidade_total_25)
x4 = np.sort(Capacidade_total_64)

# Plotar apenas o eixo x em decibéis

plt.plot(x1, np.arange(0, len(Capacidade_total_1)) / len(Capacidade_total_1), label = '1 AP')
plt.plot(x2, np.arange(0, len(Capacidade_total_1)) / len(Capacidade_total_1), label = '9 APs')
plt.plot(x3, np.arange(0, len(Capacidade_total_1)) / len(Capacidade_total_1), label = '25 APs')
plt.plot(x4, np.arange(0, len(Capacidade_total_1)) / len(Capacidade_total_1), label = '64 APs')
plt.legend()

# Adicionando linha tracejada em x=100e6
plt.axvline(x=100e6, color='grey', linestyle='--', label='x=100e6')

# Adicionando linha tracejada em y=0.1
plt.axhline(y=0.1, color='grey', linestyle='--', label='y=0.1')


plt.xlabel('Capacidade (bps)')
plt.ylabel('Porcentagem')
plt.title('CDF da Capacidade')
plt.show()
import numpy as np
import matplotlib.pyplot as plt

# Parâmetros iniciais
n_pontos = 500
n_sim = 1000
centro_x, centro_y = 0.5, 0.5
raio = 0.5

# Simulação de Monte Carlo para estimar a área usando 500 pontos em 1000 simulações
area_estimada = []

for i in range(n_sim):               # Inicia um laço para rodar o número de simulações

    # Gerar 500 pontos aleatórios (x, y) no quadrado de lado 1
    x = np.random.rand(n_pontos)
    y = np.random.rand(n_pontos)

    dist_cp = np.sqrt((x - centro_x)**2 + (y - centro_y)**2)                # Atribui o módulo do ponto no plano
    dentro_do_circulo = dist_cp <= raio                                     # Verifica se o ponto está dentro do circulo

    contagem_dentro = np.sum([1 for x in dentro_do_circulo if x==True])       # Incrementa o número de pontos dentro do circulo
    area_estimada.append(contagem_dentro / n_pontos)                           # Razão do número de pontos dentro

area_estimada.sort()

# Plotar o círculo e os pontos aleatórios (uma simulação)
plt.figure(figsize=(6, 6))
plt.xlim(0,1)
plt.ylim(0,1)
circle = plt.Circle((0.5, 0.5), 0.5, linewidth=.5, edgecolor='k', facecolor='none')
plt.gca().add_patch(circle)
for i in range(500):
    plt.scatter(np.random.random(1), np.random.random(1), marker='+', color='black')
plt.grid(False)
plt.show()


import numpy as np
import matplotlib.pyplot as plt

# Parâmetros iniciais
n_pontos = 500
n_sim = 1000
centro_x, centro_y = 0.5, 0.5
raio = 0.5

# Simulação de Monte Carlo para estimar a área usando 500 pontos em 1000 simulações
estimativas_area = []
area_estimada = []
contagem_dentro = 0

for i in range(n_sim):               # Inicia um laço para rodar o número de simulações

    # Gerar 500 pontos aleatórios (x, y) no quadrado de lado 1
    x = np.random.rand(n_pontos)
    y = np.random.rand(n_pontos)

    dist_cp = np.sqrt((x - centro_x)**2 + (y - centro_y)**2)                # Atribui o módulo do ponto no plano
    dentro_do_circulo = dist_cp <= raio                                     # Verifica se o ponto está dentro do circulo

    contagem_dentro = np.sum([1 for x in dentro_do_circulo if x==True])       # Incrementa o número de pontos dentro do circulo
    area_estimada.append(contagem_dentro / n_pontos)                           # Razão do número de pontos dentro
    estimativas_area.append(np.mean(area_estimada))

# estimativas_area.sort()

# Plotar o gráfico das estimativas de área
plt.figure(figsize=(10, 6))
plt.plot(range(1, n_sim + 1), estimativas_area, label='Estimativa da Área')
plt.axhline(y=np.pi * raio**2, color='r', linestyle='--', label='Área real do círculo')

# Configurar o gráfico
plt.title('Estimativa da área do círculo pelo número de simulações (Método Monte Carlo)')
plt.xlabel('Número de Simulações')
plt.ylabel('Estimativa da Área')
plt.legend()
plt.grid(True)
plt.show()
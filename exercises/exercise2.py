import numpy as np
import matplotlib.pyplot as plt

# Parâmetros iniciais
n_pontos = 500
n_simulacoes = 1000
centro_x, centro_y = 0.5, 0.5
raio = 0.5

# Gerar 500 pontos aleatórios (x, y) no quadrado de lado 1
x = np.random.rand(n_pontos)
y = np.random.rand(n_pontos)

dist_cp = (x - centro_x)**2 + (y - centro_y)**2
dentro_do_circulo = dist_cp <= raio

# Criar o gráfico dos pontos
plt.figure(figsize=(6, 6))
plt.scatter(x[dentro_do_circulo], y[dentro_do_circulo], color='blue', label='Dentro do círculo')
plt.scatter(x[~dentro_do_circulo], y[~dentro_do_circulo], color='red', label='Fora do círculo')

# Desenhar o círculo circunscrito
circulo = plt.Circle((centro_x, centro_y), raio, color='green', fill=False, linewidth=2)
plt.gca().add_artist(circulo)

# Ajustar os limites do gráfico para o quadrado [0, 1] x [0, 1]
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.title('Pontos aleatórios no quadrado de lado 1 e círculo circunscrito')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.gca().set_aspect('equal', adjustable='box')
plt.show()

# Simulação de Monte Carlo para estimar a área usando 500 pontos em 1000 simulações
estimativas_area = []
contagem_dentro = 0

for i in range(1, n_simulacoes + 1):    # Inicia um laço para rodar o número de simulações
    id = np.random.randn(n_pontos)      # Escolhe um ponto aleatório dentre os 500
    if dist_cp[id] <= raio**2:          # Verifica se o ponto está dentro do circulo
        contagem_dentro+=1              # Incrementa o número de pontos dentro do circulo
    
    area_estimada = 4 * (contagem_dentro / i) * raio**2
    estimativas_area.append(area_estimada)


# Plotar o gráfico das estimativas de área
plt.figure(figsize=(10, 6))
plt.plot(range(1, n_simulacoes + 1), estimativas_area, label='Estimativa da Área')
plt.axhline(y=np.pi * raio**2, color='r', linestyle='--', label='Área real do círculo')

# Configurar o gráfico
plt.title('Estimativa da área do círculo pelo número de simulações (Método Monte Carlo)')
plt.xlabel('Número de Simulações')
plt.ylabel('Estimativa da Área')
plt.legend()
plt.grid(True)
plt.show()
    
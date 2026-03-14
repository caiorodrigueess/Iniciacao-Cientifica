import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

# Parâmetros da malha
tamanho_area = 1000  # metros
passo = 1            # metro
d_cor = 35           # Distância de decorrelação em metros

# Parâmetros da distribuição Normal X (antes da exponencial)
mu_X = 0           # Média 
sigma_X = 2        # Desvio padrão

# gerar a matriz base de variáveis normais X ~ N(mu_X, sigma_X)
n_pontos = int(tamanho_area / passo)
ruido_normal = np.random.normal(mu_X, sigma_X, (n_pontos, n_pontos))

# aplicar a correlação espacial (filtro gaussiano)
filtro_sigma = d_cor / (np.sqrt(2) * passo)
X_correlacionado = gaussian_filter(ruido_normal, sigma=filtro_sigma)

# como o filtro altera a média e a variância do sinal, precisamos re-normalizar para garantir que os parâmetros mu_X e sigma_X se mantenham
X_correlacionado = (X_correlacionado - np.mean(X_correlacionado)) / np.std(X_correlacionado)
X_correlacionado = (X_correlacionado * sigma_X) + mu_X

# aplicar a transformação Lognormal ( x = exp(X) )
x_lognormal = np.exp(X_correlacionado)

# converter para dB para a plotagem
shadowing_db = 10 * np.log10(x_lognormal)

# --- Visualização (gráfico) ---
x = np.arange(0, tamanho_area, passo)
y = np.arange(0, tamanho_area, passo)
X_grid, Y_grid = np.meshgrid(x, y)

fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# Plotando a malha
surf = ax.plot_surface(X_grid, Y_grid, shadowing_db, cmap='jet', edgecolor='k', linewidth=0.5, alpha=0.8)

ax.set_title('Malha de Shadowing - Distribuição Lognormal')
ax.set_xlabel('Eixo X [m]')
ax.set_ylabel('Eixo Y [m]')
ax.set_zlabel('Shadowing (dB)')
fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, label='dB')
plt.tight_layout()
plt.show()
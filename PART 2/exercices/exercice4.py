import numpy as np
# Parametros
mean = 175  # Média
std_dev = 10  # Desvio padrão

# Generate random values vector
vetor = np.random.normal(mean, std_dev, 10**4)

print(vetor[0:10])  # Print first 10 values for verification

# Plot histogram
import matplotlib.pyplot as plt
plt.hist(vetor, bins=70, density=True, rwidth=1, alpha=0.6, color='m', edgecolor='black')
plt.title('Histogram of Random Values')
plt.xlabel('Value')
plt.ylabel('Density')
plt.grid(True)
plt.show()

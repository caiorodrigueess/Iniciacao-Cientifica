import numpy as np
import matplotlib.pyplot as plt

# Parameters
X = np.random.lognormal(mean=0, sigma=2, size=1000)  # Log-normal distribution
d = np.arange(1, 1001)

# Calculate path gain using vectorized operations
pr = np.where(d != 0, X * 1e-4 / (d ** 4), 0)

# Plot graph
plt.figure(figsize=(10, 6))
plt.plot(d, pr, label='Path Gain')
plt.yscale('log')
plt.title('Path Gain vs Distance')
plt.xlabel('Distance (m)')
plt.ylabel('Path Gain')
plt.grid(True)
plt.show()

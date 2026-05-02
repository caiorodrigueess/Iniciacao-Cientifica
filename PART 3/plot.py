import matplotlib.pyplot as plt
import numpy as np
from simulacao import distribuir_AP, gain_matrix, attach_AP_UE, UE

M, L = 4, 100

plt.figure(figsize=(8, 6))
aps = distribuir_AP(M, L)
ues = [UE(L) for _ in range(2)]
G = gain_matrix(ues, aps, L)
attach_AP_UE(ues, aps, G)

for ap in aps:
    plt.scatter(ap.x, ap.y, marker='^', color='blue', label='AP' if ap == aps[0] else "")
for ue in ues:
    plt.scatter(ue.x, ue.y, marker='s', color='red', label='UE' if ue == ues[0] else "")
    plt.plot([ue.ap.x, ue.x], [ue.ap.y, ue.y], linestyle='-', color='k', linewidth=0.25, )

ap1 = [aps[2].x, aps[2].y]
ap2 = [aps[3].x, aps[3].y]

plt.plot([ap1[0], ap2[0]], [ap1[1], ap2[1]], linestyle='--', color='k', linewidth=0.5)

# Calcula o meio
x_mid = (ap1[0] + ap2[0]) / 2
y_mid = (ap1[1] + ap2[1]) / 2

# Adiciona o texto logo acima
plt.text(x_mid, y_mid + 3, '50m', 
         horizontalalignment='center', 
         verticalalignment='bottom',
         fontsize=10)

plt.xlim(0, L)
plt.ylim(0, L)
plt.xlabel('x')
plt.ylabel('y')
plt.title('APs e UEs')
dx = L/(np.sqrt(M))
'''plt.xticks(np.arange(0, L+1, dx))
plt.yticks(np.arange(0, L+1, dx))'''
#plt.grid(True, linestyle='-', alpha=0.5)
plt.legend()
plt.show()
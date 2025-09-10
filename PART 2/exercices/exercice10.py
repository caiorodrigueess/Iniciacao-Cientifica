import numpy as np
import matplotlib.pyplot as plt

class AP:
    def __init__(self, x: float, y: float):
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
    

def attach_AP_UE(ue: UE, aps: list) -> float:
    ap_coords = np.array([[ap.x, ap.y] for ap in aps])

    # Cálculo vetorizado do ganho
    x = np.random.lognormal(0, 2, len(aps))
    distances = np.linalg.norm(ap_coords - np.array([ue.x, ue.y]), axis=1)
    gains = x * 1e-4 / (distances ** 4)
    
    return gains

def simular_experimento():
    ap1 = AP(0, 0)
    ap2 = AP(3100, 0)
    ue1 = UE(1, 0)
    gains = np.zeros((2, 3099))

    for n in range(3099):
        gains1, gains2 = attach_AP_UE(ue1, [ap1, ap2])
        ue1.x += 1
        gains[0, n] = gains1
        gains[1, n] = gains2

    return gains

def plot_gain(gains: np.ndarray):
    plt.figure(figsize=(10, 6))
    plt.semilogy(range(len(gains[0])), gains[0], linewidth = 0.35, color='#1F4DE6', label='AP1')
    plt.semilogy(range(len(gains[1])), gains[1], linewidth = 0.35, color="#F16717", label='AP2')
    plt.xlabel('Position (m)')
    plt.ylabel('Gain')
    plt.title('Gain of each AP for UE')
    plt.grid()
    plt.legend()
    plt.show()
    
if __name__ == "__main__":

    gains = simular_experimento()
    plot_gain(gains)
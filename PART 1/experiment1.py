import numpy as np
import matplotlib.pyplot as plt

class AP:
    def __init__(self, x: float, y: float, id: int):
        self.x = x
        self.y = y
        self.id = id
        self.channel = None
        self.ues = []

    def __str__(self):
        return f'AP[{self.id}]({self.x}, {self.y})'

class UE:
    def __init__(self, N: int):
        self.x = np.random.randint(0, 1001)
        self.y = np.random.randint(0, 1001)
        self.ap = None
        self.channel = np.random.randint(1, N+1)

    def __str__(self):
        return f'UE({self.x}, {self.y})'

def distribuir_AP(M: int) -> np.ndarray:
    APs = []
    dx = 1000/(2*np.sqrt(M))                    # passo
    a = np.arange(dx, 1001-dx, 2*dx)
    x, y = np.meshgrid(a, a)
    id = 1
    for xi, yi in zip(x.ravel(), y.ravel()):
        APs.append(AP(xi, yi, id))
        id += 1
    return APs

def dist_AP_UE(M: int, N: int) -> float:
    aps = distribuir_AP(M)
    ue = UE(N)
    dist = 1e4
    for ap in aps:
        x = np.linalg.norm(np.array([ue.x, ue.y]) - np.array([ap.x, ap.y]))
        if x < dist:
            dist = x
            ue.ap = ap
            ap.channel = ue.channel
    return dist

def SINR(dist: float, N: int) -> float:
    pt=1        # transmited power
    k=1e-4      # constant
    n=4         # path gain
    bt=1e8      # avaiable bandwidth
    k0=1e-20    # constant for the noise power

    pr = pt*k/(dist**n)
    pn = k0*bt/N
    return pr/pn

def channel_capacity(SINR: float, N: int) -> float:
    bt=100      # avaiable bandwidth
    return np.around((bt/N)*np.log2(1+SINR), 2)

def simular_experimento(M: int, N: int, sim: int) -> np.ndarray:
    cap_canal = np.zeros(sim)
    for i in range(sim):
        cap_canal[i] = channel_capacity(SINR(dist_AP_UE(M, N), N), N)
    return cap_canal

if __name__ == '__main__':
    M = [1, 9, 36, 64]
    N = [1, 2, 3]
    sim = 10000
    
    teste = np.sort(simular_experimento(64, 1, 1000))
    print(teste)
    print(f'10th: {np.percentile(teste, 10)}')
    print(f'50th: {np.percentile(teste, 50)}')
    print(f'90th: {np.percentile(teste, 90)}')
    
    '''
    cdf = np.zeros((len(M), len(N), sim))

    i=0
    for m in M:
        j=0
        for n in N:
            x = simular_experimento(m, n, sim)
            cdf[i,j,:] = x
            j+=1
        i+=1

    cdf_1 = np.sort(cdf[0,:,:])
    cdf_9 = np.sort(cdf[1,:,:])
    cdf_36 = np.sort(cdf[2,:,:])
    cdf_64 = np.sort(cdf[3,:,:])
    '''

    '''
    Até o momento, tenho uma matriz de M linhas e N colunas
    Falta separar o vetor e plotar o gráfico
    cdf.sort()

    
    print(cdf.shape)

    plt.figure(figsize=(10, 6))
    plt.plot(range(1, sim + 1), cdf_1[0,:], label='CDF', color="b")
    plt.xlabel('Número de Simulações')
    plt.ylabel('Capacidade do Canal')
    plt.legend()
    plt.grid(True)
    plt.show()


    
    # Código inicial do gráfico -> precisa de alterações
    print(f'10th: {round(np.percentile(cdf, 10), 4)}')
    print(f'50th: {round(np.percentile(cdf, 50), 4)}')
    print(f'90th: {round(np.percentile(cdf, 90), 4)}')

    # Plotar o gráfico das estimativas de área
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, 100 + 1), cdf, label='CDF')
    plt.axhline(y=np.percentile(cdf, 50), color='r', linestyle='--', label='Mediana')

    # Configurar o gráfico
    plt.title('Estimativa da Capacidade do Canal pelo número de simulações (Método Monte Carlo)')
    plt.xlabel('Número de Simulações')
    plt.ylabel('Capacidade do Canal')
    plt.legend()
    plt.grid(True)
    plt.show()
    '''
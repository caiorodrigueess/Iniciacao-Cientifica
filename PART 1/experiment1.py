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
    dx = 1000/(2*np.sqrt(M))
    a = np.arange(dx, 1001-dx, 2*dx)
    x, y = np.meshgrid(a, a)
    id = 1
    for xi, yi in zip(x.ravel(), y.ravel()):
        APs.append(AP(xi, yi, id))
        id += 1
    return APs

def dist_AP_UE(M: int, UE: UE) -> float:
    aps = distribuir_AP(M)
    dist = 1e4
    ue = UE
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

def simular_experimento(M: int, N: int, sim: int) -> tuple:
    sinr = np.zeros(sim)
    cap_canal = np.zeros(sim)
    for i in range(sim):
        ue = UE(N)
        sinr[i] = SINR(dist_AP_UE(M, ue), N)
        cap_canal[i] = channel_capacity(sinr[i], N)
    return cap_canal, sinr, ue

def plot_APs_UEs(M: int, ue: UE):
    plt.figure(figsize=(10, 6))
    aps = distribuir_AP(M)

    for ap in aps:
        plt.scatter(ap.x, ap.y, marker='^', color='blue', label='AP' if ap == aps[0] else "")
    plt.scatter(ue.x, ue.y, marker='s', color='red', label='UE')
    plt.plot([ue.ap.x, ue.x], [ue.ap.y, ue.y], linestyle='-', color='k', linewidth=1)

    plt.xlim(0, 1000)
    plt.ylim(0, 1000)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('APs e UEs')
    dx = 1000/(np.sqrt(M))
    plt.xticks(np.arange(0, 1001, dx))
    plt.yticks(np.arange(0, 1001, dx))
    plt.grid(True, linestyle='-', alpha=0.5)
    plt.show()
    
    
def plot_cdf(cdf: np.ndarray, tipo: str = 'capacity'):
    cdf.sort()
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(cdf) + 1), cdf, label='CDF')
    plt.axhline(y=np.percentile(cdf, 50), color='r', linestyle='--', label='Mediana')

    if tipo == 'sinr':
        plt.title('Estimativa do SINR pelo número de simulações (Método Monte Carlo)')
        plt.ylabel('SINR')
    else:
        plt.title('Estimativa da Capacidade do Canal pelo número de simulações (Método Monte Carlo)')
        plt.ylabel('Capacidade do Canal (Mbps)')
    plt.xlabel('Número de Simulações')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == '__main__':
    M = [1, 9, 36, 64]
    N = [1, 2, 3]
    sim = 10000

    for m in M:
        for n in N:
            teste, sinr, ue_retornada = simular_experimento(m, n, sim)
            print(f'Cenário: {m} APs e {n} canal(is)')
            teste.sort()
            print(f'10th: {np.percentile(teste, 10)}')
            print(f'50th: {np.percentile(teste, 50)}')
            print(f'90th: {np.percentile(teste, 90)}')
            '''plot_APs_UEs(64, ue_retornada)
            plot_cdf(teste)
            plot_cdf(sinr)'''
            


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
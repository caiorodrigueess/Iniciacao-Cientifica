import numpy as np
from imca import UE, AP

# definindo os parâmetros de potência e ruído
p_max = 1.0         # potência máxima (W)
p_min = 0.001       # potência mínima (W)
k0 = 1e-20          # constante de ruído (W/Hz)
bt = 100e6          # largura de banda (Hz)
pn = k0*bt          # potência de ruído (W)
y_tar = 1           # SINR alvo (linear)
t_max = 10          # número máximo de iterações

# Matriz X
X = np.array([
    [5.3434e-2, 2.8731e-1, 1.9691e-2, 7.3013e-1],
    [3.2318,    1.5770,    2.6449e-1, 5.6379   ],
    [6.1470e-3, 1.1424,    2.6826e-1, 4.5709   ],
    [1.3485e-1, 4.6690e-1, 7.8250e-1, 1.6742   ]
])

# Matriz R
R = np.array([
    [1.248699, 3.248041, 0.772754, 0.708962],
    [0.498887, 0.104890, 0.647280, 0.940906],
    [0.382966, 0.682700, 1.891256, 0.327100],
    [0.065737, 0.649500, 1.981107, 1.259538]
])


def simulate_noise_limited_scenario():
    # Vetor u
    u = np.array([
        225.83+203.33j,
        566.79+321.88j,
        765.51+146.88j,
        265.95+702.39j])

    print([f'Distancia: {np.linalg.norm(np.array([u[i].real, u[i].imag]) - np.array([250, 250]))} m' for i in range(4)])

    # lista com os UEs
    ues = [UE(1) for _ in range(4)]

    # atribuir as coordenadas dos UEs a partir do vetor u
    for i in range(4):
        ues[i].x = u[i].real
        ues[i].y = u[i].imag

    # distribuindo os aps
    aps_init = []
    dx = 1000/(2*np.sqrt(4))            # L = 1000m
    a = np.arange(dx, 1001-dx, 2*dx)
    x, y = np.meshgrid(a, a)
    id = 0
    for xi, yi in zip(x.ravel(), y.ravel()):
        aps_init.append(AP(xi, yi, id))
        id += 1

    aps = [aps_init[0], aps_init[2], aps_init[1], aps_init[3]]  # muda a ordem dos APs
    for ap in aps:
        ap.id = aps.index(ap)  # atualiza os IDs dos APs de acordo com a nova ordem

    # path gain matrix
    G = np.zeros((len(ues), len(aps)))
    ap_coords = np.array([[ap.x, ap.y] for ap in aps])

    for i, ue in enumerate(ues):
        ue_coord = np.array([ue.x, ue.y])

        for j in range(len(aps)):
            d = np.linalg.norm(ue_coord - ap_coords[j])
            G[i, j] = X[i][j] * 1e-4 / (d**4)

    # associando os UEs aos APs
    for i, ue in enumerate(ues):
        # Encontra o AP com o maior ganho para este UE
        best_index = np.argmax(G[i, :])
        ue.ap = aps[best_index]     # associa o UE ao AP com maior ganho
        ue.gain = G[i, best_index]  # armazena o ganho para este UE
        ue.dist = np.linalg.norm(np.array([ue.x, ue.y]) - np.array([ue.ap.x, ue.ap.y]))     # armazena a distancia
        aps[best_index].ues.append(ue)      # adiciona o UE à lista de UEs do AP


    # vetor de potencias
    p = np.ones((len(ues), t_max))         # inicializa as potências com 1W para cada UE
    y = np.zeros((len(ues), t_max))        # vetor para armazenar os SINRs ao longo das iterações

    for t in range(t_max):
        for i, ue in enumerate(ues):
            # Calcula a interferência total para este UE
            interference = sum([p[k][t] * G[k][ue.ap.id] * R[k][ue.ap.id] for k in range(len(ues)) if k != i])
            ue.interference.append(interference)

            # Calcula o SINR para este UE
            sinr = (p[i][t] * ue.gain * R[i][ue.ap.id]) / (interference + pn)
            y[i][t] = sinr

            # Atualiza a potência usando o algoritmo de controle de potência
            pt = min(max(p_min, p[i][t] * y_tar / sinr), p_max)

            if t < t_max - 1:  # Evita atualizar a potência na última iteração
                p[i][t+1] = pt

    return ues, aps, p, y

def simulate_interference_limited_scenario():
    # Vetor u
    u = np.array([
        22.583+20.333j,
        56.679+32.188j,
        76.551+14.688j,
        26.595+70.239j])

    # lista com os UEs
    ues = [UE(1) for _ in range(4)]

    # atribuir as coordenadas dos UEs a partir do vetor u
    for i in range(4):
        ues[i].x = u[i].real
        ues[i].y = u[i].imag

    # distribuindo os aps
    aps_init = []
    dx = 100/(2*np.sqrt(4))            # L = 100 m
    a = np.arange(dx, 101-dx, 2*dx)
    x, y = np.meshgrid(a, a)
    id = 0
    for xi, yi in zip(x.ravel(), y.ravel()):
        aps_init.append(AP(xi, yi, id))
        id += 1
    
    aps = [aps_init[0], aps_init[2], aps_init[1], aps_init[3]]  # muda a ordem dos APs
    for ap in aps:
        ap.id = aps.index(ap)  # atualiza os IDs dos APs de acordo com a nova ordem

    # path gain matrix
    G = np.zeros((len(ues), len(aps)))
    ap_coords = np.array([[ap.x, ap.y] for ap in aps])

    for i, ue in enumerate(ues):
        ue_coord = np.array([ue.x, ue.y])

        for j in range(len(aps)):
            d = np.linalg.norm(ue_coord - ap_coords[j])
            G[i, j] = X[i][j] * 1e-4 / (d**4)

    # associando os UEs aos APs
    for i, ue in enumerate(ues):
        # Encontra o AP com o maior ganho para este UE
        best_index = np.argmax(G[i, :])
        ue.ap = aps[best_index]     # associa o UE ao AP com maior ganho
        ue.gain = G[i, best_index]  # armazena o ganho para este UE
        ue.dist = np.linalg.norm(np.array([ue.x, ue.y]) - np.array([ue.ap.x, ue.ap.y]))     # armazena a distancia
        aps[best_index].ues.append(ue)      # adiciona o UE à lista de UEs do AP

    # vetor de potencias
    p = np.ones((len(ues), t_max))         # inicializa as potências com 1W para cada UE
    y = np.zeros((len(ues), t_max))        # vetor para armazenar os SINRs ao longo das iterações

    for t in range(t_max):
        for i, ue in enumerate(ues):
            # Calcula a interferência total para este UE
            interference = sum([p[k][t] * G[k][ue.ap.id] * R[k][ue.ap.id] for k in range(len(ues)) if k != i])
            ue.interference.append(interference)
            
            # Calcula o SINR para este UE
            sinr = (p[i][t] * ue.gain * R[i][ue.ap.id]) / (interference + pn)
            y[i][t] = sinr

            # Atualiza a potência usando o algoritmo de controle de potência
            pt = min(max(p_min, p[i][t] * y_tar / sinr), p_max)

            if t < t_max - 1:  # Evita atualizar a potência na última iteração
                p[i][t+1] = pt

    return ues, aps, p, y

'''ues, aps, p, y = simulate_noise_limited_scenario()

for i in range(4):
    print(f'UE {i+1}')
    print(f'Potência: {p[i][0]} W')
    print(f'Ganho: {ues[i].gain:.4e}')
    print(f'Interferência: {ues[i].interference[0]:.4e} W')
    print(f'Ruído: {pn:.4e} W')
    print(f'SINR: {y[i][0]:.2f}')
    print(f'Capacidade do canal: {100*np.log2(1+y[i][0]):.2f} Mbps\n')

sum_cap = sum([100*np.log2(1+y[i][0]) for i in range(4)])
p_total = sum([p[i][0] for i in range(len(p))])
print(f'Sum-capacity: {sum_cap:.4f} Mbps')
print(f'Energy Efficiency: {sum_cap/p_total:.4f} Mbits/Joule')'''
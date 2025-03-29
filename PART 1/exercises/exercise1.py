def pr(dist: float) -> float:   # Função de potência recebida
    pt=1
    k=1e-4
    n=4
    return pt*k/dist**n

d11 = 50    # Distância entre o UE1 e o AP1
d12 = 110   # Distância entre o UE1 e o AP2
d21 = 150   # Distância entre o UE2 e o AP1
d22 = 90    # Distância entre o UE2 e o AP2

UE1=pr(d11)/pr(d12)
UE2=pr(d22)/pr(d21)
print(f'SNR UE1: {UE1:.02f} \nSIR UE2: {UE2:.02f}')

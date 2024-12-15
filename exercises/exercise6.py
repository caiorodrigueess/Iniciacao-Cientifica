
from math import log2

def pr(pt: float, k: float, d: float, n: int):
    return pt*k/(d**n)

def pn(k0: float, bt: float, N: int):
    return k0*bt/N

def main():
    pt=1        # transmited power
    k=10e-4     
    n=4         # path gain
    d11 = 50
    d12 = 110
    d21 = 150
    d22 = 90
    bt=1e8      # avaiable bandwidth
    k0=10e-17   # constant for the noise power
    N = 1

    print("Item a): ")

    y11 = pr(pt, k, d11, n)/(pr(pt, k, d12, n) + pn(k0, bt, N))
    print(pr(pt, k, d11, n), pn(k0, bt, N))
    y22 = pr(pt, k, d22, n)/(pr(pt, k, d21, n) + pn(k0, bt, N))

    print(pr(pt, k, d12, n) + pn(k0, bt, N), pr(pt, k, d12, n), pn(k0, bt, N))

    print(f"SINR y1,1: {y11}")
    print(f"SINR y2,2: {y22}")

    print('--------------------------------\n')
    print("Item b):")

    c11 = bt/N*log2(1+y11)
    c22 = bt/N*log2(1+y22)
    print(bt/N, log2(1+y22))


    print(f"Channel Capacity C1,1: {(c11/10e6):.04f} Mbps")
    print(f"Channel Capacity C2,2: {(c22/10e6):.04f} Mbps")

    print('--------------------------------\n')

    print("Item c):")
    N = 2               # 2 channels

    y11 = pr(pt, k, d11, n)/pn(k0, bt, N)
    y22 = pr(pt, k, d22, n)/pn(k0, bt, N)

    c11 = bt/N*log2(1+y11)
    c22 = bt/N*log2(1+y22)
    print(bt/N, log2(1+y22))

    print(f"Channel Capacity C1,1: {(c11/10e6):.04f} Mbps")
    print(f"Channel Capacity C2,2: {(c22/10e6):.04f} Mbps")

    print('--------------------------------\n')
    print("Item d):")
    print("O melhor cenário é aquele que separa em canais.\n")

if __name__=='__main__':
    main()
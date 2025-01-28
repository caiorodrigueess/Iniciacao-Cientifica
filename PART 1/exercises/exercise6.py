from math import log2

def pr(pt: float, k: float, d: float, n: int):
    return pt*k/(d**n)

def pn(k0: float, bt: float, N: int):
    return k0*bt/N

def main():
    pt=1        # transmited power
    k=1e-4      
    n=4         # path gain
    d11 = 50    # dist ue1-ap1
    d12 = 150   # dist ue1-ap2
    d21 = 110   # dist ue2-ap1
    d22 = 90    # dist ue-ap2
    bt=1e8      # avaiable bandwidth
    k0=1e-20    # constant for the noise power
    N = 1

    print("Item a): ")

    y11 = pr(pt, k, d11, n)/(pr(pt, k, d21, n) + pn(k0, bt, N))
    y22 = pr(pt, k, d22, n)/(pr(pt, k, d12, n) + pn(k0, bt, N))
    print(f"SINR y1,1: {y11:.02f}")
    print(f"SINR y2,2: {y22:.02f}")
    
    print('--------------------------------\n')
    print("Item b):")

    c11 = (bt/N)*log2(1+y11)
    c22 = (bt/N)*log2(1+y22)

    print(f"Channel Capacity C1,1: {(c11/1e6):.04f} Mbps")
    print(f"Channel Capacity C2,2: {(c22/1e6):.04f} Mbps")

    print('--------------------------------\n')

    print("Item c):")
    N = 2               # 2 channels

    y11 = pr(pt, k, d11, n)/pn(k0, bt, N)
    y22 = pr(pt, k, d22, n)/pn(k0, bt, N)

    c11 = (bt/N)*log2(1+y11)
    c22 = (bt/N)*log2(1+y22)

    print(f"Channel Capacity C1,1: {(c11/1e6):.04f} Mbps")
    print(f"Channel Capacity C2,2: {(c22/1e6):.04f} Mbps")

    print('--------------------------------\n')
    print("Item d):")
    print("O melhor cenário é aquele que mantem no mesmo canal.\n")
    

if __name__=='__main__':
    main()
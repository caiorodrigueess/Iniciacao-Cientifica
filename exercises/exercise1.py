def pr(pt, k, d, n):
    return pt*k/d**n

def pn(k0, bt, N):
    return k0*bt/N    

def main():
    pt=1
    k=10e-4
    n=4
    N=1
    dist = [50, 150, 110, 90]
    bt=100*10e6
    k0=10e-17
    soma=0

    for i in dist:
        soma += pr(pt, k, i, n)
    
    UE1=pr(pt, k, dist[0], n)/(soma)
    UE2=pr(pt, k, dist[3], n)/(soma)
    print(f'UE1: {UE1} \nUE2: {UE2}')

if __name__=='__main__':
    main()
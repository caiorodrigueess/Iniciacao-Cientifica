def pr(pt, k, d, n):
    return pt*k/d**n

def main():
    pt=1
    k=1e-4
    n=4
    d11 = 50
    d12 = 110
    d21 = 150
    d22 = 90
    
    UE1=pr(pt, k, d11, n)/pr(pt, k, d12, n)
    UE2=pr(pt, k, d22, n)/pr(pt, k, d21, n)
    print(f'UE1: {UE1:.02f} \nUE2: {UE2:.02f}')

if __name__=='__main__':
    main()
import sys
from functools import lru_cache

sys.setrecursionlimit(10**5)  

def solve():
    data = sys.stdin.read().split()
    n = int(data[0])
    x = int(data[1])
    a = list(map(int, data[2:2+n]))
    
    if x == 0:
        print(0)
        return
    
    @lru_cache(None)
    def dp(idx, rem):
        if rem == 0:
            return 0

        if idx == 0:          
            return rem
        
        curr = a[idx]
        count = rem // curr
        leftover = rem % curr
        
        if leftover == 0:
            return count + dp(idx - 1, 0)   
        
        res1 = count + dp(idx - 1, leftover)
        res2 = (count + 1) + dp(idx - 1, curr - leftover)
        
        return min(res1, res2)
    
    print(dp(n - 1, x))

if __name__ == "__main__":
    solve()
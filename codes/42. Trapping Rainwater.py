from typing import List

class Solution:
    def trap(self, height: List[int]) -> int:
        n = len(height)
        if n == 0:
            return 0
        l,r,lmax,rmax,res=0,n-1,0,0,0
        while(l<=r):
            if(lmax<rmax):
                res += max(0,lmax-height[l])
                lmax = max(lmax,height[l])
                l+=1
            else:
                res += max(0,rmax-height[r])
                rmax = max(rmax,height[r])
                r-=1
        
        return res

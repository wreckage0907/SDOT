class Solution:
    def lengthOfLongestSubstring(self, s: str) -> int:
        m=[-1]*256
        leng,r,l=0,0,0
        while(r<len(s)):
            if(m[ord(s[r])]!=-1 and l<m[ord(s[r])]+1):
                l=m[ord(s[r])]+1
            m[ord(s[r])]=r
            leng=max(leng,r-l+1)
            r=r+1
        return leng
        
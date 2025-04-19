class Solution:
    def longestPrefix(self, s: str) -> str:
        pre = [0]*len(s)
        i=0
        j=1
        while j<len(s):
            if(s[i]==s[j]):
                i=i+1
                pre[j]=i
                j=j+1
            else:
                if i==0:
                    j=j+1
                else:
                    i=pre[i-1]
                    
        return s[:i]

        
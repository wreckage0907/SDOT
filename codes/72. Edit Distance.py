class Solution:
    def minDistance(self, word1: str, word2: str) -> int:
        n, m = len(word1), len(word2)
        dp = [i for i in range(n+1)]
        for i in range(1, m+1):
            curr = [0] * (n+1)
            curr[0] = i
            for j in range(1, n+1):
                if word1[j-1] == word2[i-1]:
                    curr[j] = dp[j-1]
                else:
                    curr[j] = 1 + min(dp[j], dp[j-1], curr[j-1])
            dp = curr
        return dp[n]
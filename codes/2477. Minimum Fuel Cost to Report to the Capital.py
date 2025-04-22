class Solution:
    def minimumFuelCost(self, roads: List[List[int]], seats: int) -> int:
        n=len(roads)+1
        graph =[[] for _ in range(n)]
        for u,v in roads:
            graph[u].append(v)
            graph[v].append(u)
        self.ans=0
        def dfs(node,parent):
            people=1
            for child in graph[node]:
                if child != parent:
                    people += dfs(child,node)
            if node!= 0:
                self.ans += (people+seats-1) // seats
            return people

        dfs(0,-1)
        return self.ans        
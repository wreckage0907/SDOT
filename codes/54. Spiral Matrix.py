class Solution:
    def spiralOrder(self, matrix: List[List[int]]) -> List[int]:
        res=[]
        if not matrix or not matrix[0]:
            return res
        r,c=len(matrix),len(matrix[0])
        sr,sc,er,ec=0,0,r-1,c-1
        cnt,total=0,r*c

        while cnt<total:
            for j in range(sc,ec+1):
                if cnt<total:
                    res.append(matrix[sr][j])
                    cnt+=1
            sr+=1            

            for i in range(sr,er+1):
                if cnt<total:
                    res.append(matrix[i][ec])
                    cnt+=1
            
            ec -= 1

            for j in range(ec,sc-1,-1):
                if cnt<total:
                    res.append(matrix[er][j])
                    cnt+=1
            er-=1
            for i in range(er,sr-1,-1):
                if cnt<total:
                    res.append(matrix[i][sc])
                    cnt+=1
            sc+=1
        return res
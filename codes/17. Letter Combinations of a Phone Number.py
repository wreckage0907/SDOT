class Solution:
    def letterCombinations(self, digits: str) -> List[str]:
        res=[]
        letter=["","","abc","def","ghi", "jkl", "mno", "pqrs", "tuv", "wxyz"]
        def backtrack(i,curr):
            if(len(curr)==len(digits)):
                res.append(curr)
                return
            for c in letter[int(digits[i])]:
                backtrack(i+1,curr+c)
        
        backtrack(0,"")
        return res if res[0] else []
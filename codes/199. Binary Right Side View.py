# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def rightSideView(self, root: Optional[TreeNode]) -> List[int]:
        def recursiveright(root,level,mx,res):
            if root is None: return
            if level> mx[0]:
                result.append(root.val)
                mx[0]=level
            recursiveright(root.right,level+1,mx,res)
            recursiveright(root.left,level+1,mx,res)
        result=[]
        mx=[-1]
        recursiveright(root,0,mx,result)
        return result
        
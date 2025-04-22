# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def diameterOfBinaryTree(self, root: Optional[TreeNode]) -> int:
        def dfs(root,diam):
            if(not root): return 0
            lh = dfs(root.left,diam)
            rh = dfs(root.right,diam)
            diam[0] = max(lh+rh,diam[0])

            return 1+max(lh,rh)
        
        diam=[0]
        dfs(root,diam)
        return diam[0]
        
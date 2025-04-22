# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def kthSmallest(self, root: Optional[TreeNode], k: int) -> int:
        def inorder(root,ans):
            if(root is None): return
            inorder(root.left,ans)
            ans.append(root.val)
            inorder(root.right,ans)
        
        ans=[]
        inorder(root,ans)
        return ans[k-1]
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def zigzagLevelOrder(self, root: Optional[TreeNode]) -> List[List[int]]:
        if not root: return []
        result=[]
        queue = deque([root])
        flag = True

        while queue:
            level = len(queue)
            current = [0]*level
            for i in range(level):
                node = queue.popleft()
                index = i if flag else level-1-i
                current[index]=node.val
                if node.left:
                    queue.append(node.left)
                if node.right:
                    queue.append(node.right)
            result.append(current)
            flag = not flag
        return result
        
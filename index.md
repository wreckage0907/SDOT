---
layout: default
title: Home
---

<details>
<summary>0. Implementation of LinkedList</summary>

<pre><code class="language-python">
class Node:
    def __init__(self,data):
        self.data=data
        self.next=None

class LinkedList:
    def __init__(self):
        self.head=None
        self.tail=None
    def insert(self,data):
        newNode = Node(data);
        if self.head is None:
            self.head=newNode
            self.tail=newNode
        else:
            self.tail.next=newNode
            self.tail=newNode
    def insertatbeg(self,data):
        newNode = Node(data);
        if self.head is None:
            self.head=newNode
            self.tail=newNode
        else:
            newNode.next=self.head
            self.head=newNode
    def insertatmid(self,pos,data):
        newNode = Node(data);
        if pos==0:
            self.insertatbeg(data)
        else:
            temp=self.head
            for i in range(pos-1):
                temp=temp.next
            newNode.next=temp.next
            temp.next=newNode
    def reverse(self):
        curr = self.head
        prev = None
        future = None
        while curr is not None:
            future = curr.next
            curr.next = prev
            prev = curr
            curr = future
        self.head=prev
    def print(self):
        temp=self.head
        while temp is not None:
            print(temp.data,end=" ")
            temp=temp.next

if __name__=="__main__":
    ll=LinkedList()
    ele = list(map(int,input().split()))
    for e in ele:
        if e==-1:
            break
        ll.insertatbeg(e)
    ll.print();
    ll.reverse()
    print("\n")
    ll.print()

</code></pre>
</details>
---

<details>
<summary>103. Binary Tree Zig Zag Traversal</summary>

<pre><code class="language-python">
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
        
</code></pre>
</details>
---

<details>
<summary>109. Converted SortedList to BST</summary>

<pre><code class="language-python">
class Solution:
    def sortedListToBST(self, head: ListNode) -> TreeNode:
        if not head:
            return None
        if not head.next:
            return TreeNode(head.val)

        slow, fast = head, head
        slow_prev = None

        # Find the middle node (slow pointer)
        while fast and fast.next:
            slow_prev = slow
            slow = slow.next
            fast = fast.next.next

        # Create root node from the middle element
        root = TreeNode(slow.val)

        # Disconnect the left half from the middle
        slow_prev.next = None

        # Recursively construct left and right subtrees
        root.left = self.sortedListToBST(head)
        root.right = self.sortedListToBST(slow.next)

        return root
</code></pre>
</details>
---

<details>
<summary>114. Flatten Binary Tree to Linkedlist</summary>

<pre><code class="language-python">
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def flatten(self, root: Optional[TreeNode]) -> None:
        """
        Do not return anything, modify root in-place instead.
        """
        curr=root
        while(curr != None):
            if(curr.left != None):
                prev = curr.left
                while(prev.right != None): prev = prev.right
                prev.right=curr.right
                curr.right=curr.left
                curr.left=None
            curr = curr.right
        
        
</code></pre>
</details>
---

<details>
<summary>129. Sum Root to Leaf Numbers</summary>

<pre><code class="language-python">
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def sumNumbers(self, root: Optional[TreeNode]) -> int:
        def summ(root,curr):
            if(root is None): return 0
            curr=curr*10+root.val
            if(root.left is None and root.right is None): return curr
            return summ(root.left,curr)+summ(root.right,curr)
        return summ(root,0)
        
</code></pre>
</details>
---

<details>
<summary>1392. Longest Happy Prefix</summary>

<pre><code class="language-python">
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

        
</code></pre>
</details>
---

<details>
<summary>143. Reorder List</summary>

<pre><code class="language-python">
class Solution:
    def reorderList(self, head):
        if not head or not head.next:
            return
        
        # Step 1: Find the middle of the list
        slow, fast = head, head
        while fast and fast.next:
            slow = slow.next
            fast = fast.next.next
        
        # Step 2: Reverse the second half
        prev, curr = None, slow.next
        slow.next = None  # Disconnect the two halves
        while curr:
            temp = curr.next
            curr.next = prev
            prev = curr
            curr = temp
        
        # Step 3: Merge the two halves
        first, second = head, prev
        while second:
            temp1, temp2 = first.next, second.next
            first.next = second
            second.next = temp1
            first, second = temp1, temp2
</code></pre>
</details>
---

<details>
<summary>150. Evaluate Reverse Polish Notation</summary>

<pre><code class="language-python">
class Solution:
    def evalRPN(self, tokens: List[str]) -> int:
        st = []

        for c in tokens:
            if c == "+":
                st.append(st.pop() + st.pop())
            elif c == "-":
                #second, first = st.pop(), st.pop()
                st.append(-1*st.pop()+st.pop())
            elif c == "*":
                st.append(st.pop() * st.pop())
            elif c == "/":
                second, first = st.pop(), st.pop()
                st.append(int(first / second))                
            else:
                st.append(int(c))
        
        return st[0]
</code></pre>
</details>
---

<details>
<summary>199. Binary Right Side View</summary>

<pre><code class="language-python">
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
        
</code></pre>
</details>
---

<details>
<summary>200. Number of Islands</summary>

<pre><code class="language-python">
class Solution:
    def numIslands(self, grid: List[List[str]]) -> int:
        if(len(grid)==0 or len(grid[0])==0): return 0
        row,col,islands=len(grid),len(grid[0]),0
        def dfs(r,c):
            if(r<0 or  c<0 or r>=row or c>=col or grid[r][c]!='1' ): return
            grid[r][c]='0'
            dfs(r-1,c)
            dfs(r+1,c)
            dfs(r,c-1)
            dfs(r,c+1)
        
        for r in range(row):
            for c in range(col):
                if(grid[r][c]=='1'):
                    dfs(r,c)
                    islands+=1
        
        return islands       
</code></pre>
</details>
---

<details>
<summary>207. Course Schedule</summary>

<pre><code class="language-python">
from typing import List

class Solution:
    def dfs(self, i: int, adj: List[List[int]], hash: set, visited: List[bool]) -> bool:
        hash.add(i)
        visited[i] = True

        for neighbor in adj[i]:
            if not visited[neighbor]:
                if not self.dfs(neighbor, adj, hash, visited):
                    return False
            elif neighbor in hash:  
                return False

        hash.remove(i)
        return True

    def canFinish(self, numCourses: int, prerequisites: List[List[int]]) -> bool:
        adj = [[] for _ in range(numCourses)]
        for dest, src in prerequisites:
            adj[src].append(dest)

        visited = [False] * numCourses
        for i in range(numCourses):
            if not visited[i]:
                hash = set()
                if not self.dfs(i, adj, hash, visited):
                    return False

        return True
</code></pre>
</details>
---

<details>
<summary>21. Merge Sorted Linked List</summary>

<pre><code class="language-python">
class Solution:
    def mergeTwoLists(self, list1: Optional[ListNode], list2: Optional[ListNode]) -> Optional[ListNode]:
        if list1 is None : return list2
        if list2 is None : return list1
        temp = ListNode(-1)
        head = temp
        while list1 and list2:
            if list1.val<list2.val:
                temp.next=list1
                list1=list1.next
            else:
                temp.next=list2
                list2=list2.next
            temp=temp.next
        if list1: temp.next=list1
        if list2: temp.next=list2
        return head.next
</code></pre>
</details>
---

<details>
<summary>224. Basic Calculator</summary>

<pre><code class="language-python">
class Solution:
    def calculate(self, s: str) -> int:
        number = 0
        sign_value = 1
        result = 0
        operations_stack = []

        for c in s:
            if c.isdigit():
                number = number * 10 + int(c)
            elif c in "+-":
                result += number * sign_value
                sign_value = -1 if c == '-' else 1
                number = 0
            elif c == '(':
                operations_stack.append(result)
                operations_stack.append(sign_value)
                result = 0
                sign_value = 1
            elif c == ')':
                result += sign_value * number
                result *= operations_stack.pop()
                result += operations_stack.pop()
                number = 0

        return result + number * sign_value
</code></pre>
</details>
---

<details>
<summary>225. Implementing Stack using Queues</summary>

<pre><code class="language-python">
class MyStack:

    def __init__(self):
        self.queue=[]        

    def push(self, x: int) -> None:
        self.queue.append(x)

    def pop(self) -> int:
        return self.queue.pop()

    def top(self) -> int:
        return self.queue[-1]

    def empty(self) -> bool:
        return len(self.queue)==0
        


# Your MyStack object will be instantiated and called as such:
# obj = MyStack()
# obj.push(x)
# param_2 = obj.pop()
# param_3 = obj.top()
# param_4 = obj.empty()
</code></pre>
</details>
---

<details>
<summary>23. Merge K Sorted LinkedList</summary>

<pre><code class="language-python">
from typing import List, Optional
class Solution:
    def mergeTwoLists(self, l1: Optional[ListNode], l2: Optional[ListNode]) -> Optional[ListNode]:
        if not l1:
            return l2
        if not l2:
            return l1

        if l1.val < l2.val:
            l1.next = self.mergeTwoLists(l1.next, l2)
            return l1
        else:
            l2.next = self.mergeTwoLists(l1, l2.next)
            return l2

    def mergeKLists(self, lists: List[Optional[ListNode]]) -> Optional[ListNode]:
        if not lists:
            return None
        return self.divideAndConquer(lists, 0, len(lists) - 1)

    def divideAndConquer(self, lists: List[Optional[ListNode]], left: int, right: int) -> Optional[ListNode]:
        if left == right:
            return lists[left]

        mid = left + (right - left) // 2
        l1 = self.divideAndConquer(lists, left, mid)
        l2 = self.divideAndConquer(lists, mid + 1, right)
        return self.mergeTwoLists(l1, l2)
</code></pre>
</details>
---

<details>
<summary>230. Kth smallest element in BST</summary>

<pre><code class="language-python">
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
</code></pre>
</details>
---

<details>
<summary>232. Implementing Queue using Stacks</summary>

<pre><code class="language-python">
class MyQueue:

    def __init__(self):
        self.stack = []

    def push(self, x: int) -> None:
        self.stack.append(x)

    def pop(self) -> int:
        top = self.stack[0]
        self.stack.remove(self.stack[0])
        return top

    def peek(self) -> int:
        return self.stack[0]

    def empty(self) -> bool:
        return len(self.stack) == 0
</code></pre>
</details>
---

<details>
<summary>234. Palindromic Linked List</summary>

<pre><code class="language-python">
class Solution:
    def isPalindrome(self, head: Optional[ListNode]) -> bool:
        if not head or not head.next:
            return True
        
        slow, fast = head, head
        while fast and fast.next:
            slow = slow.next
            fast = fast.next.next
        
        prev = None
        while slow:
            temp = slow.next
            slow.next = prev
            prev = slow
            slow = temp
        

        left, right = head, prev
        while right:  
            if left.val != right.val:
                return False
            left = left.next
            right = right.next
        
        return True
</code></pre>
</details>
---

<details>
<summary>235. Lowest Common Ancestor of BST</summary>

<pre><code class="language-python">
 # Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    def lowestCommonAncestor(self, root: 'TreeNode', p: 'TreeNode', q: 'TreeNode') -> 'TreeNode':
        if(root is None): return None
        if(p.val<root.val and q.val <root.val): return self.lowestCommonAncestor(root.left,p,q)
        if(p.val>root.val and q.val >root.val): return self.lowestCommonAncestor(root.right,p,q)
        return root
        
</code></pre>
</details>
---

<details>
<summary>2477. Minimum Fuel Cost to Report to the Capital</summary>

<pre><code class="language-python">
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
</code></pre>
</details>
---

<details>
<summary>25. Reverse Nodes in k-Group</summary>

<pre><code class="language-python">
class Solution:
    def reverseKGroup(self, head: ListNode, k: int) -> ListNode:
        if not head or k == 1:
            return head

        dummy = ListNode(0)
        dummy.next = head
        prev = dummy
        curr = head

        # Count the number of nodes in the list
        count = 0
        while curr:
            count += 1
            curr = curr.next

        # Reverse k nodes at a time
        while count >= k:
            curr = prev.next
            nxt = curr.next

            # Reverse k nodes
            for _ in range(1, k):
                curr.next = nxt.next
                nxt.next = prev.next
                prev.next = nxt
                nxt = curr.next

            prev = curr
            count -= k

        return dummy.next
</code></pre>
</details>
---

<details>
<summary>3. Longest Substring Without Repeating Characters</summary>

<pre><code class="language-python">
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
        
</code></pre>
</details>
---

<details>
<summary>32. Longest Valid Parantheses</summary>

<pre><code class="language-python">
class Solution:
    def longestValidParentheses(self, s: str) -> int:
        stack = [-1]
        max_len = 0

        for i in range(len(s)):
            if s[i] == "(":
                stack.append(i)
            else:
                stack.pop()
                if len(stack) == 0:
                    stack.append(i)
                else:
                    max_len = max(max_len, i - stack[-1])
        
        return max_len
</code></pre>
</details>
---

<details>
<summary>328. Odd Even Linked List</summary>

<pre><code class="language-python">
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next

class Solution:
    def oddEvenList(self, head: Optional[ListNode]) -> Optional[ListNode]:
        if head is None or head.next is None or head.next.next is None:
            return head

        odd = head
        even = head.next
        temp = even

        while even and even.next:
            odd.next = even.next
            odd = odd.next
            even.next = odd.next
            even = even.next

        odd.next = temp
        return head

</code></pre>
</details>
---

<details>
<summary>543. Diameter of Binary Tree</summary>

<pre><code class="language-python">
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
        
</code></pre>
</details>
---

<details>
<summary>61. Rotate List</summary>

<pre><code class="language-python">
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def rotateRight(self, head: Optional[ListNode], k: int) -> Optional[ListNode]:
        if head is None or head.next is None or k==0:
            return head
        
        s=1
        temp=head
        tail=head
        while temp.next:
            temp=temp.next
            tail=temp
            s=s+1

        k = k % s
        if k==0: return head
        tail.next = head

        for i in range(s-k):
            tail = head
            head = head.next
        tail.next=None
        return head

</code></pre>
</details>
---

<details>
<summary>98. Validate BST</summary>

<pre><code class="language-python">
class Solution:
    def isValidBST(self, root: Optional[TreeNode]) -> bool:
        def helper(node, low, high):
            if not node:
                return True
            if low < node.val < high:
                return helper(node.left, low, node.val) and helper(node.right, node.val, high)
            return False
        return helper(root, float("-inf"), float("+inf"))
</code></pre>
</details>
---

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
<summary>1095. Find in Mountain Array</summary>

<pre><code class="language-python">
class Solution:
    def findInMountainArray(self, target: int, mountain_arr: 'MountainArray') -> int:
        length = mountain_arr.length()

        def find_target(left, right, target, is_upside):
            while left <= right:
                mid = (left + right) // 2
                mid_val = mountain_arr.get(mid)

                if mid_val == target:
                    return mid
                
                if is_upside:
                    if target > mid_val:
                        left = mid + 1
                    else:
                        right = mid - 1
                else:
                    if target > mid_val:
                        right = mid - 1
                    else:
                        left = mid + 1

            return -1

        def find_peak():
            nonlocal length

            left, right = 0, length - 1

            while left < right:
                mid = (left + right) // 2
                if mountain_arr.get(mid) < mountain_arr.get(mid + 1):
                    left = mid + 1
                else:
                    right = mid
            
            return left

        peak_index = find_peak()


        result = find_target(0, peak_index, target, True)
        if result != -1:
            return result
        
        return find_target(peak_index + 1, length - 1, target, False)        
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
<summary>1358. Number of Substrings Containing All 3 Characters</summary>

<pre><code class="language-python">
class Solution:
    def numberOfSubstrings(self, s: str) -> int:
        count = 0
        left = 0
        char_count = {'a': 0, 'b': 0, 'c': 0}
        
        for right in range(len(s)):
            char_count[s[right]] += 1
            
            while char_count['a'] > 0 and char_count['b'] > 0 and char_count['c'] > 0:
                count += len(s) - right
                char_count[s[left]] -= 1
                left += 1
        
        return count
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
<summary>17. Letter Combinations of a Phone Number</summary>

<pre><code class="language-python">
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
</code></pre>
</details>
---

<details>
<summary>198. House Robber</summary>

<pre><code class="language-python">
class Solution:
    def rob(self, nums: List[int]) -> int:
        rob,norob=0,0
        for i in range(len(nums)):
            newrob = norob+nums[i]
            newNorob = max(norob,rob)
            rob = newrob
            norob = newNorob
        return max(rob,norob)
        
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
<summary>214. Shortest Pallindrome</summary>

<pre><code class="language-python">
class Solution:
    def shortestPalindrome(self, s: str) -> str:
        count = self.kmp(s[::-1], s)
        return s[count:][::-1] + s
    def kmp(self, txt: str, patt: str) -> int:
        new_string = patt + '#' + txt
        pi = [0] * len(new_string)
        i = 1
        k = 0
        while i < len(new_string):
            if new_string[i] == new_string[k]:
                k += 1
                pi[i] = k
                i += 1
            else:
                if k > 0:
                    k = pi[k - 1]
                else:
                    pi[i] = 0
                    i += 1
        return pi[-1]
</code></pre>
</details>
---

<details>
<summary>22. Generate Parentheses</summary>

<pre><code class="language-python">
class Solution:
    def generateParenthesis(self, n: int) -> List[str]:
        res = []

        def dfs(openP, closeP, s):
            if openP == closeP and openP + closeP == n * 2:
                res.append(s)
                return
            
            if openP < n:
                dfs(openP + 1, closeP, s + "(")
            
            if closeP < openP:
                dfs(openP, closeP + 1, s + ")")

        dfs(0, 0, "")

        return res
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
<summary>2344. Minimum Deletions to Make Array Divisible</summary>

<pre><code class="language-python">
class Solution:
    def minOperations(self, nums: List[int], numsDivide: List[int]) -> int:
        def find_gcd(arr):
            return reduce(gcd, arr)

        target_gcd = find_gcd(numsDivide)
        nums.sort()

        for i, num in enumerate(nums):
            if target_gcd % num == 0:
                return i 
        return -1  

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
<summary>39. Combination Sum</summary>

<pre><code class="language-python">
class Solution:
    def combinationSum(self, candidates: List[int], target: int) -> List[List[int]]:
        def backtrack(remaining, combination, start):
            if remaining == 0:
                result.append(list(combination))
                return
            if remaining < 0:
                return

            for i in range(start, len(candidates)):
                combination.append(candidates[i])
                backtrack(remaining - candidates[i], combination, i)
                combination.pop()

        result = []
        backtrack(target, [], 0)
        return result
</code></pre>
</details>
---

<details>
<summary>42. Trapping Rainwater</summary>

<pre><code class="language-python">
from typing import List

class Solution:
    def trap(self, height: List[int]) -> int:
        n = len(height)
        if n == 0:
            return 0
        l,r,lmax,rmax,res=0,n-1,0,0,0
        while(l<=r):
            if(lmax<rmax):
                res += max(0,lmax-height[l])
                lmax = max(lmax,height[l])
                l+=1
            else:
                res += max(0,rmax-height[r])
                rmax = max(rmax,height[r])
                r-=1
        
        return res

</code></pre>
</details>
---

<details>
<summary>44. Wildcard Matching</summary>

<pre><code class="language-python">
class Solution:
    def isMatch(self, s: str, p: str) -> bool:
        m, n = len(s), len(p)
        dp = [[False for _ in range(n + 1)] for i in range(m + 1)]
        dp[0][0] = True
        for j in range(1, n + 1):
            if p[j - 1] != "*":
                break
            dp[0][j] = True
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if p[j - 1] == s[i - 1] or p[j - 1] == "?":
                    dp[i][j] = dp[i - 1][j - 1]
                elif p[j - 1] == "*":
                    dp[i][j] = dp[i - 1][j] or dp[i][j - 1]
        return dp[m][n]
</code></pre>
</details>
---

<details>
<summary>46. Permutations</summary>

<pre><code class="language-python">
class Solution:
    def permute(self, nums: List[int]) -> List[List[int]]:
        if len(nums) == 1:
            return [nums[:]]
        
        res = []

        for _ in range(len(nums)):
            n = nums.pop(0)
            perms = self.permute(nums)

            for p in perms:
                p.append(n)
            
            res.extend(perms)
            nums.append(n)
        
        return res
            
</code></pre>
</details>
---

<details>
<summary>5. Longest Palindromic Substring</summary>

<pre><code class="language-python">
class Solution:
    def longestPalindrome(self, s: str) -> str:
        if not s:
            return ""

        def expand_around_center(s: str, left: int, right: int):
            while left >= 0 and right < len(s) and s[left] == s[right]:
                left -= 1
                right += 1
            return right - left - 1


        start = 0
        end = 0

        for i in range(len(s)):
            odd = expand_around_center(s, i, i)
            even = expand_around_center(s, i, i + 1)
            max_len = max(odd, even)
            
            if max_len > end - start:
                start = i - (max_len - 1) // 2
                end = i + max_len // 2
        
        return s[start:end+1]
</code></pre>
</details>
---

<details>
<summary>54. Spiral Matrix</summary>

<pre><code class="language-python">
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
<summary>673. Number of Longest Increasing Subsequence</summary>

<pre><code class="language-python">
class Solution:
    def findNumberOfLIS(self, nums: List[int]) -> int:
        n = len(nums)
        if n <= 1:
            return n

        lengths = [1] * n
        counts = [1] * n

        for i in range(1, n):
            for j in range(i):
                if nums[i] > nums[j]:
                    if lengths[j] + 1 > lengths[i]:
                        lengths[i] = lengths[j] + 1
                        counts[i] = counts[j]
                    elif lengths[j] + 1 == lengths[i]:
                        counts[i] += counts[j]

        max_length = max(lengths)
        return sum(count for length, count in zip(lengths, counts) if length == max_length)
</code></pre>
</details>
---

<details>
<summary>72. Edit Distance</summary>

<pre><code class="language-python">
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

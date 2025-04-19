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

---

<details>
<summary>21. Merge Sorted Linked List.py</summary>

```python
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
        return head.next```
</details>
---

<details>
<summary>234. Palindromic Linked List.py</summary>

```python
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
        
        return True```
</details>
---

<details>
<summary>LinkedList.py</summary>

```python
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
```
</details>
---

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

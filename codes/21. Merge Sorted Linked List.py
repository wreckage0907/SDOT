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
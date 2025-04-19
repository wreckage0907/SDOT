---
layout: default
title: Home
---

# SDOT is boring

_SDOD IS BORING !!!_

[View on GitHub](https://github.com/wreckage0907/SDOT)

---

<details>
<summary>Click to view LinkedList implementation</summary>

<pre><code class="language-python">
class Node:
    def __init__(self, data):
        self.data = data
        self.next = None

class LinkedList:
    def __init__(self):
        self.head = None
        self.tail = None

    def insert(self, data):
        newNode = Node(data)
        if self.head is None:
            self.head = newNode
            self.tail = newNode
        else:
            self.tail.next = newNode
            self.tail = newNode

    def insertatbeg(self, data):
        newNode = Node(data)
        if self.head is None:
            self.head = newNode
            self.tail = newNode
        else:
            newNode.next = self.head
            self.head = newNode

    def insertatmid(self, pos, data):
        newNode = Node(data)
        if pos == 0:
            self.insertatbeg(data)
        else:
            temp = self.head
            for i in range(pos - 1):
                temp = temp.next
            newNode.next = temp.next
            temp.next = newNode

    def reverse(self):
        curr = self.head
        prev = None
        while curr is not None:
            future = curr.next
            curr.next = prev
            prev = curr
            curr = future
        self.head = prev

    def print(self):
        temp = self.head
        while temp is not None:
            print(temp.data, end=" ")
            temp = temp.next

if __name__ == "__main__":
    ll = LinkedList()
    ele = list(map(int, input().split()))
    for e in ele:
        if e == -1:
            break
        ll.insertatbeg(e)
    ll.print()
    ll.reverse()
    print("\n")
    ll.print()
</code></pre>

</details>


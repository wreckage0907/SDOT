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

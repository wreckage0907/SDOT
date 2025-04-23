class Solution:
    def rob(self, nums: List[int]) -> int:
        rob,norob=0,0
        for i in range(len(nums)):
            newrob = norob+nums[i]
            newNorob = max(norob,rob)
            rob = newrob
            norob = newNorob
        return max(rob,norob)
        
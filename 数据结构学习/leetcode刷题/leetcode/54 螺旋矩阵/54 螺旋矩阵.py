# -*- coding: utf-8 
from typing import List

class Solution:

    def spiralOrder(self, matrix: List[List[int]]) -> List[int]:
        
        # def check_next(direction, i, j):
        #     if direction == "left":
        #         next_i = i+1
        #         next_j = j
        #     elif direction == "right":
        #         next_i = i-1
        #         next_j = j
        #     elif direction == "down":
        #         next_i = i-1
        #         next_j = j 
        #     elif direction == "up":
        #         next_i = i+1
        #         next_j = j

        pass 


if __name__ == "__main__":
    test_1  = [[1,2,3],[4,5,6],[7,8,9]]
    test_1_rst = [1,2,3,6,9,8,7,4,5]
    assert Solution().spiralOrder(test_1) == test_1_rst 

    test_2 = [[1,2,3,4],[5,6,7,8],[9,10,11,12]]
    test_2_rst = [1,2,3,4,8,12,11,10,9,5,6,7]
    assert Solution().spiralOrder(test_2) == test_2_rst 

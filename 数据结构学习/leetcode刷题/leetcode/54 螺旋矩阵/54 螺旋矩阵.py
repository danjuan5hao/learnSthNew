# -*- coding: utf-8 
from typing import List

class Solution:

    def spiralOrder(self, matrix: List[List[int]]) -> List[int]:
        rst = []
        i = 0
        j = -1
        row_step = len(matrix[0])
        col_step = len(matrix)-1

        direction = "right"

        while True:                     
            if direction ==  "right" and row_step !=0:
                for _ in range(row_step):
                    j += 1
                    rst.append(matrix[i][j])
                    
                direction = "down"
                row_step -= 1
            elif direction == "down" and col_step !=0:
                for _ in range(col_step):
                    i += 1
                    rst.append(matrix[i][j])
                direction = "left"
                col_step -= 1
            elif direction == "left" and row_step !=0:
                for _ in range(row_step):
                    j -= 1
                    rst.append(matrix[i][j])
                direction = "up"
                row_step -= 1
            elif direction == "up" and col_step !=0:
                for _ in range(col_step):
                    i -= 1
                    rst.append(matrix[i][j])
                direction = "right"
                col_step -= 1
            else:
                break

        return rst


if __name__ == "__main__":
    test_1  = [[1,2,3],[4,5,6],[7,8,9]]
    test_1_rst = [1,2,3,6,9,8,7,4,5]
    # assert Solution().spiralOrder(test_1) == test_1_rst 
    print(Solution().spiralOrder(test_1))

    test_2 = [[1,2,3,4],[5,6,7,8],[9,10,11,12]]
    test_2_rst = [1,2,3,4,8,12,11,10,9,5,6,7]
    # assert Solution().spiralOrder(test_2) == test_2_rst 
    print(Solution().spiralOrder(test_2))

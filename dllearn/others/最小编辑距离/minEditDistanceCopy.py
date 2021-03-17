# -*- coding: utf-8 -*-
def med_re_aux(str1, str2):
    if str1[-1] == str2[-1]:
        return med_re(str1[:-1], str2[:-1])
    else:
        return min(med_re(str1[:-1], str2) + 1,
                    med_re(str1, str2[:-1]) + 1,
                    med_re(str1[:-1], str2[:-1]) + 1)
    
def med_re(str1, str2):
    if len(str1) == 0 or len(str2) == 0:
        return max(len(str1), len(str2))
    else:
        return med_re_aux(str1, str2) 

def med_dg(str1, str2):
    """
    """
    matrix = [[ i + j for j in range(len(str2) + 1)] for i in range(len(str1) + 1)]

    for i in range(1, len(str1)+1):
        for j in range(1, len(str2)+1):
            if(str1[i-1] == str2[j-1]):
                d = 0
            else:
                d = 1
            
            matrix[i][j] = min(matrix[i-1][j]+1, matrix[i][j-1]+1, matrix[i-1][j-1]+d)

    return matrix[len(str1)][len(str2)]

    



if __name__ == "__main__":
    import minEditDistanceCSDN

    with open(r"D:\projects\learnSthNew\最小编辑距离\最小编辑距离测试样例.txt", "r", encoding="utf-8") as f:
        for line in f:
            sent1, sent2 = line.strip().split("__##__")
            print(med_re(sent1, sent2))
            print(minEditDistanceCSDN.Levenshtein_Distance(sent1, sent2))
            print("*"*20)
# -*- coding: utf-8 -*-
class DisStr(str):
    def __init__(self, string):
        super(DisStr, self).__init__(string)

    def _med_aux(self, disStr2):
        if self[-1] == disStr2[-1]:
            return self.med(disStr2[:-1])
        else:
            return min(self.med(disStr2) + 1,
                       self.med(str1, str2[:-1]) + 1,
                        med(str1[:-1], str2[:-1]) + 1)
    

    def med(self, disStr2):
        if len(str1) == 0 or len(str2) == 0:
            return max(len(str1), len(str2))
        else:
            return med_aux(str1, str2)

    # def minEditDistance(self, another):
    #     return 



if __name__ == "__main__":
    import minEditDistanceCSDN

    with open(r"D:\projects\learnSthNew\最小编辑距离\最小编辑距离测试样例.txt", "r", encoding="utf-8") as f:
        for line in f:
            sent1, sent2 = line.strip().split("__##__")
            print()
            break
from ch2_3_3 import NAND, OR, AND

def XOR(x1, x2):
    s1 = NAND(x1, x2)
    s2 = OR(x1, x2)
    # 単層パーセプトロンの2出力を単層パーセプトロンの入力
    # = パーセプトロン2層でXORを表現
    return AND(s1, s2)

# 0
print(XOR(0, 0))
# 1
print(XOR(1, 0))
# 1
print(XOR(0, 1))
# 0
print(XOR(1, 1))

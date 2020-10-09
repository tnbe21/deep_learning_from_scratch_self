from ch5_4_1_mul_layer import MulLayer


class AddLayer:
    """
    加算レイヤ
    """
    def __init__(self):
        pass

    def forward(self, x, y):
        out = x + y
        return out

    def backword(self, dout):
        dx = dout * 1
        dy = dout * 1
        return dx, dy


if __name__ == '__main__':
    apple = 100
    apple_num = 2
    orange = 150
    orange_num = 3
    tax = 1.1

    mul_apple_layer = MulLayer()
    mul_orange_layer = MulLayer()
    add_apple_orange_layer = AddLayer()
    mul_tax_layer = MulLayer()

    # forward
    apple_price = mul_apple_layer.forward(apple, apple_num)
    orange_price = mul_orange_layer.forward(orange, orange_num)
    added_price = add_apple_orange_layer.forward(apple_price, orange_price)
    price = mul_tax_layer.forward(added_price, tax)

    print(f"price: {price}")

    # backword
    dprice = 1
    dadded_price, dtax = mul_tax_layer.backword(dprice)
    dapple_price, dorange_price = add_apple_orange_layer.backword(dadded_price)
    dapple, dapple_num = mul_apple_layer.backword(dapple_price)
    dorange, dorange_num = mul_orange_layer.backword(dorange_price)

    print(f"price: {price}")
    print(f"dadded_price: {dadded_price}")
    print(f"dtax: {dtax}")

    print(f"dapple_price: {dapple_price}")
    print(f"dapple_num: {dapple_num}")
    print(f"dapple: {dapple}")

    print(f"dorange_price: {dorange_price}")
    print(f"dorange_num: {dorange_num}")
    print(f"dorange: {dorange}")

import random

if __name__ == "__main__":
    nTests = 72
    seller_price_lower = [1, 2, 3]
    seller_price_upper = [4, 5, 6]
    seller_volume = [7, 8, 9]
    buyer_price = [10, 11, 12]
    buyer_volume = [13, 14, 15]
    seller_name = [1, 2, 3]
    buyer_name = [1, 2, 3]
    seller_data = list(zip(seller_price_lower, seller_price_upper, seller_volume, seller_name))
    buyer_data = list(zip(buyer_price, buyer_volume, buyer_name))
    for itr in range(nTests):
        random.shuffle(seller_data)
        random.shuffle(buyer_data)
        seller_data = sorted(seller_data)  # sort sellers' metadata according to their prices' lower bound increasingly
        buyer_data = sorted(buyer_data, reverse=True)  # sort buyers' metadata according to their prices decreasingly
        seller_price_lower, seller_price_upper, seller_volume, seller_name = zip(*seller_data)
        buyer_price, buyer_volume, buyer_name = zip(*buyer_data)
        assert seller_price_lower == (1, 2, 3)
        assert seller_price_upper == (4, 5, 6)
        assert seller_volume == (7, 8, 9)
        assert seller_name == (1, 2, 3)
        assert buyer_price == (12, 11, 10)
        assert buyer_volume == (15, 14, 13)
        assert buyer_name == (3, 2, 1)
        print("Test Iteration %d Passed" % (itr + 1))

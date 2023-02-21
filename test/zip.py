if __name__ == "__main__":
    # Data rearrange
    # _action data layout:
    #   [seller0_price_lower, seller0_price_upper, seller0_volume, ..., buyer0_price, buyer0_volume, ...]
    seller_price_lower = [1, 2, 3]
    seller_price_upper = [4, 5, 6]
    seller_volume = [7, 8, 9]
    buyer_price = [10, 11, 12]
    buyer_volume = [13, 14, 15]
    _seller_data = [seller_price_lower, seller_price_upper, seller_volume]
    _buyer_data = [buyer_price, buyer_volume]
    _seller_action = [val for tup in zip(*_seller_data) for val in tup]
    _buyer_action = [val for tup in zip(*_buyer_data) for val in tup]
    _action = _seller_action + _buyer_action
    assert _action == [1, 4, 7, 2, 5, 8, 3, 6, 9, 10, 13, 11, 14, 12, 15]

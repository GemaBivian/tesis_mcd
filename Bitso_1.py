# -*- coding: utf-8 -*-
"""
Created on Mon Mar  8 13:06:31 2021

@author: gema_
"""

import requests as req

def print_btc_value(price_mxn):
    response = req.get('https://api.bitso.com/v3/order_book/?book=btc_mxn')
    json_response = response.json()
    print(json_response)
    price_mxn = price_mxn
    value = 0 
    for i in range(50):
        price=float(json_response['payload']['asks'][i]['price'])
        amount = float(json_response['payload']['asks'][i]['amount'])
        value+=price*amount
        if value > price_mxn:
            price_btc = price
            value_btc = (price_mxn/price_btc)*1.04
            print("Precio del BTC {}".format(price_btc))
            print("El valor de {} MXN en BTC {}".format(price_mxn,value_btc))
            break
    return value_btc
value = print_btc_value(5000);
import jsonpickle
from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState
import numpy as np
import math

class Strategy:
    
    def __init__(self, symbol: str, limit: int):
        self.symbol = symbol
        self.limit = limit
        self.orders = []
    
    def act(self, state: TradingState) -> list[Order]:
        raise NotImplementedError
    
    def run(self, state: TradingState) -> list[Order]:
        self.act(state)
        return self.orders

    def buy(self, state: TradingState, price: int, quantity: int) -> None:
        self.orders.append(Order(self.symbol, price, quantity))

    def sell(self, state: TradingState, price: int, quantity: int) -> None:
        self.orders.append(Order(self.symbol, price, -quantity))

class MarketMakingStrategy(Strategy):

    def get_true_value(self, state: TradingState) -> int:
        raise NotImplementedError
    
    def act(self, state: TradingState) -> None:

        # Get True Value of Asset
        true_value = self.get_true_value(state)

        # Get Buy and Sell Orders
        order_depth = state.order_depths[self.symbol]
        buy_orders = sorted(order_depth.buy_orders.items(), reverse=True)
        sell_orders = sorted(order_depth.sell_orders.items())

        # Get buy and sell limits
        position = state.position.get(self.symbol, 0)
        to_buy = self.limit - position
        to_sell = self.limit + position

        max_buy_price = true_value - 1 if position > self.limit * 0.5 else true_value
        min_sell_price = true_value + 1 if position < self.limit * -0.5 else true_value

        for price, volume in sell_orders:
            if to_buy > 0 and price <= max_buy_price:
                quantity = min(to_buy, -volume)
                self.buy(state, price, quantity)
                to_buy -= quantity

        for price, volume in buy_orders:
            if to_sell > 0 and price >= min_sell_price:
                quantity = min(to_sell, volume)
                self.sell(state, price, quantity)
                to_sell -= quantity
        

class ResinStrategy(MarketMakingStrategy):

    def get_true_value(self, state: TradingState) -> int:
        return 10_000

class Trader:
    
    def __init__(self):
        limits = {
            "RAINFOREST_RESIN" : 50,
            "KELP" : 50
        }

        self.strategies = {symbol: constructor(symbol, limits[symbol]) for symbol, constructor in {
            "RAINFOREST_RESIN" : ResinStrategy,
        }.items()}

    def run(self, state : TradingState) -> tuple[dict[Symbol, list[Order]], int , str]:
        conversions = 0
        trading_data = {}
        if state.traderData != None and state.traderData != "":
            trading_data = jsonpickle.decode(state.traderData)
        
        orders = {}
        for symbol, strategy in self.strategies.items():
            if symbol in state.order_depths:
                orders[symbol] = strategy.run(state)

        trader_data = jsonpickle.encode(trading_data)
        return orders, conversions, trader_data

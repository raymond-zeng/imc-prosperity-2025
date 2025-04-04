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

    def buy(self, state: TradingState, price: int, quantity: int) -> None:
        self.orders.append(Order(self.symbol, price, -quantity))

class MarketMakingStrategy(Strategy):

    def get_true_value(state: TradingState) -> int:
        raise NotImplementedError
    
    def act(self, state: TradingState) -> None:
        true_value = self.get_true_value(state)

        order_depth = state.order_depths[self.symbol]
        buy_orders = sorted(order_depth.buy_orders.items(), reverse=True)
        sell_orders = sorted(order_depth.sell_orders.items())

        position = state.position.get(self.symbol, 0)
        to_buy = self.limit - position
        to_sell = self.limit + position

class ResinStrategy(MarketMakingStrategy):

    def get_true_value(state: TradingState) -> int:
        return 10_000

class Trader:
    
    def __init__(self):
        limits = {
            "RAINFOREST_RESIN" : 50,
            "KELP" : 50
        }

    def run(self, state : TradingState) -> tuple[dict[Symbol, list[Order]], int , str]:
        old_trading_data = {}
        if state.traderData != None and state.traderData != "":
            old_trading_data = jsonpickle.decode(state.traderData)
        orders = {}
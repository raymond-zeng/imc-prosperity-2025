import jsonpickle
from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState
import numpy as np
import math
from collections import deque

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
    def __init__(self, symbol: str, limit: int):
        super().__init__(symbol, limit)
        self.window = deque()
        self.window_size = 10
        self.limits_hit = 0
        self.limit_threshold = 5

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

        if abs(position) >= self.limit * 0.9:
            self.limits_hit += 1
        else:
            self.limits_hit = 0

        max_buy_price = true_value - 1 if position > self.limit * 0.5 else true_value
        min_sell_price = true_value + 1 if position < self.limit * -0.5 else true_value

        if self.limits_hit >= self.limit_threshold and to_buy > 0:
            self.limits_hit = 0
            quantity = to_buy // 2
            liquidation_price = true_value - 1

            # Try to match against the current book first
            for price, volume in sell_orders:
                if price <= liquidation_price and to_buy > 0:
                    fill_qty = min(to_buy, -volume)
                    self.buy(state, price, fill_qty)
                    to_buy -= fill_qty

            # If not filled, place a limit order at liquidation_price?
            if to_buy > 0:
                self.buy(state, max(sell_orders[0][0], liquidation_price), to_buy)

        for price, volume in sell_orders:
            if to_buy > 0 and price <= max_buy_price:   
                quantity = min(to_buy, -volume)
                self.buy(state, price, quantity)
                to_buy -= quantity

        if self.limits_hit >= self.limit_threshold and to_sell > 0:
            self.limits_hit = 0
            quantity = to_sell // 2
            liquidation_price = true_value + 1

            # Try to match against the current book first
            for price, volume in buy_orders:
                if price >= liquidation_price and to_sell > 0:
                    fill_qty = min(to_sell, volume)
                    self.sell(state, price, fill_qty)
                    to_sell -= fill_qty

            # If not filled, place a limit order at liquidation_price?
            if to_sell > 0:
                self.sell(state, min(buy_orders[0][0], liquidation_price), to_sell)

        for price, volume in buy_orders:
            if to_sell > 0 and price >= min_sell_price:
                quantity = min(to_sell, volume)
                self.sell(state, price, quantity)
                to_sell -= quantity
        

class ResinStrategy(MarketMakingStrategy):

    def get_true_value(self, state: TradingState) -> int:
        order_depth = state.order_depths[self.symbol]
        buy_orders = sorted(order_depth.buy_orders.items(), reverse=True)
        sell_orders = sorted(order_depth.sell_orders.items())

        mean_buy_price = np.mean([price for price, volume in buy_orders])
        mean_sell_price = np.mean([price for price, volume in sell_orders])

        # total_buy_volume = sum(volume for _, volume in buy_orders)
        # vw_buy = sum(price * volume for price, volume in buy_orders) / total_buy_volume if total_buy_volume else 0

        # total_sell_volume = sum(volume for _, volume in sell_orders)
        # vw_sell = sum(price * volume for price, volume in sell_orders) / total_sell_volume if total_sell_volume else 0

        mean_price = (mean_buy_price + mean_sell_price) / 2
        # mean_price = (vw_buy + vw_sell) / 2

        spread = sell_orders[0][0] - buy_orders[0][0]
        mean_price += 0.1 * spread  # bias toward ask side

        return int(mean_price)

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

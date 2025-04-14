import jsonpickle
from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState
import numpy as np
import math
from collections import deque

# f = open("test.txt", "w")
import json
from typing import Any
from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState

from statistics import NormalDist

def black_scholes_call(S, K, T, r, sigma):
    """
    Calculate Black-Scholes price for a call option using Python's statistics module
    
    Parameters:
    S: Current price of underlying asset (VOLCANIC_ROCK)
    K: Strike price of the voucher
    T: Time to expiration in years
    r: Risk-free interest rate
    sigma: Volatility of the underlying asset
    
    Returns:
    Call option price
    """
    d1 = (np.log(S/K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    # Use Python's built-in NormalDist for the cumulative distribution function
    norm_cdf = NormalDist(mu=0, sigma=1).cdf
    
    call_price = S * norm_cdf(d1) - K * np.exp(-r * T) * norm_cdf(d2)
    return call_price

def calculate_option_greeks(S, K, T, r, sigma):
    """Calculate option Greeks for risk management using Python's statistics module"""
    d1 = (np.log(S/K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    # Get normal distribution functions
    norm_cdf = NormalDist(mu=0, sigma=1).cdf
    
    # For PDF, we can use the standard normal PDF formula since it's not directly in statistics
    def norm_pdf(x):
        return (1.0 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * x**2)
    
    # Delta - sensitivity to underlying price changes
    delta = norm_cdf(d1)
    
    # Gamma - rate of change of delta
    gamma = norm_pdf(d1) / (S * sigma * np.sqrt(T))
    
    # Theta - sensitivity to time decay (daily)
    theta = -(S * sigma * norm_pdf(d1)) / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * norm_cdf(d2)
    theta = theta / 365  # Convert to daily theta
    
    # Vega - sensitivity to volatility changes
    vega = S * np.sqrt(T) * norm_pdf(d1) * 0.01  # For 1% change in volatility
    
    return {"delta": delta, "gamma": gamma, "theta": theta, "vega": vega}

class Logger:
    def __init__(self) -> None:
        self.logs = ""
        self.max_log_length = 3750

    def print(self, *objects: Any, sep: str = " ", end: str = "\n") -> None:
        self.logs += sep.join(map(str, objects)) + end

    def flush(self, state: TradingState, orders: dict[Symbol, list[Order]], conversions: int, trader_data: str) -> None:
        base_length = len(
            self.to_json(
                [
                    self.compress_state(state, ""),
                    self.compress_orders(orders),
                    conversions,
                    "",
                    "",
                ]
            )
        )

        # We truncate state.traderData, trader_data, and self.logs to the same max. length to fit the log limit
        max_item_length = (self.max_log_length - base_length) // 3

        print(
            self.to_json(
                [
                    self.compress_state(state, self.truncate(state.traderData, max_item_length)),
                    self.compress_orders(orders),
                    conversions,
                    self.truncate(trader_data, max_item_length),
                    self.truncate(self.logs, max_item_length),
                ]
            )
        )

        self.logs = ""

    def compress_state(self, state: TradingState, trader_data: str) -> list[Any]:
        return [
            state.timestamp,
            trader_data,
            self.compress_listings(state.listings),
            self.compress_order_depths(state.order_depths),
            self.compress_trades(state.own_trades),
            self.compress_trades(state.market_trades),
            state.position,
            self.compress_observations(state.observations),
        ]

    def compress_listings(self, listings: dict[Symbol, Listing]) -> list[list[Any]]:
        compressed = []
        for listing in listings.values():
            compressed.append([listing.symbol, listing.product, listing.denomination])

        return compressed

    def compress_order_depths(self, order_depths: dict[Symbol, OrderDepth]) -> dict[Symbol, list[Any]]:
        compressed = {}
        for symbol, order_depth in order_depths.items():
            compressed[symbol] = [order_depth.buy_orders, order_depth.sell_orders]

        return compressed

    def compress_trades(self, trades: dict[Symbol, list[Trade]]) -> list[list[Any]]:
        compressed = []
        for arr in trades.values():
            for trade in arr:
                compressed.append(
                    [
                        trade.symbol,
                        trade.price,
                        trade.quantity,
                        trade.buyer,
                        trade.seller,
                        trade.timestamp,
                    ]
                )

        return compressed

    def compress_observations(self, observations: Observation) -> list[Any]:
        conversion_observations = {}
        for product, observation in observations.conversionObservations.items():
            conversion_observations[product] = [
                observation.bidPrice,
                observation.askPrice,
                observation.transportFees,
                observation.exportTariff,
                observation.importTariff,
                observation.sugarPrice,
                observation.sunlightIndex,
            ]

        return [observations.plainValueObservations, conversion_observations]

    def compress_orders(self, orders: dict[Symbol, list[Order]]) -> list[list[Any]]:
        compressed = []
        for arr in orders.values():
            for order in arr:
                compressed.append([order.symbol, order.price, order.quantity])

        return compressed

    def to_json(self, value: Any) -> str:
        return json.dumps(value, cls=ProsperityEncoder, separators=(",", ":"))

    def truncate(self, value: str, max_length: int) -> str:
        if len(value) <= max_length:
            return value

        return value[: max_length - 3] + "..."

logger = Logger()

class MarketMakeStrategy():

    def __init__(self, symbol: str, limit: int, order_depth: OrderDepth, trader_data):
        self.symbol = symbol
        self.order_depth = order_depth
        self.limit = limit
        self.prevent_adverse = False # Flag to prevent taking orders that could lead to adverse price movements
        self.take_width = 1
        self.clear_width = 0
        self.disregard_edge = 1 # Orders inside this range from fair value are disregarded for joining or undercutting
        self.join_edge = 0 # Orders within this edge will be joined directly
        self.reversion_beta = 1  # Factor used to adjust fair value predictions based on past price deviations
        self.default_edge = 1 # Default offset from the fair value if no optimal edge is available
        self.adverse_volume = 0 # Volume threshold to define adverse market conditions
        self.soft_position_limit = 30 # Soft limit for managing position size before adjustments
        self.manage_position = False
        self.fair_value = 0
        self.trader_data = trader_data

    def fair_price(self, order_depth: OrderDepth, trader_data) -> float:
        raise NotImplementedError

    def market_make(self, orders: list[Order], bid: int, ask: int, position: int, buy_order_volume: int, 
                sell_order_volume: int) -> tuple[int, int]:
        # Scale order sizes based on position
        position_ratio = abs(position) / self.limit
        scaling_factor = max(0.2, 1.0 - position_ratio)
        
        # Calculate available capacity to buy
        max_buy_qty = self.limit - (position + buy_order_volume)
        # Scale down order size as we approach limit
        buy_quantity = min(max_buy_qty, max(1, int(self.limit * 0.25 * scaling_factor)))
        if buy_quantity > 0:
            orders.append(Order(self.symbol, round(bid), buy_quantity))
            buy_order_volume += buy_quantity
        
        # Calculate available capacity to sell
        max_sell_qty = self.limit + (position - sell_order_volume)
        # Scale down order size as we approach limit
        sell_quantity = min(max_sell_qty, max(1, int(self.limit * 0.25 * scaling_factor)))
        if sell_quantity > 0:
            orders.append(Order(self.symbol, round(ask), -sell_quantity))  
            sell_order_volume += sell_quantity
            
        return buy_order_volume, sell_order_volume


    def take_orders(self, position: int, buy_order_volume: int, 
                    sell_order_volume: int) -> tuple[list[Order], int, int]:
        orders = []
        #placing buy orders
        # Check if any sell orders exist: these are offers to sell at various prices.
        if len(self.order_depth.sell_orders) != 0:
            # Find the best ask (lowest sell price)
            best_ask = min(self.order_depth.sell_orders.keys())
            # Multiply by -1 because the sell order volumes are stored as negative values
            best_ask_amount = -1 * self.order_depth.sell_orders[best_ask]

            # Only consider the order if it does not trigger an adverse condition
            if not self.prevent_adverse or abs(best_ask_amount) <= self.adverse_volume:
                # If the best ask is below the threshold (fair value minus take_width), it's an attractive buy.
                if best_ask <= self.fair_value - self.take_width:
                    # Calculate how many units can be bought without exceeding position limits.
                    quantity = min(best_ask_amount, self.limit - position)
                    if quantity > 0:
                        # Place a buy order at the best ask price.
                        orders.append(Order(self.symbol, best_ask, quantity))
                        buy_order_volume += quantity
                        # Update the order depth to reflect the filled volume.
                        self.order_depth.sell_orders[best_ask] += quantity
                        if self.order_depth.sell_orders[best_ask] == 0:
                            del self.order_depth.sell_orders[best_ask]
        
        # Repeat the process for buy orders (bids) from the order book.
        if len(self.order_depth.buy_orders) != 0:
            # Find the best bid (highest buy price)
            best_bid = max(self.order_depth.buy_orders.keys())
            best_bid_amount = self.order_depth.buy_orders[best_bid]

            if not self.prevent_adverse or abs(best_bid_amount) <= self.adverse_volume:
                # If the best bid is above the threshold (fair value plus take_width), it's an attractive sell.
                if best_bid >= self.fair_value + self.take_width:
                    # Calculate how many units can be sold without exceeding position limits.
                    quantity = min(best_bid_amount, self.limit + position)
                    if quantity > 0:
                        # Place a sell order at the best bid price.
                        orders.append(Order(self.symbol, best_bid, -1 * quantity))
                        sell_order_volume += quantity
                        self.order_depth.buy_orders[best_bid] -= quantity
                        if self.order_depth.buy_orders[best_bid] == 0:
                            del self.order_depth.buy_orders[best_bid]
        
        return orders, buy_order_volume, sell_order_volume
    
    def clear_orders(self, position: int, buy_order_volume: int, 
                    sell_order_volume: int) -> tuple[list[Order], int, int]:
        orders = []
        # calculate new position after taking
        new_position = position + buy_order_volume - sell_order_volume

        # determine fair bids and asks
        fair_bid = round(self.fair_value - self.clear_width)
        fair_ask = round(self.fair_value + self.clear_width)

        # determine buy and sell quantities
        buy_quantity = self.limit - (position + buy_order_volume)
        sell_quantity = self.limit + (position - sell_order_volume)

        # if new_position is positive, sell some positions
        if new_position > 0:
            # calculate quantity that is at or above fair ask value
            clear_quantity = sum(
                volume
                for price, volume in self.order_depth.buy_orders.items()
                if price >= fair_ask
            )
            clear_quantity = min(clear_quantity, new_position)
            # can only sell as many as possible
            sent_quantity = min(sell_quantity, clear_quantity)
            if sent_quantity > 0:
                # Send a sell order to reduce the position.
                orders.append(Order(self.symbol, fair_ask, -abs(sent_quantity)))
                sell_order_volume += abs(sent_quantity)
        
        # otherwise buy some positions
        elif new_position < 0:
            clear_quantity = sum(
                abs(volume)
                for price, volume in self.order_depth.sell_orders.items()
                if price <= fair_bid
            )
            clear_quantity = min(clear_quantity, abs(new_position))
            sent_quantity = min(buy_quantity, clear_quantity)
            if sent_quantity > 0:
                # Send a buy order to reduce the short position.
                orders.append(Order(self.symbol, fair_bid, abs(sent_quantity)))
                buy_order_volume += abs(sent_quantity)
        
        return orders, buy_order_volume, sell_order_volume
    
    def make_orders(self, position, buy_order_volume: int, sell_order_volume: int) -> tuple[list[Order], int, int]:
        orders = []

        # Calculate remaining capacity based on position and orders already placed
        remaining_buy_capacity = self.limit - (position + buy_order_volume)
        remaining_sell_capacity = self.limit + (position - sell_order_volume)
        
        # Position management - reduce order sizes as limits are approached
        position_ratio = abs(position) / self.limit
        scaling_factor = max(0.2, 1.0 - position_ratio)
        
        # Identify asks and bids beyond disregard threshold
        asks_above_fair = [
            price
            for price in self.order_depth.sell_orders.keys()
            if price > self.fair_value + self.disregard_edge
        ]
        bids_below_fair = [
            price
            for price in self.order_depth.buy_orders.keys()
            if price < self.fair_value - self.disregard_edge
        ]

        best_ask_above_fair = min(asks_above_fair) if len(asks_above_fair) > 0 else None
        best_bid_below_fair = max(bids_below_fair) if len(bids_below_fair) > 0 else None

        # Initialize ask and bid with default edge
        ask = round(self.fair_value + self.default_edge)
        if best_ask_above_fair != None:
            if abs(best_ask_above_fair - self.fair_value) <= self.join_edge:
                ask = best_ask_above_fair  # Join the level
            else:
                ask = best_ask_above_fair - 1  # Undercut by one unit

        bid = round(self.fair_value - self.default_edge)
        if best_bid_below_fair != None:
            if abs(self.fair_value - best_bid_below_fair) <= self.join_edge:
                bid = best_bid_below_fair  # Join the level
            else:
                bid = best_bid_below_fair + 1  # Raise bid slightly

        # Position-based price adjustments
        if position > self.soft_position_limit * 0.7:
            ask = max(bid + 1, ask - 1)  # Tighten ask to encourage selling
        elif position < -1 * self.soft_position_limit * 0.7:
            bid = min(ask - 1, bid + 1)  # Tighten bid to encourage buying

        # Size orders based on position and scaling factor
        buy_qty = min(remaining_buy_capacity, max(1, int(self.limit * 0.25 * scaling_factor)))
        if buy_qty > 0 and bid < ask:  # Ensure no crossing
            orders.append(Order(self.symbol, round(bid), buy_qty))
            buy_order_volume += buy_qty
        
        sell_qty = min(remaining_sell_capacity, max(1, int(self.limit * 0.25 * scaling_factor)))
        if sell_qty > 0 and ask > bid:  # Ensure no crossing
            orders.append(Order(self.symbol, round(ask), -sell_qty))
            sell_order_volume += sell_qty

        return orders, buy_order_volume, sell_order_volume

    def act(self, state: TradingState):
        position = state.position.get(self.symbol, 0)
        buy_order_volume = 0
        sell_order_volume = 0
        
        # Use all three order types: taking, clearing, and making
        take, buy_order_volume, sell_order_volume = self.take_orders(position, buy_order_volume, sell_order_volume)
        clear, buy_order_volume, sell_order_volume = self.clear_orders(position, buy_order_volume, sell_order_volume)
        make, _, _ = self.make_orders(position, buy_order_volume, sell_order_volume)
    
        # Combine all order types
        return {self.symbol: take + clear + make}
        
# Continuing the strategies from Round 1
class ResinStrategy(MarketMakeStrategy):

    def __init__(self, symbol: str, limit: int, order_depth: OrderDepth, trader_data):
        super().__init__(symbol, limit, order_depth, trader_data)
        self.join_edge = 2
        self.default_edge = 4
        self.soft_position_limit = 33
        self.manage_position = True
        self.fair_value = 10_000
    
class KelpStrategy(MarketMakeStrategy):

    def __init__(self, symbol: str, limit: int, order_depth: OrderDepth, trader_data):
        super().__init__(symbol, limit, order_depth, trader_data)
        self.prevent_adverse = True
        self.adverse_volume = 24
        self.reversion_beta = -0.3
        self.join_edge = 0
        self.default_edge = 1
        self.fair_value = 0

    def fair_price(self) -> float:
        fair = None
        if len(self.order_depth.sell_orders) != 0 and len(self.order_depth.buy_orders) != 0:
            best_ask = min(self.order_depth.sell_orders.keys())
            best_bid = max(self.order_depth.buy_orders.keys())

            # Filter out orders that are below the adverse volume threshold to avoid extreme moves.
            filtered_ask = [
                price
                for price in self.order_depth.sell_orders.keys()
                if abs(self.order_depth.sell_orders[price])
                >= self.adverse_volume
            ]
            filtered_bid = [
                price
                for price in self.order_depth.buy_orders.keys()
                if abs(self.order_depth.buy_orders[price])
                >= self.adverse_volume
            ]
            mm_ask = min(filtered_ask) if len(filtered_ask) > 0 else None
            mm_bid = max(filtered_bid) if len(filtered_bid) > 0 else None

            # If filtered values are missing, use the best bid/ask or fall back to the last price.
            if mm_ask == None or mm_bid == None:
                if self.trader_data.get("KELP_last_price", None) == None:
                    mmmid_price = (best_ask + best_bid) / 2
                else:
                    mmmid_price = self.trader_data["KELP_last_price"]
            else:
                mmmid_price = (mm_ask + mm_bid) / 2

            # Apply a reversion adjustment based on how far the current mid-price is from the last recorded price.
            if self.trader_data.get("KELP_last_price", None) != None:
                last_price = self.trader_data["KELP_last_price"]
                last_returns = (mmmid_price - last_price) / last_price
                pred_returns = (
                    last_returns * self.reversion_beta
                )
                fair = mmmid_price + (mmmid_price * pred_returns)
            else:
                fair = mmmid_price

            # Update the trader object with the latest mid-price.
            self.trader_data["KELP_last_price"] = mmmid_price
        return fair

    def act(self, state: TradingState):
        self.fair_value = self.fair_price()
        return super().act(state)
    
class SquidInkStrategy(MarketMakeStrategy):

    def __init__(self, symbol: str, limit: int, order_depth: OrderDepth, trader_data):
        super().__init__(symbol, limit, order_depth, trader_data)
        self.take_width = 2
        self.clear_width = 1
        self.disregard_edge = 0
        self.join_edge = 2
        self.default_edge = 3
        self.mean_reversion_alpha = 0.17625
        self.threshold = 2.0

    def fair_price(self) -> float:
        fair = None
        if len(self.order_depth.sell_orders) != 0 and len(self.order_depth.buy_orders) != 0:
            best_ask = min(self.order_depth.sell_orders.keys())
            best_bid = max(self.order_depth.buy_orders.keys())
            mid_price = (best_ask + best_bid) / 2

            if "SQUID_INK_price_history" not in self.trader_data:
                self.trader_data["SQUID_INK_price_history"] = deque(maxlen=500)
            
            self.trader_data["SQUID_INK_price_history"].append(mid_price)  # Store the latest mid-price
            prices = self.trader_data["SQUID_INK_price_history"]

            rolling_mean = np.mean(prices)  # Calculate the rolling mean of the last 500 prices
            std_dev = np.std(prices)  # Calculate the standard deviation of the last 500 prices

            signal = None
            if mid_price < rolling_mean - self.threshold * std_dev:
                signal = "BUY"
            elif mid_price > rolling_mean + self.threshold * std_dev:
                signal = "SELL"
            self.trader_data["SQUID_INK_signal"] = signal

            cooldown = self.trader_data.get("SQUID_INK_cooldown", 0)
            if signal and cooldown == 0:
                self.trader_data["SQUID_INK_signal_triggered"] = True
                self.trader_data["SQUID_INK_signal_cooldown"] = 10
            else:
                self.trader_data["SQUID_INK_signal_triggered"] = False
                self.trader_data["SQUID_INK_signal_cooldown"] = max(0, cooldown - 1)
            
            fair = (1 - self.mean_reversion_alpha) * mid_price + self.mean_reversion_alpha * rolling_mean

        return fair
    
    def act(self, state: TradingState):
        self.fair_value = self.fair_price()
        position = state.position.get(self.symbol, 0)
        orders = []
        signal = self.trader_data.get("SQUID_INK_signal", None)
        triggered = self.trader_data.get("SQUID_INK_signal_triggered", False)
        buy_order_volume = 0
        sell_order_volume = 0

        if triggered:
            if signal == "BUY":
                best_ask = min(self.order_depth.sell_orders.keys())
                qty = min(self.limit - position, -self.order_depth.sell_orders[best_ask])
                if qty > 0:
                    orders.append(Order(self.symbol, best_ask, qty))
                    buy_order_volume += qty
            elif signal == "SELL":
                best_bid = max(self.order_depth.buy_orders.keys())
                qty = min(self.limit + position, self.order_depth.buy_orders[best_bid])
                if qty > 0:
                    orders.append(Order(self.symbol, best_bid, -qty))
                    sell_order_volume += qty
        
        clear, buy_order_volume, sell_order_volume = self.clear_orders(position, buy_order_volume, sell_order_volume)
        make, _, _ = self.make_orders(position, buy_order_volume, sell_order_volume)
        return {self.symbol: orders + clear + make}
    

class PicnicBasketStrategy(MarketMakeStrategy):
    def __init__(self, symbol: str, limit: int, order_depth: OrderDepth, trader_data):
        # components = ["CROISSANTS", "JAMS", "DJEMBES"]
        # quantities = [6, 3, 1]
        super().__init__(symbol, limit, order_depth, trader_data)

        # Parameters optimized for component-based mean reversion
        self.take_width = 1
        self.clear_width = 0
        self.disregard_edge = 1
        self.join_edge = 1
        self.default_edge = 2
        self.arb_threshold = 1  # Tighter threshold for more aggressive mean reversion
        self.component_reversion_strength = 0.4  # How strongly to revert to component value
        self.prevent_adverse = True
        self.adverse_volume = 15  # Avoid thin liquidity
        self.soft_position_limit = int(self.limit * 0.85)
        
    def fair_price(self, state: TradingState) -> float:
        if len(self.order_depth.sell_orders) != 0 and len(self.order_depth.buy_orders) != 0:
            best_ask = min(self.order_depth.sell_orders.keys())
            best_bid = max(self.order_depth.buy_orders.keys())
            basket_mid_price = (best_ask + best_bid) / 2
            
            # Calculate theoretical value from components
            components_value = 0
            components_available = True
            
            for i, component in enumerate(self.components):
                if component in state.order_depths:
                    component_order_depth = state.order_depths[component]
                    if (len(component_order_depth.sell_orders) != 0 and 
                        len(component_order_depth.buy_orders) != 0):
                        comp_best_ask = min(component_order_depth.sell_orders.keys())
                        comp_best_bid = max(component_order_depth.buy_orders.keys())
                        comp_mid_price = (comp_best_ask + comp_best_bid) / 2
                        components_value += comp_mid_price * self.quantities[i]
                    else:
                        components_available = False
                else:
                    components_available = False
            
            # Track and update price history
            if f"{self.symbol}_price_history" not in self.trader_data:
                self.trader_data[f"{self.symbol}_price_history"] = deque(maxlen=100)
            self.trader_data[f"{self.symbol}_price_history"].append(basket_mid_price)
            
            # Calculate price deviation from theoretical component value
            if components_available:
                # Store raw deviation
                raw_deviation = basket_mid_price - components_value
                self.trader_data[f"{self.symbol}_raw_deviation"] = raw_deviation
                
                # Calculate percentage deviation (relative to component value)
                pct_deviation = (basket_mid_price - components_value) / components_value if components_value != 0 else 0
                self.trader_data[f"{self.symbol}_pct_deviation"] = pct_deviation
                
                # Keep history of percentage deviations for statistical analysis
                if f"{self.symbol}_pct_deviation_history" not in self.trader_data:
                    self.trader_data[f"{self.symbol}_pct_deviation_history"] = deque(maxlen=50)
                self.trader_data[f"{self.symbol}_pct_deviation_history"].append(pct_deviation)
                
                deviation_history = list(self.trader_data[f"{self.symbol}_pct_deviation_history"])
                if len(deviation_history) > 5:  # Need enough data for meaningful statistics
                    mean_deviation = np.mean(deviation_history)
                    std_deviation = np.std(deviation_history) 
                    
                    # Calculate z-score (how many standard deviations from mean)
                    z_score = (pct_deviation - mean_deviation) / std_deviation
                    self.trader_data[f"{self.symbol}_z_score"] = z_score
                    
                    # Use z-score for more consistent threshold adjustments
                    if abs(z_score) > 2.0:  # Very significant deviation (>2 standard deviations)
                        # For large statistical deviations, increase reversion strength
                        self.component_reversion_strength = 0.78
                        self.arb_threshold = 0.001  # Almost any deviation is worth trading
                        self.take_width = 0  # Extremely aggressive, take at fair value
                    elif abs(z_score) > 1.0:  # Moderate deviation (1-2 standard deviations)
                        self.component_reversion_strength = 0.75
                        self.arb_threshold = 0.002  # Small percentage threshold
                        self.take_width = 1
                    else:  # Small deviation (<1 standard deviation)
                        # For small deviations, normal settings
                        self.component_reversion_strength = 0.6
                        self.arb_threshold = 0.004  # Higher percentage threshold
                        self.take_width = 1
                        
                    # Use percentage deviation for trading decisions
                    self.trader_data[f"{self.symbol}_deviation"] = pct_deviation
                    
                    # Adjust spread based on z-score - wider when close to mean
                    self.default_edge = max(1, min(3, int(4 - abs(z_score))))
                    
                    # f.write(f"Z-score: {z_score}, Pct Dev: {pct_deviation}, Threshold: {self.arb_threshold}\n")
                    
                else:
                    # Not enough history yet, use basic percentage-based approach
                    if abs(pct_deviation) > 0.01:  # 1% deviation
                        self.component_reversion_strength = 0.5
                        self.arb_threshold = 0.005
                        self.take_width = 0
                    elif abs(pct_deviation) > 0.005:  # 0.5% deviation
                        self.component_reversion_strength = 0.4
                        self.arb_threshold = 0.002
                        self.take_width = 1
                    else:
                        self.component_reversion_strength = 0.3
                        self.arb_threshold = 0.001
                        self.take_width = 1
                        
                    # Use percentage deviation for trading decisions
                    self.trader_data[f"{self.symbol}_deviation"] = pct_deviation
                    
                    # Adjust spread based on percentage deviation
                    self.default_edge = max(1, min(3, int(4 - abs(pct_deviation) * 200)))
                    
                    # f.write(f"Pct Dev: {pct_deviation}, Threshold: {self.arb_threshold}\n")
                
                # Make fair value calculations based on appropriate measurements
                # Always use raw deviation for price adjustment to get correct absolute values
                fair = basket_mid_price - (raw_deviation * self.component_reversion_strength)
                
                # Store other data for market making and position management
                self.trader_data[f"{self.symbol}_components_value"] = components_value
                self.trader_data[f"{self.symbol}_market_value"] = basket_mid_price
                
                # Store component prices for analysis
                for i, component in enumerate(self.components):
                    if component in state.order_depths:
                        component_order_depth = state.order_depths[component]
                        if (len(component_order_depth.sell_orders) != 0 and 
                            len(component_order_depth.buy_orders) != 0):
                            comp_best_ask = min(component_order_depth.sell_orders.keys())
                            comp_best_bid = max(component_order_depth.buy_orders.keys())
                            self.trader_data[f"{self.symbol}_{component}_bid"] = comp_best_bid
                            self.trader_data[f"{self.symbol}_{component}_ask"] = comp_best_ask
            else:
                # If components aren't available, use simple mid price
                fair = basket_mid_price
                
            return fair
        return None
    
    def hedge(self, state: TradingState, order: Order) -> dict[str: Order]:
        quantity = order.quantity
        orders = {'CROISSANTS': None, 'JAMS': None, 'DJEMBES': None}
        for i, component in enumerate(self.components):
            if component in state.order_depths:
                component_order_depth = state.order_depths[component]
                if quantity > 0:
                    best_ask = min(component_order_depth.sell_orders.keys())
                    hedge_quantity = min(quantity * self.quantities[i], 
                                     component_order_depth.sell_orders[best_ask])
                    if hedge_quantity > 0:
                        orders[component] = Order(component, best_ask, hedge_quantity)
                if quantity < 0:
                    best_bid = max(component_order_depth.buy_orders.keys())
                    hedge_quantity = min(-quantity * self.quantities[i], 
                                     component_order_depth.buy_orders[best_bid])
                    if hedge_quantity > 0:
                        orders[component] = Order(component, best_bid, -hedge_quantity)
        return orders
                    
    def act(self, state: TradingState):
        self.fair_value = self.fair_price(state)
        if self.fair_value is None:
            # Fall back to simple market making if pricing data is missing
            position = state.position.get(self.symbol, 0)
            take, buy_order_volume, sell_order_volume = self.take_orders(position, 0, 0)
            clear, buy_order_volume, sell_order_volume = self.clear_orders(position, buy_order_volume, sell_order_volume)
            make, _, _ = self.make_orders(position, buy_order_volume, sell_order_volume)
            return take + clear + make
            
        position = state.position.get(self.symbol, 0)
        orders = []
        component_orders = {COMPONENT : [] for COMPONENT in self.components}
        buy_order_volume = 0
        sell_order_volume = 0
        
        # Get correct deviation metric for trading decisions based on statistical models
        deviation = self.trader_data.get(f"{self.symbol}_deviation", 0)
        components_value = self.trader_data.get(f"{self.symbol}_components_value", 0)
        market_value = self.trader_data.get(f"{self.symbol}_market_value", 0)
        z_score = self.trader_data.get(f"{self.symbol}_z_score", 0)
        
        # Position-aware sizing - adjust position limit dynamically
        position_ratio = abs(position) / self.limit
        position_factor = max(0.4, 1.0 - position_ratio)
        
        # Adjust strategy based on position direction
        direction_factor = 1.0
        if (deviation > 0 and position < 0) or (deviation < 0 and position > 0):
            # Already have position in the right direction - be less aggressive
            direction_factor = 0.7
        
        # If basket price deviates from component value, trade to profit from reversion
        if abs(deviation) > self.arb_threshold:
            if deviation > 0:  # Basket overpriced vs components - SELL
                if len(self.order_depth.buy_orders) > 0:
                    best_bid = max(self.order_depth.buy_orders.keys())
                    best_bid_volume = self.order_depth.buy_orders[best_bid]
                    available_capacity = self.limit + position
                    
                    # More adaptive sizing based on deviation magnitude, position, and direction
                    base_sizing = min(0.9, 0.5 + (abs(deviation) * 50))  # Scale percentage deviation
                    sizing_factor = base_sizing * position_factor * direction_factor
                    
                    quantity = min(best_bid_volume, int(available_capacity * sizing_factor))
                    
                    if quantity > 0:
                        orders.append(Order(self.symbol, best_bid, -quantity))
                        hedge_orders = self.hedge(state, orders[-1])
                        for i, component in enumerate(self.components):
                            if (hedge_orders[component] != None): component_orders[component].append(hedge_orders[component])

                        sell_order_volume += quantity
                        
                        # Sweep additional levels with more intelligent logic
                        if available_capacity - quantity > 0:
                            # Dynamic level determination based on z-score
                            sweep_levels = min(3, max(1, round(abs(z_score))))
                            remaining_capacity = available_capacity - quantity
                            bid_prices = sorted(self.order_depth.buy_orders.keys(), reverse=True)
                            
                            # Calculate sweep threshold based on percentage deviation
                            if abs(deviation) > 0.01:  # >1% deviation
                                sweep_threshold = components_value * (1 + self.arb_threshold)
                            else:
                                # For smaller deviations, be more conservative
                                sweep_threshold = components_value * (1 + self.arb_threshold * 1.5)
                            
                            levels_swept = 0
                            for bid_price in bid_prices[1:]:  # Skip best bid we already took
                                if levels_swept >= sweep_levels:
                                    break
                                    
                                if bid_price >= sweep_threshold:
                                    bid_volume = self.order_depth.buy_orders[bid_price]
                                    # Reduce size for each level deeper in the book
                                    level_factor = 0.8 ** (levels_swept + 1)
                                    sweep_quantity = min(bid_volume, 
                                                        int(remaining_capacity * sizing_factor * level_factor))
                                    
                                    if sweep_quantity > 0:
                                        orders.append(Order(self.symbol, bid_price, -sweep_quantity))
                                        hedge_orders = self.hedge(state, orders[-1])
                                        for i, component in enumerate(self.components):
                                            if (hedge_orders[component] != None): component_orders[component].append(hedge_orders[component])
                                        sell_order_volume += sweep_quantity
                                        remaining_capacity -= sweep_quantity
                                        levels_swept += 1
            
            else:  # Basket underpriced vs components - BUY
                if len(self.order_depth.sell_orders) > 0:
                    best_ask = min(self.order_depth.sell_orders.keys())
                    best_ask_volume = -self.order_depth.sell_orders[best_ask]
                    available_capacity = self.limit - position
                    
                    # More adaptive sizing based on deviation magnitude, position, and direction
                    base_sizing = min(0.9, 0.5 + (abs(deviation) * 50))  # Scale percentage deviation
                    sizing_factor = base_sizing * position_factor * direction_factor
                    
                    quantity = min(best_ask_volume, int(available_capacity * sizing_factor))
                    
                    if quantity > 0:
                        orders.append(Order(self.symbol, best_ask, quantity))
                        hedge_orders = self.hedge(state, orders[-1])
                        for i, component in enumerate(self.components):
                            if (hedge_orders[component] != None): component_orders[component].append(hedge_orders[component])
                        buy_order_volume += quantity
                        
                        # Sweep additional levels with more intelligent logic
                        if available_capacity - quantity > 0:
                            # Dynamic level determination based on z-score
                            sweep_levels = min(3, max(1, round(abs(z_score))))
                            remaining_capacity = available_capacity - quantity
                            ask_prices = sorted(self.order_depth.sell_orders.keys())
                            
                            # Calculate sweep threshold based on percentage deviation
                            if abs(deviation) > 0.01:  # >1% deviation
                                sweep_threshold = components_value * (1 - self.arb_threshold)
                            else:
                                # For smaller deviations, be more conservative
                                sweep_threshold = components_value * (1 - self.arb_threshold * 1.5)
                            
                            levels_swept = 0
                            for ask_price in ask_prices[1:]:  # Skip best ask we already took
                                if levels_swept >= sweep_levels:
                                    break
                                    
                                if ask_price <= sweep_threshold:
                                    ask_volume = -self.order_depth.sell_orders[ask_price]
                                    # Reduce size for each level deeper in the book
                                    level_factor = 0.8 ** (levels_swept + 1)
                                    sweep_quantity = min(ask_volume, 
                                                        int(remaining_capacity * sizing_factor * level_factor))
                                    
                                    if sweep_quantity > 0:
                                        orders.append(Order(self.symbol, ask_price, sweep_quantity))
                                        hedge_orders = self.hedge(state, orders[-1])
                                        for i, component in enumerate(self.components):
                                            if (hedge_orders[component] != None): component_orders[component].append(hedge_orders[component])
                                        buy_order_volume += sweep_quantity
                                        remaining_capacity -= sweep_quantity
                                        levels_swept += 1
        
        # Position management through market making
        if abs(position) > self.soft_position_limit * 0.7:
            if position > 0:
                # With large long position, tighten ask to sell more
                self.default_edge = max(1, self.default_edge - 1)
            else:
                # With large short position, tighten bid to buy more
                self.default_edge = max(1, self.default_edge - 1)
        
        # Standard market making layers with adaptive parameters
        take, buy_order_volume, sell_order_volume = self.take_orders(position, buy_order_volume, sell_order_volume)
        for order in take:
            hedge_orders = self.hedge(state, order)
            for i, component in enumerate(self.components):
                if (hedge_orders[component] != None): component_orders[component].append(hedge_orders[component])
        clear, buy_order_volume, sell_order_volume = self.clear_orders(position, buy_order_volume, sell_order_volume)
        for order in clear:
            hedge_orders = self.hedge(state, order)
            for i, component in enumerate(self.components):
                if (hedge_orders[component] != None): component_orders[component].append(hedge_orders[component])
        make, _, _ = self.make_orders(position, buy_order_volume, sell_order_volume)
        for order in make:
            hedge_orders = self.hedge(state, order)
            for i, component in enumerate(self.components):
                if (hedge_orders[component] != None): component_orders[component].append(hedge_orders[component])
        component_orders[self.symbol] = orders + clear + make
        return component_orders

class PicnicBasket1Strategy(PicnicBasketStrategy):
    def __init__(self, symbol, limit, order_depth, trader_data):
        self.components = ["CROISSANTS", "JAMS", "DJEMBES"]
        self.quantities = [6, 3, 1]
        super().__init__(symbol, limit, order_depth, trader_data)

        # if f"{self.symbol}_price_history" not in self.trader_data:
        #     # Preload with reasonable historical values
        #     # Historical average price and typical deviations from backtesting
        #     self.trader_data[f"{self.symbol}_price_history"] = deque([58690, 58695, 58700, 58685, 58680], maxlen=100)
            
        #     # Preload deviation history with typical values
        #     self.trader_data[f"{self.symbol}_pct_deviation_history"] = deque(
        #         [0.002, -0.001, 0.0015, -0.0005, 0.001, 0.002, -0.0015, 0.0005, -0.001, 0.0025], 
        #         maxlen=50
        #     )
            
        #     # Initialize other statistical values
        #     self.trader_data[f"{self.symbol}_components_value"] = 58687
        #     self.trader_data[f"{self.symbol}_market_value"] = 58690
        #     self.trader_data[f"{self.symbol}_z_score"] = 0.1
        #     self.trader_data[f"{self.symbol}_deviation"] = 0.005

        # Parameters optimized for component-based mean reversion
        self.take_width = 1
        self.clear_width = 0
        self.disregard_edge = 1
        self.join_edge = 1
        self.default_edge = 2
        self.arb_threshold = 1  # Tighter threshold for more aggressive mean reversion
        self.component_reversion_strength = 0.  # How strongly to revert to component value
        self.prevent_adverse = True
        self.adverse_volume = 15  # Avoid thin liquidity
        self.soft_position_limit = int(self.limit * 6)

class PicnicBasket2Strategy(PicnicBasketStrategy):
    components = []
    def __init__(self, symbol, limit, order_depth, trader_data):
        self.components = ["CROISSANTS", "JAMS"]
        self.quantities = [4, 2]
        super().__init__(symbol, limit, order_depth, trader_data)
        self.take_width = 1
        self.clear_width = 0
        self.disregard_edge = 1
        self.join_edge = 1
        self.default_edge = 2
        self.arb_threshold = 1  # Tighter threshold for more aggressive mean reversion
        self.component_reversion_strength = 0.4  # How strongly to revert to component value
        self.prevent_adverse = True
        self.adverse_volume = 15  # Avoid thin liquidity
        self.soft_position_limit = int(self.limit * 0.85)

# Base class for all Volcanic Rock Voucher strategies
class BaseVolcanicRockVoucherStrategy(MarketMakeStrategy):
    def __init__(self, symbol: str, limit: int, order_depth: OrderDepth, trader_data):
        super().__init__(symbol, limit, order_depth, trader_data)
        
        # Extract strike price from symbol
        self.strike = int(symbol.split('_')[-1])
        
        # Black-Scholes parameters - these can be overridden in child classes
        self.volatility = trader_data.get(f"{symbol}_volatility", 0.3)  # Initial estimate
        self.risk_free_rate = 0.01
        self.volatility_adjustment = 1.0  # Multiplier for volatility (adjust in child classes)
        self.pricing_thresholds = (0.95, 1.05)  # (buy_threshold, sell_threshold)
        
        # Get current round from trader_data to track days until expiry
        self.current_round = trader_data.get("current_round", 1)
        self.remaining_days = 8 - self.current_round  # 7 days at round 1, decreases each round
        
        # Parameters for market making
        self.take_width = 2
        self.clear_width = 1
        self.disregard_edge = 1
        self.join_edge = 2
        self.default_edge = 3
        self.prevent_adverse = True
        self.adverse_volume = 10
        self.soft_position_limit = int(limit * 0.8)
        
        # Delta hedging parameters
        self.delta_hedge_threshold = 5  # Min difference to trigger hedge
        self.delta_hedge_factor = 1.0   # Multiplier for hedge quantity
        
        # Time decay parameters
        self.time_weight_1day = 0.2     # Weight for theoretical value when 1 day left
        self.time_weight_3day = 0.7     # Weight for theoretical value when 2-3 days left
        
        # Position sizing parameters
        self.mm_size_factor = 0.2       # Base size for market making orders
        
        # Track underlying price and option values
        self.fair_value = 0
        self.underlying_price = trader_data.get("VOLCANIC_ROCK_mid_price", 10000)
        self.option_value = 0
        self.greeks = {}
    
    def update_volatility(self, state):
        """Update volatility estimate based on recent price movements"""
        symbol = "VOLCANIC_ROCK"
        if f"{symbol}_price_history" not in self.trader_data:
            self.trader_data[f"{symbol}_price_history"] = deque(maxlen=50)
            
        # If we have the underlying in the order book, update its price
        if symbol in state.order_depths:
            rock_depth = state.order_depths[symbol]
            if len(rock_depth.buy_orders) > 0 and len(rock_depth.sell_orders) > 0:
                best_bid = max(rock_depth.buy_orders.keys())
                best_ask = min(rock_depth.sell_orders.keys())
                mid_price = int((best_bid + best_ask) / 2)
                
                self.trader_data[f"{symbol}_price_history"].append(mid_price)
                self.underlying_price = mid_price
                self.trader_data["VOLCANIC_ROCK_mid_price"] = mid_price
        
        # Calculate historical volatility if we have enough data
        if len(self.trader_data[f"{symbol}_price_history"]) > 5:
            prices = np.array(list(self.trader_data[f"{symbol}_price_history"]))
            returns = np.diff(np.log(prices))
            historical_vol = np.std(returns) * np.sqrt(252)  # Annualized volatility
            
            # Apply strike-specific volatility adjustment
            adjusted_vol = historical_vol * self.volatility_adjustment
            
            # Smoothly adjust volatility estimate with exponential weighting
            alpha = 0.7  # Weight for new observation
            self.volatility = alpha * adjusted_vol + (1 - alpha) * self.volatility
            
            # Ensure volatility is within reasonable bounds
            self.volatility = min(max(self.volatility, 0.1), 0.8)
            
            # Store for future reference
            self.trader_data[f"{self.symbol}_volatility"] = self.volatility
    
    def fair_price(self, state: TradingState) -> float:
        # Update volatility based on market data
        self.update_volatility(state)
        
        # Convert days to expiry to years
        T = self.remaining_days / 365
        
        # If no days left, option is at intrinsic value
        if self.remaining_days <= 0:
            intrinsic_value = max(0, self.underlying_price - self.strike)
            return int(intrinsic_value)
        
        # If underlying price is available, use Black-Scholes to calculate option value
        if self.underlying_price > 0:
            S = self.underlying_price
            K = self.strike
            r = self.risk_free_rate
            sigma = self.volatility
            
            # Calculate option value
            self.option_value = black_scholes_call(S, K, T, r, sigma)
            
            # Calculate Greeks for risk management
            self.greeks = calculate_option_greeks(S, K, T, r, sigma)
            
            # Store values in trader_data for monitoring
            self.trader_data[f"{self.symbol}_theoretical_value"] = int(self.option_value)
            self.trader_data[f"{self.symbol}_delta"] = self.greeks["delta"]
            self.trader_data[f"{self.symbol}_gamma"] = self.greeks["gamma"]
            self.trader_data[f"{self.symbol}_days_remaining"] = self.remaining_days
            
            # Adjust based on time to expiry - be more conservative as expiry approaches
            if self.remaining_days <= 3:
                # Calculate intrinsic value (what the option is worth if exercised immediately)
                intrinsic_value = max(0, S - K)
                
                # For very close to expiry options, adjust pricing
                if self.remaining_days <= 1:
                    # On last day, option value converges to intrinsic value
                    time_weight = self.time_weight_1day
                    self.option_value = (intrinsic_value * (1 - time_weight) + 
                                         self.option_value * time_weight)
                else:
                    # As expiry approaches, increase weight of intrinsic value
                    time_weight = self.time_weight_3day
                    self.option_value = (intrinsic_value * (1 - time_weight) + 
                                         self.option_value * time_weight)
                
                # In the last days, if the option is far out of the money, be extra cautious
                if S < K - 200:
                    # Deeply OTM options have limited time value
                    epsilon = 5 if self.remaining_days <= 1 else 10
                    self.option_value = min(self.option_value, intrinsic_value + epsilon)
            
            return int(self.option_value)
            
        # If no underlying price available, use market mid price as fair value
        if len(self.order_depth.sell_orders) != 0 and len(self.order_depth.buy_orders) != 0:
            best_ask = min(self.order_depth.sell_orders.keys())
            best_bid = max(self.order_depth.buy_orders.keys())
            return int((best_ask + best_bid) / 2)
            
        return None
    
    def delta_hedge(self, state: TradingState) -> list[Order]:
        """Generate orders to delta hedge the option position"""
        # Only hedge if we have meaningful delta information
        if not self.greeks.get("delta"):
            return []
            
        position = state.position.get(self.symbol, 0)
        if position == 0:
            return []
            
        underlying_symbol = "VOLCANIC_ROCK"
        underlying_position = state.position.get(underlying_symbol, 0)
        
        # Calculate target position in underlying to hedge delta
        # Apply strike-specific delta hedge factor for fine-tuning
        target_hedge = -position * self.greeks["delta"] * self.delta_hedge_factor
        hedge_difference = target_hedge - underlying_position
        
        # Only adjust if difference is significant
        if abs(hedge_difference) < self.delta_hedge_threshold:
            return []
            
        orders = []
        
        # Check if we can place hedging orders
        if underlying_symbol in state.order_depths:
            depth = state.order_depths[underlying_symbol]
            
            if hedge_difference > 0:  # Need to buy underlying
                if len(depth.sell_orders) > 0:
                    best_ask = min(depth.sell_orders.keys())
                    # Ensure quantity is an integer
                    max_qty = min(int(hedge_difference), -depth.sell_orders[best_ask])
                    if max_qty > 0:
                        orders.append(Order(underlying_symbol, int(best_ask), int(max_qty)))
            
            elif hedge_difference < 0:  # Need to sell underlying
                if len(depth.buy_orders) > 0:
                    best_bid = max(depth.buy_orders.keys())
                    # Ensure quantity is an integer
                    max_qty = min(int(-hedge_difference), depth.buy_orders[best_bid])
                    if max_qty > 0:
                        orders.append(Order(underlying_symbol, int(best_bid), -int(max_qty)))
                        
        return orders
    
    def act(self, state: TradingState):
        self.fair_value = self.fair_price(state)
        position = state.position.get(self.symbol, 0)
        buy_order_volume = 0
        sell_order_volume = 0
        
        orders = {self.symbol: [], "VOLCANIC_ROCK": []}
        
        # Generate delta hedging orders for the underlying
        hedge_orders = self.delta_hedge(state)
        if hedge_orders:
            orders["VOLCANIC_ROCK"].extend(hedge_orders)
        
        # Check if we have a valid fair price
        if self.fair_value is None:
            # Fall back to standard market making if no fair price available
            take, buy_order_volume, sell_order_volume = self.take_orders(position, 0, 0)
            clear, buy_order_volume, sell_order_volume = self.clear_orders(position, buy_order_volume, sell_order_volume)
            make, _, _ = self.make_orders(position, buy_order_volume, sell_order_volume)
            orders[self.symbol].extend(take + clear + make)
            return orders
        
        # Ensure fair_value is an integer for calculations
        fair_value_int = int(self.fair_value)
        
        # Use enhanced take orders based on theoretical value
        if len(self.order_depth.sell_orders) != 0:
            best_ask = min(self.order_depth.sell_orders.keys())
            best_ask_amount = -self.order_depth.sell_orders[best_ask]
            
            # Buy when market price is significantly below theoretical value
            if best_ask <= fair_value_int * self.pricing_thresholds[0]:
                quantity = min(best_ask_amount, self.limit - position)
                if quantity > 0:
                    orders[self.symbol].append(Order(self.symbol, int(best_ask), int(quantity)))
                    buy_order_volume += quantity
        
        if len(self.order_depth.buy_orders) != 0:
            best_bid = max(self.order_depth.buy_orders.keys())
            best_bid_amount = self.order_depth.buy_orders[best_bid]
            
            # Sell when market price is significantly above theoretical value
            if best_bid >= fair_value_int * self.pricing_thresholds[1]:
                quantity = min(best_bid_amount, self.limit + position)
                if quantity > 0:
                    orders[self.symbol].append(Order(self.symbol, int(best_bid), -int(quantity)))
                    sell_order_volume += quantity
        
        # Add market making orders
        # Position-aware edge adjustment
        position_ratio = abs(position) / self.limit
        edge_adjustment = int(position_ratio * 3)
        
        # Time-aware edge adjustment - widen spreads as expiry approaches
        time_factor = max(1, 3 - self.remaining_days * 0.3)
        
        ask_edge = max(1, self.default_edge + edge_adjustment)
        bid_edge = max(1, self.default_edge + edge_adjustment)
        
        if position > 0:
            ask_edge = max(1, ask_edge - 1)  # Tighten ask to sell more
        elif position < 0:
            bid_edge = max(1, bid_edge - 1)  # Tighten bid to buy more
        
        # Scale by time factor and ensure integers
        ask_edge = int(ask_edge * time_factor)
        bid_edge = int(bid_edge * time_factor)
        
        # Ensure integer fair value for price calculations
        ask = int(fair_value_int + ask_edge)
        bid = int(fair_value_int - bid_edge)
        
        # Calculate order sizes with position management
        remaining_buy_capacity = self.limit - (position + buy_order_volume)
        remaining_sell_capacity = self.limit + (position - sell_order_volume)
        
        scaling_factor = max(0.3, 1.0 - position_ratio)
        
        buy_qty = min(remaining_buy_capacity, max(1, int(self.limit * self.mm_size_factor * scaling_factor)))
        if buy_qty > 0 and bid < ask:
            orders[self.symbol].append(Order(self.symbol, int(bid), int(buy_qty)))
        
        sell_qty = min(remaining_sell_capacity, max(1, int(self.limit * self.mm_size_factor * scaling_factor)))
        if sell_qty > 0 and ask > bid:
            orders[self.symbol].append(Order(self.symbol, int(ask), -int(sell_qty)))
        
        return orders
            
# Individual strategies for each strike price
class VolcanicRockVoucher9500Strategy(BaseVolcanicRockVoucherStrategy):
    def __init__(self, symbol: str, limit: int, order_depth: OrderDepth, trader_data):
        super().__init__(symbol, limit, order_depth, trader_data)
        # Deep ITM options - less volatility, more aggressive pricing
        self.volatility_adjustment = 0.95
        self.pricing_thresholds = (0.97, 1.03)  # Tighter thresholds
        self.delta_hedge_factor = 0.95  # Slightly less hedging as delta is close to 1
        self.time_weight_1day = 0.15    # More intrinsic value weight (85/15)
        self.mm_size_factor = 0.25      # Larger size for market making
        
class VolcanicRockVoucher9750Strategy(BaseVolcanicRockVoucherStrategy):
    def __init__(self, symbol: str, limit: int, order_depth: OrderDepth, trader_data):
        super().__init__(symbol, limit, order_depth, trader_data)
        # Moderately ITM options
        self.volatility_adjustment = 1
        self.pricing_thresholds = (0.96, 1.04)
        self.delta_hedge_factor = 1.0
        self.mm_size_factor = 0.23
        
class VolcanicRockVoucher10000Strategy(BaseVolcanicRockVoucherStrategy):
    def __init__(self, symbol: str, limit: int, order_depth: OrderDepth, trader_data):
        super().__init__(symbol, limit, order_depth, trader_data)
        # ATM options - highest gamma, moderate delta
        self.volatility_adjustment = 1.05  # Slightly higher vol for ATM
        self.pricing_thresholds = (0.97, 1.03)  # Tighter thresholds for more active trading
        self.delta_hedge_threshold = 4  # More sensitive hedging
        self.delta_hedge_factor = 1.0
        self.mm_size_factor = 0.25  # Larger MM size for ATM options
        
class VolcanicRockVoucher10250Strategy(BaseVolcanicRockVoucherStrategy):
    def __init__(self, symbol: str, limit: int, order_depth: OrderDepth, trader_data):
        super().__init__(symbol, limit, order_depth, trader_data)
        # Moderately OTM options
        self.volatility_adjustment = 1.1  # Higher vol for OTM
        self.pricing_thresholds = (0.95, 1.05)
        self.delta_hedge_factor = 1.05  # Hedge slightly more than delta suggests
        self.mm_size_factor = 0.22
        
class VolcanicRockVoucher10500Strategy(BaseVolcanicRockVoucherStrategy):
    def __init__(self, symbol: str, limit: int, order_depth: OrderDepth, trader_data):
        super().__init__(symbol, limit, order_depth, trader_data)
        # Deep OTM options - higher vol, wider spreads
        self.volatility_adjustment = 1.15
        self.pricing_thresholds = (0.93, 1.07)  # Wider thresholds
        self.delta_hedge_factor = 1.1   # Over-hedge slightly
        self.delta_hedge_threshold = 3  # More sensitive hedging
        self.mm_size_factor = 0.2       # Smaller size for MM

class VolcanicRockStrategy(MarketMakeStrategy):
    def __init__(self, symbol: str, limit: int, order_depth: OrderDepth, trader_data):
        super().__init__(symbol, limit, order_depth, trader_data)
        
        # Market making parameters
        self.take_width = 2
        self.clear_width = 1
        self.join_edge = 1
        self.default_edge = 2
        self.prevent_adverse = True
        self.adverse_volume = 10
        self.soft_position_limit = int(limit * 0.7)
        
        # Option-awareness parameters
        self.delta_hedge_weight = 1.4 # Weight for net delta
        
        # For tracking
        self.fair_value = 0
        
    def fair_price(self, state: TradingState) -> float:
        """Calculate fair price for volcanic rock based on order book"""
        if len(self.order_depth.sell_orders) != 0 and len(self.order_depth.buy_orders) != 0:
            best_ask = min(self.order_depth.sell_orders.keys())
            best_bid = max(self.order_depth.buy_orders.keys())
            
            # Simple mid price - ensure it's an integer
            fair = int((best_ask + best_bid) / 2)
            
            # Store price history for volatility estimation
            if "VOLCANIC_ROCK_price_history" not in self.trader_data:
                self.trader_data["VOLCANIC_ROCK_price_history"] = deque(maxlen=50)
            
            self.trader_data["VOLCANIC_ROCK_price_history"].append(fair)
            self.trader_data["VOLCANIC_ROCK_mid_price"] = fair
            
            return fair
        return None

    def act(self, state: TradingState):
        self.fair_value = self.fair_price(state)
        position = state.position.get(self.symbol, 0)
        
        # Check if we need to adjust strategy based on options exposure
        net_delta = 0
        strike_specific_vouchers = {
            "VOLCANIC_ROCK_VOUCHER_9500": VolcanicRockVoucher9500Strategy,
            "VOLCANIC_ROCK_VOUCHER_9750": VolcanicRockVoucher9750Strategy,
            "VOLCANIC_ROCK_VOUCHER_10000": VolcanicRockVoucher10000Strategy,
            "VOLCANIC_ROCK_VOUCHER_10250": VolcanicRockVoucher10250Strategy,
            "VOLCANIC_ROCK_VOUCHER_10500": VolcanicRockVoucher10500Strategy
        }
        
        # Calculate net delta from all vouchers
        for voucher in strike_specific_vouchers.keys():
            voucher_delta = self.trader_data.get(f"{voucher}_delta", 0)
            voucher_position = state.position.get(voucher, 0)
            # Apply hedge weight factor to allow for more balanced market making
            net_delta += voucher_delta * voucher_position * self.delta_hedge_weight
        
        # Adjust position limit based on options exposure and ensure it's an integer
        effective_position = int(position + net_delta)
        
        # Calculate gamma exposure for risk management
        net_gamma = 0
        for voucher in strike_specific_vouchers.keys():
            voucher_gamma = self.trader_data.get(f"{voucher}_gamma", 0)
            voucher_position = state.position.get(voucher, 0)
            net_gamma += voucher_gamma * voucher_position
        
        # Adjust market making parameters based on gamma exposure
        if abs(net_gamma) > 0.01:
            # High gamma means high risk - reduce order sizes and widen spreads
            self.default_edge = max(2, min(4, int(2 + abs(net_gamma) * 100)))
        else:
            # Low gamma - normal market making
            self.default_edge = 2
        
        buy_order_volume = 0
        sell_order_volume = 0
        
        # Use all three order types: taking, clearing, and making
        take, buy_order_volume, sell_order_volume = self.take_orders(effective_position, buy_order_volume, sell_order_volume)
        clear, buy_order_volume, sell_order_volume = self.clear_orders(effective_position, buy_order_volume, sell_order_volume)
        make, _, _ = self.make_orders(effective_position, buy_order_volume, sell_order_volume)

        # Combine all order types
        return {self.symbol: take + clear + make}

class Trader:
    def __init__(self):
        self.limits = {
            "RAINFOREST_RESIN": 50,
            "KELP": 50,
            "SQUID_INK": 50,
            "CROISSANTS": 250,
            "JAMS": 350,
            "DJEMBES": 60,
            "PICNIC_BASKET1": 60,
            "PICNIC_BASKET2": 100,
            "VOLCANIC_ROCK": 400,
            "VOLCANIC_ROCK_VOUCHER_9500": 200,
            "VOLCANIC_ROCK_VOUCHER_9750": 200,
            "VOLCANIC_ROCK_VOUCHER_10000": 200, 
            "VOLCANIC_ROCK_VOUCHER_10250": 200,
            "VOLCANIC_ROCK_VOUCHER_10500": 200
        }

    def run(self, state : TradingState) -> tuple[dict[Symbol, list[Order]], int , str]:
        trader_data = {}
        if state.traderData != None and state.traderData != "":
            trader_data = jsonpickle.decode(state.traderData)
            
        # Track the current round for option expiration tracking
        if "current_round" not in trader_data:
            trader_data["current_round"] = 1
        else:
            trader_data["current_round"] += 1
            
        # Map symbols to their appropriate strategy classes
        strategy_map = {
            # "RAINFOREST_RESIN": ResinStrategy, 
            # "KELP": KelpStrategy, 
            # "SQUID_INK": SquidInkStrategy,
            # "PICNIC_BASKET1": PicnicBasket1Strategy, 
            # "PICNIC_BASKET2": PicnicBasket2Strategy,
            "VOLCANIC_ROCK": VolcanicRockStrategy,
            "VOLCANIC_ROCK_VOUCHER_9500": VolcanicRockVoucher9500Strategy,
            "VOLCANIC_ROCK_VOUCHER_9750": VolcanicRockVoucher9750Strategy,
            "VOLCANIC_ROCK_VOUCHER_10000": VolcanicRockVoucher10000Strategy,
            "VOLCANIC_ROCK_VOUCHER_10250": VolcanicRockVoucher10250Strategy,
            "VOLCANIC_ROCK_VOUCHER_10500": VolcanicRockVoucher10500Strategy
        }
            
        # Initialize strategies only for symbols present in the order book
        strategies = {
            symbol: constructor(symbol, self.limits[symbol], state.order_depths[symbol], trader_data) 
            for symbol, constructor in strategy_map.items() 
            if symbol in state.order_depths
        }
            
        conversions = 0
        orders = {}
        
        # Process all strategies
        for symbol, strategy in strategies.items():
            if symbol in state.order_depths:
                orders_dict = strategy.act(state)
                # Merge orders from each strategy
                for s, order_list in orders_dict.items():
                    if s not in orders:
                        orders[s] = []
                    orders[s].extend(order_list)
        
        new_trader_data = jsonpickle.encode(trader_data)
        logger.flush(state, orders, conversions, new_trader_data)
        return orders, conversions, new_trader_data
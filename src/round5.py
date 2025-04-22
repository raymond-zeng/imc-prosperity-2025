import jsonpickle
from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState
import numpy as np
import math
from collections import deque
from statistics import NormalDist

# f = open("test.txt", "w")
import json
from typing import Any
from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState

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
                        getattr(trade, "counter_party", None)  # safe for market trades
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
        
class BaseVolcanicRockVoucherStrategy(MarketMakeStrategy):
    def __init__(self, symbol: str, limit: int, order_depth: OrderDepth, trader_data):
        super().__init__(symbol, limit, order_depth, trader_data)
        
        # Extract strike price from symbol
        self.strike_price = int(symbol.split("_")[-1])
        
        # Option parameters
        self.days_to_expiry = 4 # Round 3 means 5 days left (7-3+1)

        self.risk_free_rate = 0.00  # Assumed 1% risk-free rate
        self.take_width = 2
        self.clear_width = 1
        self.disregard_edge = 1
        self.join_edge = 0
        self.default_edge = 2
        self.prevent_adverse = True
        # self.adverse_volume = 15
        self.volatility_estimate = 0.25  # Initial estimate, will be adjusted
        
        # Iron condor parameters
        self.width = 200  # Wing width
        self.narrow_range = 100  # How narrow/wide the iron condor is
        self.target_profit = 0.05  # Target profit as percentage of width
        
        # Bollinger Bands parameters
        self.bollinger_window = 20
        self.bollinger_std = 2.0
        self.band_trade_threshold = 0.75  # How close to bands to trigger trades (0-1)
        
        # Initialize price history for Bollinger Bands
        if f"{self.symbol}_price_history" not in self.trader_data:
            self.trader_data[f"{self.symbol}_price_history"] = deque(maxlen=50)
        
        # Delta hedging parameters
        self.hedge_ratio = 0.5  # How aggressively to hedge delta exposure
        self.delta_limit = 0.75 * limit  # Maximum tolerated delta exposure
        
        # Iron condor state tracking
        self.leg_positions = {
            "lower_long": 0,  # Long put further OTM
            "lower_short": 0,  # Short put closer to ATM
            "upper_short": 0,  # Short call closer to ATM
            "upper_long": 0,   # Long call further OTM
        }

    def black_scholes_call(self, spot_price, strike, time_to_expiry, risk_free_rate, volatility):
        """Calculate Black-Scholes price for a call option"""
        if time_to_expiry <= 0:
            # Option at expiration is worth max(0, spot - strike)
            return max(0, spot_price - strike)
        
        # Black-Scholes formula components
        d1 = (math.log(spot_price / strike) + (risk_free_rate + 0.5 * volatility**2) * time_to_expiry) / (volatility * math.sqrt(time_to_expiry))
        d2 = d1 - volatility * math.sqrt(time_to_expiry)
        
        # Normal CDF calculation using statistics module
        nd = NormalDist(mu=0, sigma=1)
        N_d1 = nd.cdf(d1)
        N_d2 = nd.cdf(d2)
        
        # Call price
        call_price = spot_price * N_d1 - strike * math.exp(-risk_free_rate * time_to_expiry) * N_d2
        
        return call_price
        
    def black_scholes_put(self, spot_price, strike, time_to_expiry, risk_free_rate, volatility):
        """Calculate Black-Scholes price for a put option using put-call parity"""
        # Put price via put-call parity
        call_price = self.black_scholes_call(spot_price, strike, time_to_expiry, risk_free_rate, volatility)
        put_price = call_price + strike * math.exp(-risk_free_rate * time_to_expiry) - spot_price
        
        return put_price

    def estimate_implied_volatility(self, market_price, spot_price, strike, time_to_expiry, risk_free_rate, precision=0.001, is_call=True):
        """Estimate implied volatility using bisection method"""
        if market_price <= 0:
            return 0.01  # Minimum volatility floor
            
        # Define upper and lower bounds for volatility search
        vol_low = 0.01
        vol_high = 2.0
        
        # Set initial volatility guess
        vol_mid = (vol_low + vol_high) / 2
        
        # Run bisection algorithm
        for _ in range(50):  # Maximum 50 iterations
            if is_call:
                price = self.black_scholes_call(spot_price, strike, time_to_expiry, risk_free_rate, vol_mid)
            else:
                price = self.black_scholes_put(spot_price, strike, time_to_expiry, risk_free_rate, vol_mid)
            
            if abs(price - market_price) < precision:
                return vol_mid
            
            if price > market_price:
                vol_high = vol_mid
            else:
                vol_low = vol_mid
                
            vol_mid = (vol_low + vol_high) / 2
            
        return vol_mid  # Return best approximation

    def calculate_bollinger_bands(self, prices):
        """Calculate Bollinger Bands from price history"""
        if len(prices) < 5:
            # Not enough data points
            return None, None, None
            
        prices_array = np.array(prices)
        sma = np.mean(prices_array)
        std_dev = np.std(prices_array)
        
        upper_band = sma + self.bollinger_std * std_dev
        lower_band = sma - self.bollinger_std * std_dev
        
        return lower_band, sma, upper_band
        
    def calculate_iron_condor_strikes(self, spot_price):
        """Calculate the four strikes for an iron condor based on current spot price"""
        # Calculate the 4 strikes for the iron condor
        lower_long_strike = round(spot_price - self.width - self.narrow_range)
        lower_short_strike = round(spot_price - self.narrow_range)
        upper_short_strike = round(spot_price + self.narrow_range)
        upper_long_strike = round(spot_price + self.width + self.narrow_range)
        
        return {
            "lower_long": lower_long_strike,
            "lower_short": lower_short_strike, 
            "upper_short": upper_short_strike,
            "upper_long": upper_long_strike
        }
        
    def calculate_iron_condor_fair_value(self, spot_price, time_to_expiry, volatility):
        """Calculate the fair value of an iron condor spread"""
        strikes = self.calculate_iron_condor_strikes(spot_price)
        
        # Calculate prices for each leg
        lower_long_put = self.black_scholes_put(spot_price, strikes["lower_long"], time_to_expiry, 
                                               self.risk_free_rate, volatility)
        lower_short_put = self.black_scholes_put(spot_price, strikes["lower_short"], time_to_expiry, 
                                                self.risk_free_rate, volatility)
        upper_short_call = self.black_scholes_call(spot_price, strikes["upper_short"], time_to_expiry, 
                                                  self.risk_free_rate, volatility)
        upper_long_call = self.black_scholes_call(spot_price, strikes["upper_long"], time_to_expiry, 
                                                 self.risk_free_rate, volatility)
        
        # Iron condor value = credit received - debit paid
        # Buy lower_long_put, sell lower_short_put, sell upper_short_call, buy upper_long_call
        iron_condor_value = (lower_short_put - lower_long_put) + (upper_short_call - upper_long_call)
        
        # Store strikes for trading decisions
        self.trader_data[f"{self.symbol}_ic_strikes"] = strikes
        
        return iron_condor_value
        
    def evaluate_existing_positions(self, spot_price, time_to_expiry, volatility):
        """Evaluate profitability of existing iron condor positions"""
        if "iron_condor_positions" not in self.trader_data:
            self.trader_data["iron_condor_positions"] = []
            return None
            
        positions = self.trader_data["iron_condor_positions"]
        total_profit = 0
        
        for position in positions:
            entry_price = position["entry_price"]
            strikes = position["strikes"]
            size = position["size"]
            
            # Calculate current prices for each leg
            lower_long_put = self.black_scholes_put(spot_price, strikes["lower_long"], time_to_expiry, 
                                                  self.risk_free_rate, volatility)
            lower_short_put = self.black_scholes_put(spot_price, strikes["lower_short"], time_to_expiry, 
                                                   self.risk_free_rate, volatility)
            upper_short_call = self.black_scholes_call(spot_price, strikes["upper_short"], time_to_expiry, 
                                                     self.risk_free_rate, volatility)
            upper_long_call = self.black_scholes_call(spot_price, strikes["upper_long"], time_to_expiry, 
                                                    self.risk_free_rate, volatility)
            
            # Current iron condor value
            current_value = (lower_short_put - lower_long_put) + (upper_short_call - upper_long_call)
            
            # Calculate profit (for a short iron condor)
            position_profit = (entry_price - current_value) * size
            total_profit += position_profit
            
            # Update position data
            position["current_value"] = current_value
            position["profit"] = position_profit
            
        return total_profit

    def fair_price(self, state: TradingState) -> float:
        """Calculate fair price for the voucher based on our trading approach"""
        # Get current price of VOLCANIC_ROCK
        spot_price = None
        if "VOLCANIC_ROCK" in state.order_depths:
            rock_depth = state.order_depths["VOLCANIC_ROCK"]
            if len(rock_depth.buy_orders) > 0 and len(rock_depth.sell_orders) > 0:
                best_bid = max(rock_depth.buy_orders.keys())
                best_ask = min(rock_depth.sell_orders.keys())
                spot_price = (best_bid + best_ask) / 2
        
        # If we can't determine spot price, try to use historical data or default
        if spot_price is None:
            spot_price = self.trader_data.get("VOLCANIC_ROCK_last_price", 10200)
        
        # Update spot price in trader data for future reference
        self.trader_data["VOLCANIC_ROCK_last_price"] = spot_price
        
        # Calculate time to expiry in years (assuming 365 days/year)
        time_to_expiry = (self.days_to_expiry - state.timestamp / 1000000) / 365.0
        
        # Calculate if we're a call or put based on our strike price
        is_call = self.strike_price >= spot_price
        
        # Use market prices to estimate implied volatility if possible
        if len(self.order_depth.buy_orders) > 0 and len(self.order_depth.sell_orders) > 0:
            market_bid = max(self.order_depth.buy_orders.keys())
            market_ask = min(self.order_depth.sell_orders.keys())
            market_price = (market_bid + market_ask) / 2
            
            # Add current price to history for Bollinger Bands
            self.trader_data[f"{self.symbol}_price_history"].append(market_price)
            
            # Update volatility estimate based on market prices
            implied_vol = self.estimate_implied_volatility(
                market_price, spot_price, self.strike_price, time_to_expiry, self.risk_free_rate, is_call=is_call
            )
            
            # Apply smoothing to volatility updates
            if f"{self.symbol}_implied_vol" in self.trader_data:
                old_vol = self.trader_data[f"{self.symbol}_implied_vol"]
                # Exponential moving average with 0.7 weight to new observation
                implied_vol = 0.7 * implied_vol + 0.3 * old_vol
            
            self.trader_data[f"{self.symbol}_implied_vol"] = implied_vol
            self.volatility_estimate = implied_vol
        else:
            # If no market data, use historical volatility or default
            self.volatility_estimate = self.trader_data.get(f"{self.symbol}_implied_vol", 0.25)
        
        # Calculate option price based on Black-Scholes
        if is_call:
            option_price = self.black_scholes_call(
                spot_price,
                self.strike_price,
                time_to_expiry,
                self.risk_free_rate,
                self.volatility_estimate
            )
        else:
            option_price = self.black_scholes_put(
                spot_price,
                self.strike_price,
                time_to_expiry,
                self.risk_free_rate,
                self.volatility_estimate
            )
        
        # Store the theoretical price for later reference
        self.trader_data[f"{self.symbol}_theoretical_price"] = option_price
        
        # Calculate iron condor value
        iron_condor_value = self.calculate_iron_condor_fair_value(
            spot_price, time_to_expiry, self.volatility_estimate)
        self.trader_data[f"{self.symbol}_ic_value"] = iron_condor_value
        
        # Evaluate existing positions
        position_profit = self.evaluate_existing_positions(
            spot_price, time_to_expiry, self.volatility_estimate)
        self.trader_data[f"{self.symbol}_position_profit"] = position_profit
        
        # Calculate Bollinger Bands
        price_history = list(self.trader_data[f"{self.symbol}_price_history"])
        if len(price_history) >= 5:
            lower_band, sma, upper_band = self.calculate_bollinger_bands(price_history)
            self.trader_data[f"{self.symbol}_bb_lower"] = lower_band
            self.trader_data[f"{self.symbol}_bb_sma"] = sma  
            self.trader_data[f"{self.symbol}_bb_upper"] = upper_band
            
            # Adjust theoretical price based on Bollinger Bands
            if len(price_history) > self.bollinger_window / 2:
                last_price = price_history[-1]
                # If price is near upper band, expect reversion (lower fair value)
                if upper_band and last_price > sma + (upper_band - sma) * self.band_trade_threshold:
                    option_price = option_price * 0.98
                # If price is near lower band, expect reversion (higher fair value)  
                elif lower_band and last_price < sma - (sma - lower_band) * self.band_trade_threshold:
                    option_price = option_price * 1.02
        
        # Additional metrics for trading decisions
        self.trader_data[f"{self.symbol}_moneyness"] = spot_price / self.strike_price
        self.trader_data[f"{self.symbol}_days_to_expiry"] = self.days_to_expiry
        
        # Calculate Greeks for position management
        delta = self.calculate_delta(spot_price, self.strike_price, time_to_expiry, self.risk_free_rate, 
                                    self.volatility_estimate, is_call)
        theta = self.calculate_theta(spot_price, self.strike_price, time_to_expiry, self.risk_free_rate, 
                                    self.volatility_estimate, is_call)
        gamma = self.calculate_gamma(spot_price, self.strike_price, time_to_expiry, self.risk_free_rate, 
                                    self.volatility_estimate)
        
        self.trader_data[f"{self.symbol}_delta"] = delta
        self.trader_data[f"{self.symbol}_theta"] = theta
        self.trader_data[f"{self.symbol}_gamma"] = gamma
        
        return option_price
    
    def calculate_delta(self, spot_price, strike, time_to_expiry, risk_free_rate, volatility, is_call=True):
        """Calculate delta (option's sensitivity to underlying price change)"""
        if time_to_expiry <= 0:
            if is_call:
                return 1.0 if spot_price > strike else 0.0
            else:
                return -1.0 if spot_price < strike else 0.0
        
        d1 = (math.log(spot_price / strike) + (risk_free_rate + 0.5 * volatility**2) * time_to_expiry) / (volatility * math.sqrt(time_to_expiry))
        nd = NormalDist(mu=0, sigma=1)
        
        if is_call:
            return nd.cdf(d1)
        else:
            return nd.cdf(d1) - 1  # Put delta = Call delta - 1
    
    def calculate_gamma(self, spot_price, strike, time_to_expiry, risk_free_rate, volatility):
        """Calculate gamma (rate of change of delta) - same for calls and puts"""
        if time_to_expiry <= 0:
            return 0.0
            
        d1 = (math.log(spot_price / strike) + (risk_free_rate + 0.5 * volatility**2) * time_to_expiry) / (volatility * math.sqrt(time_to_expiry))
        
        # Standard normal PDF
        pdf_d1 = math.exp(-0.5 * d1**2) / math.sqrt(2 * math.pi)
        
        gamma = pdf_d1 / (spot_price * volatility * math.sqrt(time_to_expiry))
        return gamma
    
    def calculate_theta(self, spot_price, strike, time_to_expiry, risk_free_rate, volatility, is_call=True):
        """Calculate theta (option's time decay)"""
        if time_to_expiry <= 0:
            return 0.0
            
        d1 = (math.log(spot_price / strike) + (risk_free_rate + 0.5 * volatility**2) * time_to_expiry) / (volatility * math.sqrt(time_to_expiry))
        d2 = d1 - volatility * math.sqrt(time_to_expiry)
        
        nd = NormalDist(mu=0, sigma=1)
        
        # Standard normal PDF
        pdf_d1 = math.exp(-0.5 * d1**2) / math.sqrt(2 * math.pi)
        
        # Different calculation for call vs put
        if is_call:
            theta = -spot_price * pdf_d1 * volatility / (2 * math.sqrt(time_to_expiry))
            theta -= risk_free_rate * strike * math.exp(-risk_free_rate * time_to_expiry) * nd.cdf(d2)
        else:
            theta = -spot_price * pdf_d1 * volatility / (2 * math.sqrt(time_to_expiry))
            theta += risk_free_rate * strike * math.exp(-risk_free_rate * time_to_expiry) * nd.cdf(-d2)
        
        # Convert to daily theta (from annual)
        return theta / 365.0
        
    def should_open_iron_condor(self, spot_price, option_price):
        """Determine if we should open a new iron condor position"""
        # Don't open if we're close to expiration (too risky)
        if self.days_to_expiry <= 2:
            return False
            
        # Get the iron condor value
        ic_value = self.trader_data.get(f"{self.symbol}_ic_value", 0)
        
        # Width between short strikes
        strikes = self.trader_data.get(f"{self.symbol}_ic_strikes", {})
        if not strikes:
            return False
            
        width = strikes["upper_short"] - strikes["lower_short"]
        
        # Check if premium is acceptable relative to width
        # We want to collect at least target_profit * width
        min_premium = width * self.target_profit
        
        # For short iron condor, we want the premium to be high enough
        if ic_value >= min_premium:
            # Check if underlying is within acceptable range
            if (strikes["lower_short"] < spot_price < strikes["upper_short"]):
                # Don't open too many positions
                max_positions = 5
                current_positions = len(self.trader_data.get("iron_condor_positions", []))
                if current_positions < max_positions:
                    return True
                    
        return False
        
    def should_close_iron_condor(self, spot_price):
        """Determine if we should close existing iron condor positions"""
        if "iron_condor_positions" not in self.trader_data:
            return False
            
        positions = self.trader_data["iron_condor_positions"]
        if not positions:
            return False
            
        # Close if we've captured most of the value
        for position in positions:
            # For short iron condor, we're profitable when value decreases
            entry_price = position["entry_price"]
            current_value = position.get("current_value", entry_price)
            
            # Close if we've captured 80% of max profit
            if current_value <= 0.2 * entry_price:
                return True
                
            # Close if we're close to expiration
            if self.days_to_expiry <= 1:
                return True
                
            # Close if price moved outside our short strikes (risk management)
            strikes = position["strikes"]
            if spot_price <= strikes["lower_short"] or spot_price >= strikes["upper_short"]:
                # Only close if we're not too far out (might be better to adjust)
                buffer = 100
                if (strikes["lower_short"] - buffer <= spot_price <= strikes["upper_short"] + buffer):
                    return True
                    
        return False

    def execute_iron_condor(self, position, is_opening):
        """Execute orders to open or close an iron condor position"""
        orders = []
        
        # Get the strikes
        strikes = position["strikes"]
        size = position["size"]
        
        # For simplicity, we'll simulate the iron condor using a single voucher
        # This is a simplification since we can't actually trade individual spreads in this game
        
        if is_opening:
            # Opening iron condor is net credit (we receive premium)
            # We're selling the position, so negative quantity
            orders.append(Order(self.symbol, round(self.fair_value), -size))  # Added round() here
        else:
            # Closing iron condor is a buy (we pay to close)
            # We're buying back the position, so positive quantity
            orders.append(Order(self.symbol, round(self.fair_value), size))  # Added round() here
            
        return orders
        
    def adjust_parameters_for_strike(self):
        """Adjust strategy parameters based on our specific strike price"""
        # This depends on which voucher we're trading
        spot_price = self.trader_data.get("VOLCANIC_ROCK_last_price", 10000)
        moneyness = spot_price / self.strike_price
        
        # Adjust iron condor width based on strike and days to expiry
        if self.days_to_expiry <= 2:
            # Narrower condor close to expiry
            self.width = 150
            self.narrow_range = 75
        elif 0.95 <= moneyness <= 1.05:
            # Near ATM, use standard width
            self.width = 200
            self.narrow_range = 100
        else:
            # Far OTM/ITM, wider condor
            self.width = 250
            self.narrow_range = 150
    
    def act(self, state: TradingState):
        # Calculate fair value of our option
        self.fair_value = self.fair_price(state)
        
        # Get current position as integer
        position_int = state.position.get(self.symbol, 0)
        
        # Get volcanic rock price
        spot_price = self.trader_data.get("VOLCANIC_ROCK_last_price", 10200)
        
        # Adjust parameters based on strike price
        self.adjust_parameters_for_strike()
        
        # Iron condor logic
        if "iron_condor_positions" not in self.trader_data:
            self.trader_data["iron_condor_positions"] = []
            
        # Check if we should open or close iron condor positions
        open_new = self.should_open_iron_condor(spot_price, self.fair_value)
        close_existing = self.should_close_iron_condor(spot_price)
        
        iron_condor_orders = []
        
        if open_new:
            # Calculate size based on available position limit
            available_capacity = int((self.limit - abs(position_int)) * 0.5)  # Use at most 50% of remaining capacity
            if available_capacity >= 10:
                # Create new position
                new_position = {
                    "entry_price": self.trader_data.get(f"{self.symbol}_ic_value", 0),
                    "strikes": self.trader_data.get(f"{self.symbol}_ic_strikes", {}),
                    "size": min(available_capacity, 20),  # Cap at 20 contracts per iron condor
                    "entry_time": state.timestamp
                }
                
                # Execute orders
                ic_orders = self.execute_iron_condor(new_position, True)
                iron_condor_orders.extend(ic_orders)
                
                # Save position
                self.trader_data["iron_condor_positions"].append(new_position)
                
        if close_existing:
            positions = self.trader_data["iron_condor_positions"]
            for i, position in enumerate(positions):
                # Execute orders to close
                ic_orders = self.execute_iron_condor(position, False)
                iron_condor_orders.extend(ic_orders)
                
            # Clear closed positions
            if close_existing:
                self.trader_data["iron_condor_positions"] = []
                
        # Market making component - supplement iron condor strategy with regular market making
        buy_order_volume = 0
        sell_order_volume = 0
        
        # For orders already generated by iron condor strategy, track their volumes
        for order in iron_condor_orders:
            if order.quantity > 0:
                buy_order_volume += order.quantity
            else:
                sell_order_volume += abs(order.quantity)
        
        # Calculate remaining capacity for market making
        position_after_ic = position_int + buy_order_volume - sell_order_volume
        
        # Regular market making
        take, buy_order_volume, sell_order_volume = self.take_orders(position_after_ic, buy_order_volume, sell_order_volume)
        clear, buy_order_volume, sell_order_volume = self.clear_orders(position_after_ic, buy_order_volume, sell_order_volume)
        make, _, _ = self.make_orders(position_after_ic, buy_order_volume, sell_order_volume)
        
        # Combine all orders
        all_orders = iron_condor_orders + take + clear + make
        
        return {self.symbol: all_orders}

    def adjust_parameters(self, delta, moneyness):
        """Adjust strategy parameters based on option characteristics"""
        # This method should be overridden by child classes
        # For standard case, use delta-based position limits
        if delta > 0.7:
            self.soft_position_limit = int(self.limit * 0.6)
        elif delta < 0.3:
            self.soft_position_limit = int(self.limit * 0.9)
        else:
            self.soft_position_limit = int(self.limit * 0.8)

class Voucher9500Strategy(BaseVolcanicRockVoucherStrategy):
    def __init__(self, symbol: str, limit: int, order_depth: OrderDepth, trader_data):
        super().__init__(symbol, limit, order_depth, trader_data)
        # Deep ITM parameters
        self.bollinger_std = 2.2  # Wider bands for ITM options
        self.band_trade_threshold = 0.8  # More conservative mean reversion
        self.hedge_ratio = 0.7  # Higher hedge ratio for high delta options
    
    def adjust_parameters(self, delta, moneyness):
        # ITM options - adjust for high delta
        self.soft_position_limit = int(self.limit * 0.55)  # More conservative
        self.take_width = 1  # Tighter spreads
        self.clear_width = 0  # Aggressive clearing
        self.default_edge = 1
        
        # Adjust for gamma risk which is lower for deep ITM
        gamma = self.trader_data.get(f"{self.symbol}_gamma", 0)
        if gamma < 0.0001:
            self.take_width = 0  # Very tight spreads for low gamma (less risk)


class Voucher9750Strategy(BaseVolcanicRockVoucherStrategy):
    def __init__(self, symbol: str, limit: int, order_depth: OrderDepth, trader_data):
        super().__init__(symbol, limit, order_depth, trader_data)
        # Slightly ITM parameters
        self.bollinger_std = 2.1
        self.volatility_estimate = 0.28  # Slightly higher vol estimate
    
    def adjust_parameters(self, delta, moneyness):
        # Slightly ITM options
        self.soft_position_limit = int(self.limit * 0.65)
        self.take_width = 1
        self.clear_width = 0


class Voucher10000Strategy(BaseVolcanicRockVoucherStrategy):
    def __init__(self, symbol: str, limit: int, order_depth: OrderDepth, trader_data):
        super().__init__(symbol, limit, order_depth, trader_data)
        # ATM parameters - highest gamma
        self.bollinger_std = 1.8  # Tighter bands for ATM (more volatile)
        self.band_trade_threshold = 0.7
        self.volatility_estimate = 0.3  # Higher vol for ATM
    
    def adjust_parameters(self, delta, moneyness):
        # ATM options - highest gamma risk
        gamma = self.trader_data.get(f"{self.symbol}_gamma", 0)
        
        # Adjust based on moneyness
        if 0.98 <= moneyness <= 1.02:  # Very close to ATM
            self.soft_position_limit = int(self.limit * 0.6)  # More conservative for high gamma
            self.take_width = 2  # Wider spread for gamma risk
            self.clear_width = 1
            self.default_edge = 2
        else:
            self.soft_position_limit = int(self.limit * 0.7)
            self.take_width = 1
            self.default_edge = 1

class Voucher10250Strategy(BaseVolcanicRockVoucherStrategy):
    def __init__(self, symbol: str, limit: int, order_depth: OrderDepth, trader_data):
        super().__init__(symbol, limit, order_depth, trader_data)
        # Slightly OTM parameters
        self.bollinger_std = 2.0
        self.volatility_estimate = 0.27
        self.bollinger_window = 15  # Shorter window for faster reactions
    
    def adjust_parameters(self, delta, moneyness):
        # Slightly OTM options
        if delta < 0.4:
            self.soft_position_limit = int(self.limit * 0.75)
            self.take_width = 1
            self.clear_width = 1

class Voucher10500Strategy(BaseVolcanicRockVoucherStrategy):
    def __init__(self, symbol: str, limit: int, order_depth: OrderDepth, trader_data):
        super().__init__(symbol, limit, order_depth, trader_data)
        # Deep OTM parameters 
        self.bollinger_std = 2.5  # Wider bands for less liquid options
        self.band_trade_threshold = 0.85  # More conservative for OTM
        self.bollinger_window = 10  # Shorter lookback
    
    def adjust_parameters(self, delta, moneyness):
        # Deep OTM options - can be more aggressive with position
        if self.days_to_expiry <= 2 and delta < 0.1:
            # Very little time value left for deep OTM
            self.soft_position_limit = int(self.limit * 0.5)
            self.take_width = 3  # Wider spreads, less trading
            self.clear_width = 2
        else:
            self.soft_position_limit = int(self.limit * 0.85)
            self.take_width = 2
            self.clear_width = 1

class VolcanicRockSmileStrategy(MarketMakeStrategy):
    def __init__(self, symbol: str, limit: int, order_depth: OrderDepth, trader_data):
        super().__init__(symbol, limit, order_depth, trader_data)
        
        # Strategy parameters
        self.take_width = 2
        self.clear_width = 1
        self.disregard_edge = 2
        self.join_edge = 1 
        self.default_edge = 3
        self.prevent_adverse = True
        
        # Volatility smile parameters
        self.smile_a = 0.2373  # x^2 coefficient
        self.smile_b = 0.0029  # x coefficient
        self.smile_c = 0.1492  # constant term
        
        # Risk management parameters
        self.skew_threshold = 0.02  # Threshold for significant skew
        self.vol_multiplier = 25  # How much to scale volatility for position sizing
        
        # Initialize volatility tracking
        if "vol_surface" not in self.trader_data:
            self.trader_data["vol_surface"] = {}
    
    def calculate_implied_vol(self, moneyness):
        """Calculate implied volatility from moneyness using smile equation"""
        x = moneyness - 1.0  # Convert to normalized moneyness
        return self.smile_a * (x**2) + self.smile_b * x + self.smile_c
    
    def estimate_skew_and_convexity(self, spot_price):
        """Calculate volatility skew (smile asymmetry) and convexity"""
        # Use voucher strikes to sample the volatility surface
        strikes = [9500, 9750, 10000, 10250, 10500]
        vols = []
        
        for strike in strikes:
            moneyness = spot_price / strike
            vol = self.calculate_implied_vol(moneyness)
            vols.append(vol)
            self.trader_data["vol_surface"][strike] = vol
        
        # Calculate skew from left and right wing difference
        if len(vols) >= 5:
            left_wing = vols[0] - vols[2]  # Vol(9500) - Vol(10000)
            right_wing = vols[4] - vols[2]  # Vol(10500) - Vol(10000)
            skew = left_wing - right_wing
            convexity = vols[0] + vols[4] - 2*vols[2]  # Approx. of second derivative
            
            self.trader_data["vol_skew"] = skew
            self.trader_data["vol_convexity"] = convexity
            return skew, convexity
            
        return 0, 0
    
    def fair_price(self, state: TradingState) -> float:
        """Calculate fair price using market mid and volatility information"""
        # Get current market mid price
        if len(self.order_depth.sell_orders) == 0 or len(self.order_depth.buy_orders) == 0:
            return self.trader_data.get("VOLCANIC_ROCK_last_fair", 10000)
            
        best_bid = max(self.order_depth.buy_orders.keys())
        best_ask = min(self.order_depth.sell_orders.keys())
        mid_price = (best_bid + best_ask) / 2
        
        # Track current ATM volatility (at K=10000)
        atm_moneyness = mid_price / 10000
        atm_vol = self.calculate_implied_vol(atm_moneyness)
        self.trader_data["atm_vol"] = atm_vol
        
        # Calculate volatility skew and convexity
        skew, convexity = self.estimate_skew_and_convexity(mid_price)
        
        # Adjust fair price based on skew (volatility smile asymmetry)
        # Positive skew (left wing higher than right) suggests downward pressure
        # Negative skew (right wing higher than left) suggests upward pressure
        skew_adjustment = -skew * self.skew_threshold * mid_price
        
        # The adjustment is negative when skew is positive (downward pressure)
        # and positive when skew is negative (upward pressure)
        fair_price = mid_price + skew_adjustment
        
        self.trader_data["VOLCANIC_ROCK_last_fair"] = fair_price
        return fair_price
    
    def act(self, state: TradingState):
        # Calculate fair value
        self.fair_value = self.fair_price(state)
        position = state.position.get(self.symbol, 0)
        
        # Adjust position sizing based on ATM volatility
        atm_vol = self.trader_data.get("atm_vol", 0.20)
        skew = self.trader_data.get("vol_skew", 0)
        
        # Reduce position size when volatility is high
        vol_factor = max(0.4, min(1.0, 1.0 - (atm_vol - 0.15) * self.vol_multiplier))
        self.soft_position_limit = int(self.limit * vol_factor)
        
        # Adjust trading parameters based on skew and volatility
        if abs(skew) > self.skew_threshold:
            # When skew is significant, be more aggressive with clearing
            self.clear_width = 0
            if skew > 0:  # Downward pressure - prefer selling
                self.take_width = 1  # Lower threshold to take sell orders
            else:  # Upward pressure - prefer buying
                self.take_width = 1  # Lower threshold to take buy orders
        else:
            # Normal market conditions
            self.clear_width = 1
            self.take_width = 2
        
        # Standard market making
        buy_order_volume = 0
        sell_order_volume = 0
        take, buy_order_volume, sell_order_volume = self.take_orders(
            position, buy_order_volume, sell_order_volume)
        clear, buy_order_volume, sell_order_volume = self.clear_orders(
            position, buy_order_volume, sell_order_volume)
        make, _, _ = self.make_orders(
            position, buy_order_volume, sell_order_volume)
        
        all_orders = take + clear + make
        return {self.symbol: all_orders}

class MagnificentMacaronsStrategy(MarketMakeStrategy):
    def __init__(self, symbol: str, limit: int, order_depth: OrderDepth, trader_data):
        super().__init__(symbol, limit, order_depth, trader_data)
        self.limit = 75
        self.conversion_limit = 10
        self.storage_cost_rate = 0.1
        self.take_width = 2
        self.clear_width = 1
        self.join_edge = 1
        self.default_edge = 2

    def get_conversion_obs(self, state: TradingState):
        # Get the ConversionObservation for MAGNIFICENT_MACARONS
        return state.observations.conversionObservations.get(self.symbol)

    def calc_effective_prices(self, obs):
        # Returns (effective_buy_price, effective_sell_price)
        buy = obs.askPrice + obs.transportFees + obs.importTariff
        sell = obs.bidPrice - obs.transportFees - obs.exportTariff
        return buy, sell

    def is_profitable(self, buy, sell, holding_period=1):
        # Storage cost: 0.1 per unit per timestamp for net long
        storage_cost = self.storage_cost_rate * holding_period
        return (sell - buy - storage_cost) > 0

    def act(self, state: TradingState):
        obs = self.get_conversion_obs(state)
        position = state.position.get(self.symbol, 0)
        conversion_qty = 0
        orders = []
        min_profit = 0.5  

        if obs is not None:
            buy_price, sell_price = self.calc_effective_prices(obs)
            # f.write(f"Buy Price: {buy_price}, Sell Price: {sell_price}\n")
            order_depth = self.order_depth
            best_market_bid = max(order_depth.buy_orders.keys()) if order_depth.buy_orders else None
            best_market_ask = min(order_depth.sell_orders.keys()) if order_depth.sell_orders else None

            # Only trade if both conversion and market legs are available
            # 1. Buy from Pristine Cuisine (conversion), sell on market
            if (
                position < self.limit
                and best_market_bid is not None
                and order_depth.buy_orders[best_market_bid] > 0
            ):
                profit = best_market_bid - buy_price - self.storage_cost_rate
                if profit > min_profit:
                    max_qty = min(self.conversion_limit, self.limit - position, order_depth.buy_orders[best_market_bid])
                    if max_qty > 0:
                        conversion_qty = max_qty
                        orders.append(Order(self.symbol, best_market_bid, -max_qty))

            # 2. Buy from market, sell to Pristine Cuisine (conversion)
            elif (
                position > -self.limit
                and best_market_ask is not None
                and -order_depth.sell_orders[best_market_ask] > 0
            ):
                profit = sell_price - best_market_ask
                if profit > min_profit:
                    max_qty = min(self.conversion_limit, position + self.limit, -order_depth.sell_orders[best_market_ask])
                    if max_qty > 0:
                        conversion_qty = -max_qty
                        orders.append(Order(self.symbol, best_market_ask, max_qty))

        # Optionally, fallback to market making if no conversion trade
        if not orders:
            return super().act(state), 0

        # f.write(f"Conversion Orders: {conversion_qty}\n")
        return {self.symbol: orders}, conversion_qty
    
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
            "VOLCANIC_ROCK_VOUCHER_10500": 200,
            "MAGNIFICENT_MACARONS": 75
        }

    def run(self, state : TradingState) -> tuple[dict[Symbol, list[Order]], int , str]:
        trader_data = {}
        if state.traderData != None and state.traderData != "":
            trader_data = jsonpickle.decode(state.traderData)
        
        # Create mapping of symbols to their strategy constructors
        strategies = {symbol: constructor(symbol, self.limits[symbol], state.order_depths[symbol], trader_data) 
            for symbol, constructor in {
                "RAINFOREST_RESIN": ResinStrategy, 
                "KELP": KelpStrategy, 
                "SQUID_INK": SquidInkStrategy,
                "PICNIC_BASKET1": PicnicBasket1Strategy, 
                "PICNIC_BASKET2": PicnicBasket2Strategy,
                "VOLCANIC_ROCK": VolcanicRockSmileStrategy,
                "VOLCANIC_ROCK_VOUCHER_9500": Voucher9500Strategy,
                "VOLCANIC_ROCK_VOUCHER_9750": Voucher9750Strategy,
                "VOLCANIC_ROCK_VOUCHER_10000": Voucher10000Strategy,
                "VOLCANIC_ROCK_VOUCHER_10250": Voucher10250Strategy,
                "VOLCANIC_ROCK_VOUCHER_10500": Voucher10500Strategy,
                "MAGNIFICENT_MACARONS": MagnificentMacaronsStrategy,
            }.items() if symbol in state.order_depths}
        
        conversions = 0
        orders = {}
        
        for symbol, strategy in strategies.items():
            if symbol in state.order_depths:
                # Special handling for MagnificentMacaronsStrategy which returns (orders_dict, conversion_qty)
                if symbol == "MAGNIFICENT_MACARONS":
                    orders_dict, conversion_qty = strategy.act(state)
                    conversions = conversion_qty  # Only one conversion product, so this is safe
                else:
                    orders_dict = strategy.act(state)
                for symbol, order_list in orders_dict.items():
                    orders[symbol] = order_list
                    
        new_trader_data = jsonpickle.encode(trader_data)

        logger.flush(state, orders, conversions, new_trader_data)
        return orders, conversions, new_trader_data
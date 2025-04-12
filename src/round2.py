import jsonpickle
from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState
import numpy as np
import math
from collections import deque

# f = open("test.txt", "w")

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
        # Calculate available capacity to buy based on current position and order volume.
        buy_quantity = self.limit - (position + buy_order_volume)
        if buy_quantity > 0:
            orders.append(Order(self.symbol, round(bid), buy_quantity))
        
        # Calculate available capacity to sell.
        sell_quantity = self.limit + (position - sell_order_volume)
        if sell_quantity > 0:
            orders.append(Order(self.symbol, round(ask), -sell_quantity))  
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

        # Identify asks that are placed above fair value and bids below, but only those beyond the disregard threshold.
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

        # Determine the best ask above fair value, if any exist.
        best_ask_above_fair = min(asks_above_fair) if len(asks_above_fair) > 0 else None
        # Determine the best bid below fair value, if any exist.
        best_bid_below_fair = max(bids_below_fair) if len(bids_below_fair) > 0 else None

        # Initialize ask using the default edge.
        ask = round(self.fair_value + self.default_edge)
        if best_ask_above_fair != None:
            # If the best ask is close enough (within join_edge), join at that price.
            if abs(best_ask_above_fair - self.fair_value) <= self.join_edge:
                ask = best_ask_above_fair  # Join the level
            else:
                ask = best_ask_above_fair - 1  # Undercut by one unit (penny) to improve likelihood of execution

        # Similarly, initialize bid using default_edge.
        bid = round(self.fair_value - self.default_edge)
        if best_bid_below_fair != None:
            if abs(self.fair_value - best_bid_below_fair) <= self.join_edge:
                bid = best_bid_below_fair  # Join the level
            else:
                bid = best_bid_below_fair + 1  # Penny by raising the bid slightly

        # If managing position risk, adjust orders to favor reducing an extreme position.
        if self.manage_position:
            if position > self.soft_position_limit:
                ask -= 1  # Slightly lower the ask to encourage selling
            elif position < -1 * self.soft_position_limit:
                bid += 1  # Slightly raise the bid to encourage buying

        # Place the market-making orders based on the computed bid/ask prices.
        buy_order_volume, sell_order_volume = self.market_make(orders, bid, ask, position, buy_order_volume, sell_order_volume)

        return orders, buy_order_volume, sell_order_volume
    
    def act(self, state: TradingState):
        position = state.position.get(self.symbol, 0)
        take, buy_order_volume, sell_order_volume = self.take_orders(position, 0, 0)
        clear, buy_order_volume, sell_order_volume = self.clear_orders(position, buy_order_volume, sell_order_volume)
        make, _, _ = self.make_orders(position, buy_order_volume, sell_order_volume)
        return take + clear + make

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
        return orders + clear + make

# New strategies for Round 2 products
# class CroissantsStrategy(MarketMakeStrategy):
#     def __init__(self, symbol: str, limit: int, order_depth: OrderDepth, trader_data):
#         super().__init__(symbol, limit, order_depth, trader_data)
#         self.take_width = 1
#         self.clear_width = 0
#         self.disregard_edge = 1
#         self.join_edge = 1
#         self.default_edge = 2
#         self.soft_position_limit = 150
#         self.manage_position = True
#         self.fair_value = 0
        
#     def fair_price(self) -> float:
#         fair = None
#         if len(self.order_depth.sell_orders) != 0 and len(self.order_depth.buy_orders) != 0:
#             best_ask = min(self.order_depth.sell_orders.keys())
#             best_bid = max(self.order_depth.buy_orders.keys())
#             mid_price = (best_ask + best_bid) / 2
            
#             # Use simple midpoint as fair value
#             fair = mid_price
            
#             # Track price history for potential mean reversion
#             if "CROISSANTS_price_history" not in self.trader_data:
#                 self.trader_data["CROISSANTS_price_history"] = deque(maxlen=100)
            
#             self.trader_data["CROISSANTS_price_history"].append(mid_price)
            
#         return fair
    
#     def act(self, state: TradingState):
#         self.fair_value = self.fair_price()
#         return super().act(state)

# class JamsStrategy(MarketMakeStrategy):
#     def __init__(self, symbol: str, limit: int, order_depth: OrderDepth, trader_data):
#         super().__init__(symbol, limit, order_depth, trader_data)
#         self.take_width = 1
#         self.clear_width = 0
#         self.disregard_edge = 1
#         self.join_edge = 1
#         self.default_edge = 2
#         self.soft_position_limit = 200
#         self.manage_position = True
#         self.fair_value = 0
        
#     def fair_price(self) -> float:
#         fair = None
#         if len(self.order_depth.sell_orders) != 0 and len(self.order_depth.buy_orders) != 0:
#             best_ask = min(self.order_depth.sell_orders.keys())
#             best_bid = max(self.order_depth.buy_orders.keys())
#             mid_price = (best_ask + best_bid) / 2
            
#             # Use simple midpoint as fair value
#             fair = mid_price
            
#             # Track price history for potential mean reversion
#             if "JAMS_price_history" not in self.trader_data:
#                 self.trader_data["JAMS_price_history"] = deque(maxlen=100)
            
#             self.trader_data["JAMS_price_history"].append(mid_price)
            
#         return fair
    
#     def act(self, state: TradingState):
#         self.fair_value = self.fair_price()
#         return super().act(state)

# class DjembesStrategy(MarketMakeStrategy):
#     def __init__(self, symbol: str, limit: int, order_depth: OrderDepth, trader_data):
#         super().__init__(symbol, limit, order_depth, trader_data)
#         self.take_width = 2
#         self.clear_width = 1
#         self.disregard_edge = 1
#         self.join_edge = 1
#         self.default_edge = 3
#         self.soft_position_limit = 30
#         self.manage_position = True
#         self.fair_value = 0
        
#     def fair_price(self) -> float:
#         fair = None
#         if len(self.order_depth.sell_orders) != 0 and len(self.order_depth.buy_orders) != 0:
#             best_ask = min(self.order_depth.sell_orders.keys())
#             best_bid = max(self.order_depth.buy_orders.keys())
#             mid_price = (best_ask + best_bid) / 2
            
#             # Use simple midpoint as fair value
#             fair = mid_price
            
#             # Track price history for potential mean reversion
#             if "DJEMBES_price_history" not in self.trader_data:
#                 self.trader_data["DJEMBES_price_history"] = deque(maxlen=100)
            
#             self.trader_data["DJEMBES_price_history"].append(mid_price)
            
#         return fair
    
#     def act(self, state: TradingState):
#         self.fair_value = self.fair_price()
#         return super().act(state)

class PicnicBasketStrategy(MarketMakeStrategy):
    def __init__(self, symbol: str, limit: int, order_depth: OrderDepth, trader_data, components=None, quantities=None):
        super().__init__(symbol, limit, order_depth, trader_data)
        self.take_width = 1  # Reduced to be more aggressive
        self.clear_width = 0
        self.disregard_edge = 1
        self.join_edge = 1
        self.default_edge = 2  # Tighter spreads for more executions
        self.soft_position_limit = int(limit * 0.85)  # Use more capacity
        self.manage_position = True
        self.fair_value = 0
        self.components = components or []
        self.quantities = quantities or []
        self.arb_threshold = 3  # Lowered threshold for more aggressive arbitrage
        
    def fair_price(self, state: TradingState) -> float:
        fair = None
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
            
            if components_available:
                # Give more weight to the theoretical value for better arbitrage decisions
                alpha = 0.8  # Equal weight between market and theoretical value
                fair = alpha * basket_mid_price + (1 - alpha) * components_value
                
                # Store arbitrage info with more details for advanced strategies
                self.trader_data[f"{self.symbol}_components_value"] = components_value
                self.trader_data[f"{self.symbol}_market_value"] = basket_mid_price
                self.trader_data[f"{self.symbol}_arb_opportunity"] = components_value - basket_mid_price
                
                # Store more granular pricing metrics
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
                fair = basket_mid_price
                
            # Track price history
            if f"{self.symbol}_price_history" not in self.trader_data:
                self.trader_data[f"{self.symbol}_price_history"] = deque(maxlen=100)
            
            self.trader_data[f"{self.symbol}_price_history"].append(basket_mid_price)
            
        return fair
    
    def act(self, state: TradingState):
        self.fair_value = self.fair_price(state)
        position = state.position.get(self.symbol, 0)
        orders = []
        buy_order_volume = 0
        sell_order_volume = 0
        
        # Check for arbitrage opportunities
        arb_opportunity = self.trader_data.get(f"{self.symbol}_arb_opportunity", 0)
        components_value = self.trader_data.get(f"{self.symbol}_components_value", 0)
        market_value = self.trader_data.get(f"{self.symbol}_market_value", 0)
        
        # More aggressive arbitrage with lower threshold
        if abs(arb_opportunity) > self.arb_threshold:
            if components_value > market_value:
                # Basket is underpriced compared to components - buy basket aggressively
                if len(self.order_depth.sell_orders) > 0:
                    best_ask = min(self.order_depth.sell_orders.keys())
                    best_ask_volume = -self.order_depth.sell_orders[best_ask]
                    available_capacity = self.limit - position
                    # Take larger quantity for profitable arbitrage
                    quantity = min(best_ask_volume, available_capacity)
                    
                    if quantity > 0:
                        orders.append(Order(self.symbol, best_ask, quantity))
                        buy_order_volume += quantity
                        
                        # Sweep more levels if still profitable
                        remaining_capacity = available_capacity - quantity
                        if remaining_capacity > 0:
                            ask_prices = sorted(self.order_depth.sell_orders.keys())
                            for ask_price in ask_prices[1:]:  # Skip the best ask we already took
                                if ask_price <= components_value - self.arb_threshold:
                                    ask_volume = -self.order_depth.sell_orders[ask_price]
                                    sweep_quantity = min(ask_volume, remaining_capacity)
                                    if sweep_quantity > 0:
                                        orders.append(Order(self.symbol, ask_price, sweep_quantity))
                                        buy_order_volume += sweep_quantity
                                        remaining_capacity -= sweep_quantity
            else:
                # Basket is overpriced compared to components - sell basket aggressively
                if len(self.order_depth.buy_orders) > 0:
                    best_bid = max(self.order_depth.buy_orders.keys())
                    best_bid_volume = self.order_depth.buy_orders[best_bid]
                    available_capacity = self.limit + position
                    # Take larger quantity for profitable arbitrage
                    quantity = min(best_bid_volume, available_capacity)
                    
                    if quantity > 0:
                        orders.append(Order(self.symbol, best_bid, -quantity))
                        sell_order_volume += quantity
                        
                        # Sweep more levels if still profitable
                        remaining_capacity = available_capacity - quantity
                        if remaining_capacity > 0:
                            bid_prices = sorted(self.order_depth.buy_orders.keys(), reverse=True)
                            for bid_price in bid_prices[1:]:  # Skip the best bid we already took
                                if bid_price >= components_value + self.arb_threshold:
                                    bid_volume = self.order_depth.buy_orders[bid_price]
                                    sweep_quantity = min(bid_volume, remaining_capacity)
                                    if sweep_quantity > 0:
                                        orders.append(Order(self.symbol, bid_price, -sweep_quantity))
                                        sell_order_volume += sweep_quantity
                                        remaining_capacity -= sweep_quantity
        
        # For regular market making, use tighter spreads on baskets to improve execution odds
        take, buy_order_volume, sell_order_volume = self.take_orders(position, buy_order_volume, sell_order_volume)
        clear, buy_order_volume, sell_order_volume = self.clear_orders(position, buy_order_volume, sell_order_volume)
        make, _, _ = self.make_orders(position, buy_order_volume, sell_order_volume)
        
        return orders + take + clear + make

class PicnicBasket1Strategy(PicnicBasketStrategy):
    def __init__(self, symbol: str, limit: int, order_depth: OrderDepth, trader_data):
        components = ["CROISSANTS", "JAMS", "DJEMBES"]
        quantities = [6, 3, 1]
        super().__init__(symbol, limit, order_depth, trader_data, components, quantities)
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
                
                # Calculate custom statistics (mean and standard deviation)
                deviation_history = list(self.trader_data[f"{self.symbol}_pct_deviation_history"])
                if len(deviation_history) > 5:  # Need enough data for meaningful statistics
                    # Custom mean calculation
                    mean_deviation = sum(deviation_history) / len(deviation_history)
                    
                    # Custom standard deviation calculation
                    squared_diffs = [(x - mean_deviation) ** 2 for x in deviation_history]
                    variance = sum(squared_diffs) / len(squared_diffs)
                    std_deviation = max(0.0001, variance ** 0.5)  # sqrt of variance, with minimum value
                    
                    # Calculate z-score (how many standard deviations from mean)
                    z_score = (pct_deviation - mean_deviation) / std_deviation
                    self.trader_data[f"{self.symbol}_z_score"] = z_score
                    
                    # Use z-score for more consistent threshold adjustments
                    if abs(z_score) > 2.0:  # Very significant deviation (>2 standard deviations)
                        # For large statistical deviations, increase reversion strength
                        self.component_reversion_strength = 0.5
                        self.arb_threshold = 0.001  # Almost any deviation is worth trading
                        self.take_width = 0  # Extremely aggressive, take at fair value
                    elif abs(z_score) > 1.0:  # Moderate deviation (1-2 standard deviations)
                        self.component_reversion_strength = 0.4
                        self.arb_threshold = 0.002  # Small percentage threshold
                        self.take_width = 1
                    else:  # Small deviation (<1 standard deviation)
                        # For small deviations, normal settings
                        self.component_reversion_strength = 0.3
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
        clear, buy_order_volume, sell_order_volume = self.clear_orders(position, buy_order_volume, sell_order_volume)
        make, _, _ = self.make_orders(position, buy_order_volume, sell_order_volume)
        
        return orders + take + clear + make
        
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
            "PICNIC_BASKET2": 100
        }



    def run(self, state : TradingState) -> tuple[dict[Symbol, list[Order]], int , str]:
        trader_data = {}
        if state.traderData != None and state.traderData != "":
            trader_data = jsonpickle.decode(state.traderData)
        strategies = {symbol: constructor(symbol, self.limits[symbol], state.order_depths[symbol], trader_data) for symbol, constructor in {
            "RAINFOREST_RESIN" : ResinStrategy, "KELP" : KelpStrategy, "SQUID_INK" : SquidInkStrategy,
            "PICNIC_BASKET1": PicnicBasket1Strategy}.items()}
        conversions = 0
        orders = {}
        for symbol, strategy in strategies.items():
            if symbol in state.order_depths:
                orders[symbol] = strategy.act(state)
        new_trader_data = jsonpickle.encode(trader_data)
        return orders, conversions, new_trader_data
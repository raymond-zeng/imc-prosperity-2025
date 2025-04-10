import jsonpickle
from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState
import numpy as np
import math
from collections import deque

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
class CroissantStrategy(MarketMakeStrategy):
    def __init__(self, symbol: str, limit: int, order_depth: OrderDepth, trader_data):
        super().__init__(symbol, limit, order_depth, trader_data)
        self.take_width = 1
        self.clear_width = 0
        self.disregard_edge = 1
        self.join_edge = 1
        self.default_edge = 2
        self.soft_position_limit = 150
        self.manage_position = True
        self.fair_value = 0
        
    def fair_price(self) -> float:
        fair = None
        if len(self.order_depth.sell_orders) != 0 and len(self.order_depth.buy_orders) != 0:
            best_ask = min(self.order_depth.sell_orders.keys())
            best_bid = max(self.order_depth.buy_orders.keys())
            mid_price = (best_ask + best_bid) / 2
            
            # Use simple midpoint as fair value
            fair = mid_price
            
            # Track price history for potential mean reversion
            if "CROISSANT_price_history" not in self.trader_data:
                self.trader_data["CROISSANT_price_history"] = deque(maxlen=100)
            
            self.trader_data["CROISSANT_price_history"].append(mid_price)
            
        return fair
    
    def act(self, state: TradingState):
        self.fair_value = self.fair_price()
        return super().act(state)

class JamStrategy(MarketMakeStrategy):
    def __init__(self, symbol: str, limit: int, order_depth: OrderDepth, trader_data):
        super().__init__(symbol, limit, order_depth, trader_data)
        self.take_width = 1
        self.clear_width = 0
        self.disregard_edge = 1
        self.join_edge = 1
        self.default_edge = 2
        self.soft_position_limit = 200
        self.manage_position = True
        self.fair_value = 0
        
    def fair_price(self) -> float:
        fair = None
        if len(self.order_depth.sell_orders) != 0 and len(self.order_depth.buy_orders) != 0:
            best_ask = min(self.order_depth.sell_orders.keys())
            best_bid = max(self.order_depth.buy_orders.keys())
            mid_price = (best_ask + best_bid) / 2
            
            # Use simple midpoint as fair value
            fair = mid_price
            
            # Track price history for potential mean reversion
            if "JAM_price_history" not in self.trader_data:
                self.trader_data["JAM_price_history"] = deque(maxlen=100)
            
            self.trader_data["JAM_price_history"].append(mid_price)
            
        return fair
    
    def act(self, state: TradingState):
        self.fair_value = self.fair_price()
        return super().act(state)

class DjembeStrategy(MarketMakeStrategy):
    def __init__(self, symbol: str, limit: int, order_depth: OrderDepth, trader_data):
        super().__init__(symbol, limit, order_depth, trader_data)
        self.take_width = 2
        self.clear_width = 1
        self.disregard_edge = 1
        self.join_edge = 1
        self.default_edge = 3
        self.soft_position_limit = 30
        self.manage_position = True
        self.fair_value = 0
        
    def fair_price(self) -> float:
        fair = None
        if len(self.order_depth.sell_orders) != 0 and len(self.order_depth.buy_orders) != 0:
            best_ask = min(self.order_depth.sell_orders.keys())
            best_bid = max(self.order_depth.buy_orders.keys())
            mid_price = (best_ask + best_bid) / 2
            
            # Use simple midpoint as fair value
            fair = mid_price
            
            # Track price history for potential mean reversion
            if "DJEMBE_price_history" not in self.trader_data:
                self.trader_data["DJEMBE_price_history"] = deque(maxlen=100)
            
            self.trader_data["DJEMBE_price_history"].append(mid_price)
            
        return fair
    
    def act(self, state: TradingState):
        self.fair_value = self.fair_price()
        return super().act(state)

class PicnicBasketStrategy(MarketMakeStrategy):
    def __init__(self, symbol: str, limit: int, order_depth: OrderDepth, trader_data, components=None, quantities=None):
        super().__init__(symbol, limit, order_depth, trader_data)
        self.take_width = 2
        self.clear_width = 1
        self.disregard_edge = 1
        self.join_edge = 1
        self.default_edge = 3
        self.soft_position_limit = 30
        self.manage_position = True
        self.fair_value = 0
        self.components = components or []  # The component products in the basket
        self.quantities = quantities or []  # Quantity of each component in the basket
        
    def fair_price(self, state: TradingState) -> float:
        # First try to determine fair value from the order book
        fair = None
        if len(self.order_depth.sell_orders) != 0 and len(self.order_depth.buy_orders) != 0:
            best_ask = min(self.order_depth.sell_orders.keys())
            best_bid = max(self.order_depth.buy_orders.keys())
            basket_mid_price = (best_ask + best_bid) / 2
            
            # If we have components defined, calculate the theoretical value
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
            
            # If all components have pricing, use their value for arbitrage
            if components_available:
                # Use a weighted combination of order book and components
                alpha = 0.7  # Weight for the order book price
                fair = alpha * basket_mid_price + (1 - alpha) * components_value
                
                # Store arbitrage info
                self.trader_data[f"{self.symbol}_components_value"] = components_value
                self.trader_data[f"{self.symbol}_market_value"] = basket_mid_price
                self.trader_data[f"{self.symbol}_arb_opportunity"] = components_value - basket_mid_price
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
        
        # If there's a significant arbitrage opportunity
        if abs(arb_opportunity) > 5:  # Threshold for action
            if components_value > market_value:
                # Basket is underpriced compared to components - buy basket
                if len(self.order_depth.sell_orders) > 0:
                    best_ask = min(self.order_depth.sell_orders.keys())
                    best_ask_volume = -self.order_depth.sell_orders[best_ask]
                    available_capacity = self.limit - position
                    quantity = min(best_ask_volume, available_capacity)
                    
                    if quantity > 0:
                        orders.append(Order(self.symbol, best_ask, quantity))
                        buy_order_volume += quantity
            else:
                # Basket is overpriced compared to components - sell basket
                if len(self.order_depth.buy_orders) > 0:
                    best_bid = max(self.order_depth.buy_orders.keys())
                    best_bid_volume = self.order_depth.buy_orders[best_bid]
                    available_capacity = self.limit + position
                    quantity = min(best_bid_volume, available_capacity)
                    
                    if quantity > 0:
                        orders.append(Order(self.symbol, best_bid, -quantity))
                        sell_order_volume += quantity
        
        # Continue with regular market making
        take, buy_order_volume, sell_order_volume = self.take_orders(position, buy_order_volume, sell_order_volume)
        clear, buy_order_volume, sell_order_volume = self.clear_orders(position, buy_order_volume, sell_order_volume)
        make, _, _ = self.make_orders(position, buy_order_volume, sell_order_volume)
        
        return orders + take + clear + make

class PicnicBasket1Strategy(PicnicBasketStrategy):
    def __init__(self, symbol: str, limit: int, order_depth: OrderDepth, trader_data):
        components = ["CROISSANT", "JAM", "DJEMBE"]
        quantities = [6, 3, 1]
        super().__init__(symbol, limit, order_depth, trader_data, components, quantities)

class PicnicBasket2Strategy(PicnicBasketStrategy):
    def __init__(self, symbol: str, limit: int, order_depth: OrderDepth, trader_data):
        components = ["CROISSANT", "JAM"]
        quantities = [4, 2]
        super().__init__(symbol, limit, order_depth, trader_data, components, quantities)

class Trader:

    def __init__(self):
        self.limits = {
            "RAINFOREST_RESIN": 50,
            "KELP": 50,
            "SQUID_INK": 50,
            "CROISSANT": 250,
            "JAM": 350,
            "DJEMBE": 60,
            "PICNIC_BASKET1": 60,
            "PICNIC_BASKET2": 100
        }
        self.conversions_available = {
            "PICNIC_BASKET1": {"components": {"CROISSANT": 6, "JAM": 3, "DJEMBE": 1}, "result": 1},
            "PICNIC_BASKET2": {"components": {"CROISSANT": 4, "JAM": 2}, "result": 1}
        }
        self.conversion_cooldown = 0

    def check_conversion_possibility(self, state: TradingState, symbol: str, direction: str) -> int:
        """
        Check if conversion is possible for a given symbol and direction.
        Returns the maximum number of conversions possible.
        """
        if symbol not in self.conversions_available or self.conversion_cooldown > 0:
            return 0

        conversion_info = self.conversions_available[symbol]
        components = conversion_info["components"]
        result_qty = conversion_info["result"]
        
        if direction == "TO_BASKET":
            # Check if we have enough components to make baskets
            max_conversions = float('inf')
            for component, qty_needed in components.items():
                position = state.position.get(component, 0)
                if position < qty_needed:
                    return 0  # Not enough of this component
                max_conversions = min(max_conversions, position // qty_needed)
            
            # Check if we have room for the resulting baskets
            basket_position = state.position.get(symbol, 0)
            basket_limit = self.limits[symbol]
            basket_capacity = basket_limit - basket_position
            max_conversions = min(max_conversions, basket_capacity // result_qty)
            
            return max_conversions
            
        elif direction == "FROM_BASKET":
            # Check if we have enough baskets to break down
            basket_position = state.position.get(symbol, 0)
            if basket_position <= 0:
                return 0
            
            max_conversions = basket_position // result_qty
            
            # Check if we have room for the resulting components
            for component, qty_result in components.items():
                position = state.position.get(component, 0)
                component_limit = self.limits[component]
                component_capacity = component_limit - position
                if component_capacity < qty_result:
                    return 0  # Not enough capacity for this component
                max_conversions = min(max_conversions, component_capacity // qty_result)
            
            return max_conversions
        
        return 0

    def execute_conversion(self, state: TradingState, basket: str, direction: str) -> int:
        """
        Execute the conversion and return the number of conversions performed.
        """
        if self.conversion_cooldown > 0:
            self.conversion_cooldown -= 1
            return 0
            
        max_conversions = self.check_conversion_possibility(state, basket, direction)
        if max_conversions <= 0:
            return 0
            
        # Set cooldown to prevent too frequent conversions
        self.conversion_cooldown = 3
        return max_conversions

    def check_basket_arbitrage(self, state: TradingState, basket: str) -> tuple[str, float]:
        """
        Check if there's an arbitrage opportunity with this basket.
        Returns the direction and the price difference.
        """
        if basket not in self.conversions_available:
            return None, 0
            
        conversion_info = self.conversions_available[basket]
        components = conversion_info["components"]
        
        # Calculate the cost of the components
        component_cost = 0
        for component, qty in components.items():
            if component not in state.order_depths:
                return None, 0
                
            component_depth = state.order_depths[component]
            if not component_depth.sell_orders or not component_depth.buy_orders:
                return None, 0
                
            # Use mid price for component valuation
            component_mid = (min(component_depth.sell_orders.keys()) + 
                            max(component_depth.buy_orders.keys())) / 2
            component_cost += component_mid * qty
        
        # Get the basket price
        if basket not in state.order_depths:
            return None, 0
            
        basket_depth = state.order_depths[basket]
        if not basket_depth.sell_orders or not basket_depth.buy_orders:
            return None, 0
            
        basket_mid = (min(basket_depth.sell_orders.keys()) + 
                    max(basket_depth.buy_orders.keys())) / 2
        
        # Calculate the price difference
        diff = basket_mid - component_cost
        
        if diff > 5:  # Threshold for arbitrage
            return "FROM_BASKET", diff  # Sell basket, buy components
        elif diff < -5:  # Threshold for arbitrage
            return "TO_BASKET", -diff   # Buy basket, sell components
            
        return None, 0

    def run(self, state: TradingState) -> tuple[dict[Symbol, list[Order]], int, str]:
        trader_data = {}
        if state.traderData != None and state.traderData != "":
            trader_data = jsonpickle.decode(state.traderData)
            
        # Initialize strategies for all products
        strategies = {
            "RAINFOREST_RESIN": ResinStrategy,
            "KELP": KelpStrategy,
            "SQUID_INK": SquidInkStrategy,
            "CROISSANT": CroissantStrategy,
            "JAM": JamStrategy,
            "DJEMBE": DjembeStrategy,
            "PICNIC_BASKET1": PicnicBasket1Strategy,
            "PICNIC_BASKET2": PicnicBasket2Strategy
        }
        
        # Create strategy instances for symbols with order depths
        active_strategies = {
            symbol: constructor(symbol, self.limits[symbol], state.order_depths[symbol], trader_data)
            for symbol, constructor in strategies.items()
            if symbol in state.order_depths
        }
        
        # Check for arbitrage opportunities with baskets
        conversions = 0
        for basket in ["PICNIC_BASKET1", "PICNIC_BASKET2"]:
            direction, diff = self.check_basket_arbitrage(state, basket)
            if direction:
                max_conversions = self.execute_conversion(state, basket, direction)
                conversions += max_conversions
                
        # Execute trading strategies
        orders = {}
        for symbol, strategy in active_strategies.items():
            orders[symbol] = strategy.act(state)
            
        new_trader_data = jsonpickle.encode(trader_data)
        return orders, conversions, new_trader_data
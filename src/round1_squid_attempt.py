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
        self.reversion_beta = -0.229
        self.join_edge = 0
        self.default_edge = 1
        self.fair_value = 0
        self.trader_data = trader_data

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
        # Keep existing parameters
        self.prevent_adverse = True
        self.adverse_volume = 5
        self.trader_data = trader_data
        self.take_width = 2
        self.clear_width = 1
        self.disregard_edge = 1
        self.join_edge = 0
        self.default_edge = 2
        self.soft_position_limit = 20  # Reduced from 25 for more conservative approach
        self.manage_position = True
        self.fair_value = 1980
        self.mean_price = 1980
        self.reversion_strength = 0.5
        self.history_length = 50
        self.expected_amplitude = 30
        self.trend_threshold = 5
        
        # Add these new parameters for volatility protection
        self.downside_threshold = 1920  # Price threshold for increased caution
        self.extreme_downside = 1850   # Price threshold for extreme caution
        self.position_scaling = True   # Enable dynamic position sizing
        self.max_position_at_bottom = 35  # Maximum position when price is very low
        self.max_position_at_top = 5   # Maximum position when price is very high
        self.volatility_adjustment = True  # Enable volatility-based adjustments

    def fair_price(self) -> float:
        # If we don't have enough market data, return the mean price
        if len(self.order_depth.sell_orders) == 0 or len(self.order_depth.buy_orders) == 0:
            return self.mean_price
            
        # Get current best prices from the order book
        best_ask = min(self.order_depth.sell_orders.keys())
        best_bid = max(self.order_depth.buy_orders.keys())
        current_mid = (best_ask + best_bid) / 2
        
        # Initialize price history if needed
        if "SQUID_price_history" not in self.trader_data:
            self.trader_data["SQUID_price_history"] = deque(maxlen=self.history_length)
            self.trader_data["SQUID_direction_changes"] = 0
            self.trader_data["SQUID_last_direction"] = 0  # 0=neutral, 1=up, -1=down
            
        # Update price history
        self.trader_data["SQUID_price_history"].append(current_mid)
        price_history = self.trader_data["SQUID_price_history"]
        
        # Calculate basic metrics
        deviation = (current_mid - self.mean_price) / self.mean_price
        
        # Initialize current_direction with a default value
        current_direction = 0
        
        # Detect direction changes (wave turning points)
        if len(price_history) > 2:
            # Determine current direction
            last_price = price_history[-2]
            current_direction = 1 if current_mid > last_price else -1 if current_mid < last_price else 0
            
            # Check for direction change
            if "SQUID_last_direction" in self.trader_data:
                last_direction = self.trader_data["SQUID_last_direction"]
                if last_direction != 0 and current_direction != 0 and last_direction != current_direction:
                    self.trader_data["SQUID_direction_changes"] = self.trader_data.get("SQUID_direction_changes", 0) + 1
            
            self.trader_data["SQUID_last_direction"] = current_direction
        
        # Calculate recent trend strength (count consecutive moves in same direction)
        trend_strength = 0
        if len(price_history) > 2:
            trend_direction = 1 if price_history[-1] > price_history[-2] else -1 if price_history[-1] < price_history[-2] else 0
            for i in range(len(price_history)-2, 0, -1):
                move_direction = 1 if price_history[i] > price_history[i-1] else -1 if price_history[i] < price_history[i-1] else 0
                if move_direction == trend_direction:
                    trend_strength += 1
                else:
                    break
        
        # Calculate distance from extremes based on expected amplitude
        extreme_distance = 0
        if abs(deviation) > 0:
            # How close are we to expected max/min in the sinusoidal pattern?
            # 1.0 = at the mean, 0.0 = at the expected extreme
            normalized_pos = max(0, 1.0 - (abs(deviation) * self.mean_price / self.expected_amplitude))
            extreme_distance = normalized_pos
        
        # Enhanced mean reversion logic for sinusoidal patterns
        reversion_adjustment = 0
        
        # 1. Basic mean reversion component
        reversion_adjustment -= deviation * self.reversion_strength
        
        # 2. Enhance reversion when we reach expected extremes
        if abs(deviation) > 0.01:  # At least 1% away from mean
            # Strong reversion near extremes
            if extreme_distance < 0.3:  # Near expected extreme
                # Stronger reversion at extremes (opposite direction to deviation)
                reversion_adjustment -= (deviation / abs(deviation)) * 0.02
            
            # 3. Trend-following component when price is moving toward mean
            if (deviation > 0 and current_direction < 0) or (deviation < 0 and current_direction > 0):
                # Price is already moving toward mean - add momentum component
                if trend_strength > self.trend_threshold:
                    # Strong trend toward mean - enhance it
                    reversion_adjustment -= deviation * 0.2
        
        # Calculate fair value with all adjustments
        fair_value = current_mid * (1 + reversion_adjustment)
        
        # Store data for next round
        self.trader_data["SQUID_last_price"] = current_mid
        self.trader_data["SQUID_fair_value"] = fair_value    

        if "SQUID_lowest_price" not in self.trader_data:
            self.trader_data["SQUID_lowest_price"] = current_mid
            self.trader_data["SQUID_highest_price"] = current_mid
        else:
            self.trader_data["SQUID_lowest_price"] = min(self.trader_data["SQUID_lowest_price"], current_mid)
            self.trader_data["SQUID_highest_price"] = max(self.trader_data["SQUID_highest_price"], current_mid)
        
        # Calculate recent volatility
        if len(price_history) >= 10:
            # Calculate rolling volatility as standard deviation of returns
            recent_prices = list(price_history)[-10:]
            returns = [(recent_prices[i] / recent_prices[i-1]) - 1 for i in range(1, len(recent_prices))]
            volatility = np.std(returns) * 100  # Volatility as percentage
            
            # Store volatility for position sizing
            self.trader_data["SQUID_volatility"] = volatility
            
            # Adjust parameters based on volatility
            if self.volatility_adjustment and volatility > 0.5:  # High volatility threshold (0.5%)
                # In high volatility, widen spreads and reduce position size
                self.default_edge = min(5, max(2, int(volatility * 3)))
                high_vol_factor = min(0.8, max(0.4, 1 - (volatility - 0.5)))  # 0.4 to 0.8 scaling factor
                self.soft_position_limit = max(5, int(20 * high_vol_factor))
        
        return fair_value

    def act(self, state: TradingState):
        # Update fair value
        self.fair_value = self.fair_price()
        
        # Get current position and current price
        position = state.position.get(self.symbol, 0)
        current_price = self.trader_data.get("SQUID_last_price", self.mean_price)  # Get current price from trader data
        
        original_take_width = self.take_width  # Store original value to restore later
        original_default_edge = self.default_edge  # Store original value
        
        if "SQUID_max_profit" not in self.trader_data:
            self.trader_data["SQUID_max_profit"] = 0
            self.trader_data["SQUID_current_profit"] = 0

        # Simple profit tracking (rough approximation)
        if "SQUID_last_position" in self.trader_data and "SQUID_last_price" in self.trader_data:
            last_position = self.trader_data["SQUID_last_position"]
            last_price = self.trader_data["SQUID_last_price"]
            price_change = current_price - last_price
            position_profit = last_position * price_change
            
            # Update current profit
            self.trader_data["SQUID_current_profit"] += position_profit
            
            # Track maximum profit
            if self.trader_data["SQUID_current_profit"] > self.trader_data["SQUID_max_profit"]:
                self.trader_data["SQUID_max_profit"] = self.trader_data["SQUID_current_profit"]
            
            # Calculate drawdown
            max_profit = self.trader_data["SQUID_max_profit"]
            current_profit = self.trader_data["SQUID_current_profit"]
            if max_profit > 0:
                drawdown = (max_profit - current_profit) / max_profit
                
                # If drawdown exceeds threshold, reduce position size dramatically
                if drawdown > 0.15:  # 15% drawdown threshold
                    self.soft_position_limit = max(3, int(self.soft_position_limit * 0.5))
                    if position > 0 and current_price < self.mean_price:
                        # Losing money on longs in downtrend - reduce aggressively
                        self.take_width = 0  # Take any opportunity to sell
                    elif position < 0 and current_price > self.mean_price:
                        # Losing money on shorts in uptrend - reduce aggressively
                        self.take_width = 0  # Take any opportunity to buy

        # Store current position for next iteration
        self.trader_data["SQUID_last_position"] = position
        # Advanced position management with asymmetric risk controls
        if "SQUID_last_price" in self.trader_data and "SQUID_price_history" in self.trader_data:
            current_price = self.trader_data["SQUID_last_price"]
            deviation = (current_price - self.mean_price) / self.mean_price
            price_history = self.trader_data["SQUID_price_history"]
            
            # Detect extreme price scenarios
            extreme_downside_detected = current_price < self.extreme_downside
            downside_warning = current_price < self.downside_threshold
            
            # Dynamic position management based on volatility and price levels
            if self.position_scaling:
                if current_price < self.mean_price:
                    # Below mean - more aggressive buying, careful selling
                    # Calculate how far we are from the lowest price (normalized 0-1)
                    if "SQUID_lowest_price" in self.trader_data:
                        lowest = self.trader_data["SQUID_lowest_price"]
                        price_range = self.mean_price - lowest
                        if price_range > 0:
                            position_in_range = (current_price - lowest) / price_range
                            # Scale position limit based on where we are in the range
                            # Lower in range = higher position limit for buying
                            self.soft_position_limit = int(self.max_position_at_bottom - 
                                                        (position_in_range * (self.max_position_at_bottom - 20)))
                            
                            # When approaching historical lows, be very aggressive buying
                            if current_price - lowest < 20:
                                self.take_width = 4  # Very aggressive buying
                                
                    # Stop loss for short positions in downtrends
                    if position < -5 and downside_warning:
                        # Cut short positions when price gets too low
                        self.take_width = 4  # Aggressive buying to cover shorts
                        # Prohibit new shorts
                        if position < 0:
                            self.soft_position_limit = 40  # Allow buying but limit short positions
                else:
                    # Above mean - more aggressive selling, careful buying
                    if "SQUID_highest_price" in self.trader_data:
                        highest = self.trader_data["SQUID_highest_price"]
                        price_range = highest - self.mean_price
                        if price_range > 0:
                            position_in_range = (highest - current_price) / price_range
                            # Scale position limit based on where we are in the range
                            # Higher in range = lower position limit for buying
                            self.soft_position_limit = int(20 - 
                                                        (position_in_range * (20 - self.max_position_at_top)))
                    
                    # Stop loss for long positions in uptrends
                    if position > 5 and current_price > self.mean_price + self.expected_amplitude:
                        # Cut long positions when price gets too high
                        self.take_width = 4  # Aggressive selling
            
            # Volatility-based adjustments
            if self.volatility_adjustment and "SQUID_volatility" in self.trader_data:
                volatility = self.trader_data["SQUID_volatility"]
                
                # In high volatility, be more conservative
                if volatility > 0.8:  # Very high volatility
                    self.soft_position_limit = max(5, min(15, self.soft_position_limit))
                    self.default_edge = max(3, self.default_edge)  # Wider spreads
                    
                    # If we have a large position during high volatility, reduce it
                    if abs(position) > 15:
                        self.take_width = 1  # More aggressive about taking favorable orders
            
            # Profit locking mechanism
            if "SQUID_last_fair_value" in self.trader_data:
                last_fair = self.trader_data["SQUID_last_fair_value"]
                
                # If we have a profitable position, consider reducing it
                if (position > 10 and self.fair_value > last_fair + 5) or \
                (position < -10 and self.fair_value < last_fair - 5):
                    self.take_width = 1  # Be more aggressive about taking profit
                    
            # Store fair value for next comparison
            self.trader_data["SQUID_last_fair_value"] = self.fair_value
        
        # Execute strategy with adjusted parameters
        result = super().act(state)
        
        # Restore original parameters
        self.take_width = original_take_width
        self.default_edge = original_default_edge
        
        return result
    
class Trader:

    def __init__(self):
        self.limits = {
            "RAINFOREST_RESIN" : 50,
            "KELP" : 50,
            "SQUID_INK" : 50
        }

    def run(self, state : TradingState) -> tuple[dict[Symbol, list[Order]], int , str]:
        trader_data = {}
        if state.traderData != None and state.traderData != "":
            trader_data = jsonpickle.decode(state.traderData)
        strategies = {symbol: constructor(symbol, self.limits[symbol], state.order_depths[symbol], trader_data) for symbol, constructor in {
            "RAINFOREST_RESIN" : ResinStrategy, "KELP" : KelpStrategy, "SQUID_INK" : SquidInkStrategy
        }.items()}
        conversions = 0
        orders = {}
        for symbol, strategy in strategies.items():
            if symbol in state.order_depths:
                orders[symbol] = strategy.act(state)
        new_trader_data = jsonpickle.encode(trader_data)
        return orders, conversions, new_trader_data
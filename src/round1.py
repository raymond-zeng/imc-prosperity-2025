# Import essential types and models for trading.
from datamodel import OrderDepth, UserId, TradingState, Order
from typing import List
import string
import jsonpickle  # for serializing/deserializing trader-specific data across rounds
import numpy as np
import math

# Define products handled by the trader.
class Product:
    RAINFOREST_RESIN = "RAINFOREST_RESIN"  # Product identifier for RAINFOREST_RESIN
    KELP = "KELP"  # Product identifier for KELP

# Configuration parameters for each product.
# Each product has its own set of parameters:
#   - fair_value: the baseline price to compare market orders
#   - take_width: the price deviation threshold to trigger order taking
#   - clear_width: width used when clearing positions
#   - disregard_edge: used when deciding to ignore orders near the fair value (for order joining/pennying)
#   - join_edge: edge threshold to join orders rather than undercutting them
#   - default_edge: fallback edge when no other edge criteria are met
#   - soft_position_limit: limit used to manage inventory positions
PARAMS = {
    Product.RAINFOREST_RESIN: {
        "fair_value": 10000,
        "take_width": 1,
        "clear_width": 0,
        "disregard_edge": 1,  # Orders inside this range from fair value are disregarded for joining or undercutting
        "join_edge": 2,       # Orders within this edge will be joined directly
        "default_edge": 4,    # Default offset from the fair value if no optimal edge is available
        "soft_position_limit": 10,  # Soft limit for managing position size before adjustments
    },
    # Product.KELP: {
    #     "take_width": 1,
    #     "clear_width": 0,
    #     "prevent_adverse": True,  # Flag to prevent taking orders that could lead to adverse price movements
    #     "adverse_volume": 15,     # Volume threshold to define adverse market conditions
    #     "reversion_beta": -0.229, # Factor used to adjust fair value predictions based on past price deviations
    #     "disregard_edge": 1,
    #     "join_edge": 0,
    #     "default_edge": 1,
    # },
}

# Trader class encapsulating trading strategies for different products.
class Trader:
    def __init__(self, params=None):
        # Initialize trader parameters; default to global PARAMS if none provided.
        if params is None:
            params = PARAMS
        self.params = params

        # Position limits for each product to ensure risk management.
        self.LIMIT = {Product.RAINFOREST_RESIN: 20, Product.KELP: 20}

    # This method attempts to "take" the best market orders (i.e., orders already in the book)
    # if they satisfy certain price thresholds relative to the fair value.
    def take_best_orders(
        self,
        product: str,
        fair_value: int,
        take_width: float,
        orders: List[Order],
        order_depth: OrderDepth,
        position: int,
        buy_order_volume: int,
        sell_order_volume: int,
        prevent_adverse: bool = False,
        adverse_volume: int = 0,
    ) -> (int, int):
        # Retrieve product-specific position limit
        position_limit = self.LIMIT[product]

        # Check if any sell orders exist: these are offers to sell at various prices.
        if len(order_depth.sell_orders) != 0:
            # Find the best ask (lowest sell price)
            best_ask = min(order_depth.sell_orders.keys())
            # Multiply by -1 because the sell order volumes are stored as negative values
            best_ask_amount = -1 * order_depth.sell_orders[best_ask]

            # Only consider the order if it does not trigger an adverse condition
            if not prevent_adverse or abs(best_ask_amount) <= adverse_volume:
                # If the best ask is below the threshold (fair value minus take_width), it's an attractive buy.
                if best_ask <= fair_value - take_width:
                    # Calculate how many units can be bought without exceeding position limits.
                    quantity = min(best_ask_amount, position_limit - position)
                    if quantity > 0:
                        # Place a buy order at the best ask price.
                        orders.append(Order(product, best_ask, quantity))
                        buy_order_volume += quantity
                        # Update the order depth to reflect the filled volume.
                        order_depth.sell_orders[best_ask] += quantity
                        if order_depth.sell_orders[best_ask] == 0:
                            del order_depth.sell_orders[best_ask]

        # Repeat the process for buy orders (bids) from the order book.
        if len(order_depth.buy_orders) != 0:
            # Find the best bid (highest buy price)
            best_bid = max(order_depth.buy_orders.keys())
            best_bid_amount = order_depth.buy_orders[best_bid]

            if not prevent_adverse or abs(best_bid_amount) <= adverse_volume:
                # If the best bid is above the threshold (fair value plus take_width), it's an attractive sell.
                if best_bid >= fair_value + take_width:
                    # Calculate how many units can be sold without exceeding position limits.
                    quantity = min(best_bid_amount, position_limit + position)
                    if quantity > 0:
                        # Place a sell order at the best bid price.
                        orders.append(Order(product, best_bid, -1 * quantity))
                        sell_order_volume += quantity
                        order_depth.buy_orders[best_bid] -= quantity
                        if order_depth.buy_orders[best_bid] == 0:
                            del order_depth.buy_orders[best_bid]

        # Return the updated volumes for buy and sell orders taken.
        return buy_order_volume, sell_order_volume

    # Implements a basic market-making strategy by placing both buy and sell orders.
    def market_make(
        self,
        product: str,
        orders: List[Order],
        bid: int,
        ask: int,
        position: int,
        buy_order_volume: int,
        sell_order_volume: int,
    ) -> (int, int):
        # Calculate available capacity to buy based on current position and order volume.
        buy_quantity = self.LIMIT[product] - (position + buy_order_volume)
        if buy_quantity > 0:
            # Place a limit order to buy at the specified bid price.
            orders.append(Order(product, round(bid), buy_quantity))  

        # Calculate available capacity to sell.
        sell_quantity = self.LIMIT[product] + (position - sell_order_volume)
        if sell_quantity > 0:
            # Place a limit order to sell at the specified ask price.
            orders.append(Order(product, round(ask), -sell_quantity))  
        return buy_order_volume, sell_order_volume

    # Attempts to clear positions by offsetting existing orders, based on market prices.
    def clear_position_order(
        self,
        product: str,
        fair_value: float,
        width: int,
        orders: List[Order],
        order_depth: OrderDepth,
        position: int,
        buy_order_volume: int,
        sell_order_volume: int,
    ) -> List[Order]:
        # Calculate net position after orders already taken.
        position_after_take = position + buy_order_volume - sell_order_volume
        # Define target prices for clearing orders, adjusted by the clear width.
        fair_for_bid = round(fair_value - width)
        fair_for_ask = round(fair_value + width)

        # Determine the additional volume that can be bought or sold.
        buy_quantity = self.LIMIT[product] - (position + buy_order_volume)
        sell_quantity = self.LIMIT[product] + (position - sell_order_volume)

        # If the net position is positive, look to clear some long position.
        if position_after_take > 0:
            # Sum volume from all buy orders at prices that are higher than the desired ask clearance price.
            clear_quantity = sum(
                volume
                for price, volume in order_depth.buy_orders.items()
                if price >= fair_for_ask
            )
            clear_quantity = min(clear_quantity, position_after_take)
            # Decide on the amount to clear based on remaining capacity to sell.
            sent_quantity = min(sell_quantity, clear_quantity)
            if sent_quantity > 0:
                # Send a sell order to reduce the position.
                orders.append(Order(product, fair_for_ask, -abs(sent_quantity)))
                sell_order_volume += abs(sent_quantity)

        # If the net position is negative, look to clear some short position.
        if position_after_take < 0:
            # Sum volume from all sell orders at prices lower than the desired bid clearance price.
            clear_quantity = sum(
                abs(volume)
                for price, volume in order_depth.sell_orders.items()
                if price <= fair_for_bid
            )
            clear_quantity = min(clear_quantity, abs(position_after_take))
            # Determine the amount to clear based on capacity to buy.
            sent_quantity = min(buy_quantity, clear_quantity)
            if sent_quantity > 0:
                # Send a buy order to reduce the short position.
                orders.append(Order(product, fair_for_bid, abs(sent_quantity)))
                buy_order_volume += abs(sent_quantity)

        return buy_order_volume, sell_order_volume

    # Calculates a fair value for KELP based on current market depth and recent price history.
    def KELP_fair_value(self, order_depth: OrderDepth, traderObject) -> float:
        # Proceed only if there are both sell and buy orders.
        if len(order_depth.sell_orders) != 0 and len(order_depth.buy_orders) != 0:
            best_ask = min(order_depth.sell_orders.keys())
            best_bid = max(order_depth.buy_orders.keys())

            # Filter out orders that are below the adverse volume threshold to avoid extreme moves.
            filtered_ask = [
                price
                for price in order_depth.sell_orders.keys()
                if abs(order_depth.sell_orders[price])
                >= self.params[Product.KELP]["adverse_volume"]
            ]
            filtered_bid = [
                price
                for price in order_depth.buy_orders.keys()
                if abs(order_depth.buy_orders[price])
                >= self.params[Product.KELP]["adverse_volume"]
            ]
            mm_ask = min(filtered_ask) if len(filtered_ask) > 0 else None
            mm_bid = max(filtered_bid) if len(filtered_bid) > 0 else None

            # If filtered values are missing, use the best bid/ask or fall back to the last price.
            if mm_ask == None or mm_bid == None:
                if traderObject.get("KELP_last_price", None) == None:
                    mmmid_price = (best_ask + best_bid) / 2
                else:
                    mmmid_price = traderObject["KELP_last_price"]
            else:
                mmmid_price = (mm_ask + mm_bid) / 2

            # Apply a reversion adjustment based on how far the current mid-price is from the last recorded price.
            if traderObject.get("KELP_last_price", None) != None:
                last_price = traderObject["KELP_last_price"]
                last_returns = (mmmid_price - last_price) / last_price
                pred_returns = (
                    last_returns * self.params[Product.KELP]["reversion_beta"]
                )
                fair = mmmid_price + (mmmid_price * pred_returns)
            else:
                fair = mmmid_price

            # Update the trader object with the latest mid-price.
            traderObject["KELP_last_price"] = mmmid_price
            return fair
        return None

    # Convenience wrapper to take orders – aggregates volumes and uses the 'take_best_orders' function.
    def take_orders(
        self,
        product: str,
        order_depth: OrderDepth,
        fair_value: float,
        take_width: float,
        position: int,
        prevent_adverse: bool = False,
        adverse_volume: int = 0,
    ) -> (List[Order], int, int):
        orders: List[Order] = []
        buy_order_volume = 0
        sell_order_volume = 0

        # Execute order taking based on current market depth and parameters.
        buy_order_volume, sell_order_volume = self.take_best_orders(
            product,
            fair_value,
            take_width,
            orders,
            order_depth,
            position,
            buy_order_volume,
            sell_order_volume,
            prevent_adverse,
            adverse_volume,
        )
        return orders, buy_order_volume, sell_order_volume

    # Convenience wrapper to clear positions – aggregates volumes and utilizes 'clear_position_order'.
    def clear_orders(
        self,
        product: str,
        order_depth: OrderDepth,
        fair_value: float,
        clear_width: int,
        position: int,
        buy_order_volume: int,
        sell_order_volume: int,
    ) -> (List[Order], int, int):
        orders: List[Order] = []
        buy_order_volume, sell_order_volume = self.clear_position_order(
            product,
            fair_value,
            clear_width,
            orders,
            order_depth,
            position,
            buy_order_volume,
            sell_order_volume,
        )
        return orders, buy_order_volume, sell_order_volume

    # Generates new orders (for market making) considering current market depths and edge strategies.
    def make_orders(
        self,
        product,
        order_depth: OrderDepth,
        fair_value: float,
        position: int,
        buy_order_volume: int,
        sell_order_volume: int,
        disregard_edge: float,  # Ignore orders that fall too close to fair value when deciding to undercut or join
        join_edge: float,       # If orders are within this edge, join them instead of undercutting
        default_edge: float,    # A default offset from fair value if no matching levels are available
        manage_position: bool = False,
        soft_position_limit: int = 0,
    ):
        orders: List[Order] = []

        # Identify asks that are placed above fair value and bids below, but only those beyond the disregard threshold.
        asks_above_fair = [
            price
            for price in order_depth.sell_orders.keys()
            if price > fair_value + disregard_edge
        ]
        bids_below_fair = [
            price
            for price in order_depth.buy_orders.keys()
            if price < fair_value - disregard_edge
        ]

        # Determine the best ask above fair value, if any exist.
        best_ask_above_fair = min(asks_above_fair) if len(asks_above_fair) > 0 else None
        # Determine the best bid below fair value, if any exist.
        best_bid_below_fair = max(bids_below_fair) if len(bids_below_fair) > 0 else None

        # Initialize ask using the default edge.
        ask = round(fair_value + default_edge)
        if best_ask_above_fair != None:
            # If the best ask is close enough (within join_edge), join at that price.
            if abs(best_ask_above_fair - fair_value) <= join_edge:
                ask = best_ask_above_fair  # Join the level
            else:
                ask = best_ask_above_fair - 1  # Undercut by one unit (penny) to improve likelihood of execution

        # Similarly, initialize bid using default_edge.
        bid = round(fair_value - default_edge)
        if best_bid_below_fair != None:
            if abs(fair_value - best_bid_below_fair) <= join_edge:
                bid = best_bid_below_fair  # Join the level
            else:
                bid = best_bid_below_fair + 1  # Penny by raising the bid slightly

        # If managing position risk, adjust orders to favor reducing an extreme position.
        if manage_position:
            if position > soft_position_limit:
                ask -= 1  # Slightly lower the ask to encourage selling
            elif position < -1 * soft_position_limit:
                bid += 1  # Slightly raise the bid to encourage buying

        # Place the market-making orders based on the computed bid/ask prices.
        buy_order_volume, sell_order_volume = self.market_make(
            product,
            orders,
            bid,
            ask,
            position,
            buy_order_volume,
            sell_order_volume,
        )

        return orders, buy_order_volume, sell_order_volume

    # Main execution function that runs on each trading state update.
    def run(self, state: TradingState):
        # Initialize trader-specific data from previous rounds (if available),
        # using jsonpickle for serialization.
        traderObject = {}
        if state.traderData != None and state.traderData != "":
            traderObject = jsonpickle.decode(state.traderData)

        result = {}

        # Process orders for RAINFOREST_RESIN if available in both parameters and market depth data.
        if Product.RAINFOREST_RESIN in self.params and Product.RAINFOREST_RESIN in state.order_depths:
            # Retrieve the current position for RAINFOREST_RESIN; default to 0 if not present.
            amethyst_position = (
                state.position[Product.RAINFOREST_RESIN]
                if Product.RAINFOREST_RESIN in state.position
                else 0
            )
            # Generate orders by taking available orders from the market.
            amethyst_take_orders, buy_order_volume, sell_order_volume = (
                self.take_orders(
                    Product.RAINFOREST_RESIN,
                    state.order_depths[Product.RAINFOREST_RESIN],
                    self.params[Product.RAINFOREST_RESIN]["fair_value"],
                    self.params[Product.RAINFOREST_RESIN]["take_width"],
                    amethyst_position,
                )
            )
            # Generate orders to clear positions if market conditions warrant it.
            amethyst_clear_orders, buy_order_volume, sell_order_volume = (
                self.clear_orders(
                    Product.RAINFOREST_RESIN,
                    state.order_depths[Product.RAINFOREST_RESIN],
                    self.params[Product.RAINFOREST_RESIN]["fair_value"],
                    self.params[Product.RAINFOREST_RESIN]["clear_width"],
                    amethyst_position,
                    buy_order_volume,
                    sell_order_volume,
                )
            )
            # Generate market-making orders that help stabilize the order book.
            amethyst_make_orders, _, _ = self.make_orders(
                Product.RAINFOREST_RESIN,
                state.order_depths[Product.RAINFOREST_RESIN],
                self.params[Product.RAINFOREST_RESIN]["fair_value"],
                amethyst_position,
                buy_order_volume,
                sell_order_volume,
                self.params[Product.RAINFOREST_RESIN]["disregard_edge"],
                self.params[Product.RAINFOREST_RESIN]["join_edge"],
                self.params[Product.RAINFOREST_RESIN]["default_edge"],
                True,  # manage position enabled
                self.params[Product.RAINFOREST_RESIN]["soft_position_limit"],
            )
            # Combine all orders for RAINFOREST_RESIN.
            result[Product.RAINFOREST_RESIN] = (
                amethyst_take_orders + amethyst_clear_orders + amethyst_make_orders
            )

        # Process orders for KELP if applicable.
        if Product.KELP in self.params and Product.KELP in state.order_depths:
            KELP_position = (
                state.position[Product.KELP]
                if Product.KELP in state.position
                else 0
            )
            # Compute a dynamic fair value for KELP using market depth and recent price history.
            KELP_fair_value = self.KELP_fair_value(
                state.order_depths[Product.KELP], traderObject
            )
            # Take orders based on the computed fair value and market conditions.
            KELP_take_orders, buy_order_volume, sell_order_volume = (
                self.take_orders(
                    Product.KELP,
                    state.order_depths[Product.KELP],
                    KELP_fair_value,
                    self.params[Product.KELP]["take_width"],
                    KELP_position,
                    self.params[Product.KELP]["prevent_adverse"],
                    self.params[Product.KELP]["adverse_volume"],
                )
            )
            # Clear positions if necessary.
            KELP_clear_orders, buy_order_volume, sell_order_volume = (
                self.clear_orders(
                    Product.KELP,
                    state.order_depths[Product.KELP],
                    KELP_fair_value,
                    self.params[Product.KELP]["clear_width"],
                    KELP_position,
                    buy_order_volume,
                    sell_order_volume,
                )
            )
            # Generate market-making orders for KELP.
            KELP_make_orders, _, _ = self.make_orders(
                Product.KELP,
                state.order_depths[Product.KELP],
                KELP_fair_value,
                KELP_position,
                buy_order_volume,
                sell_order_volume,
                self.params[Product.KELP]["disregard_edge"],
                self.params[Product.KELP]["join_edge"],
                self.params[Product.KELP]["default_edge"],
            )
            # Combine all orders for KELP.
            result[Product.KELP] = (
                KELP_take_orders + KELP_clear_orders + KELP_make_orders
            )

        # Conversion factor used by the exchange (placeholder value).
        conversions = 1
        # Update traderData with the latest state data using jsonpickle.
        traderData = jsonpickle.encode(traderObject)

        # Return the complete set of orders for all products, the conversion value, and the updated traderData.
        return result, conversions, traderData
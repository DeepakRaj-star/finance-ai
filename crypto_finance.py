import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import datetime
import json
import os
import requests
from typing import Dict, List, Any, Optional

from utils import format_currency

# API endpoint for CoinGecko (free public API)
COINGECKO_API_BASE = "https://api.coingecko.com/api/v3"

class CryptoManager:
    def __init__(self, user_id: str):
        """
        Initialize the Crypto Manager for a specific user.

        Args:
            user_id (str): The user ID
        """
        self.user_id = user_id
        self.user_data_path = f"user_data/{user_id}"
        self.crypto_file = f"{self.user_data_path}/crypto_holdings.json"
        self.exchange_file = f"{self.user_data_path}/crypto_exchanges.json"

        # Ensure directory exists
        os.makedirs(self.user_data_path, exist_ok=True)

        # Initialize crypto holdings file if it doesn't exist
        if not os.path.exists(self.crypto_file):
            self._initialize_crypto_file()

        # Initialize exchanges file if it doesn't exist
        if not os.path.exists(self.exchange_file):
            self._initialize_exchange_file()

    def _initialize_crypto_file(self):
        """Initialize an empty crypto holdings file."""
        empty_holdings = {
            "assets": [],
            "transactions": []
        }
        with open(self.crypto_file, 'w') as f:
            json.dump(empty_holdings, f)

    def _initialize_exchange_file(self):
        """Initialize an empty crypto exchanges file."""
        empty_exchanges = {
            "exchanges": []
        }
        with open(self.exchange_file, 'w') as f:
            json.dump(empty_exchanges, f)

    def load_crypto_data(self) -> Dict[str, Any]:
        """
        Load crypto holdings data from file.

        Returns:
            Dict: Crypto holdings data
        """
        try:
            with open(self.crypto_file, 'r') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            self._initialize_crypto_file()
            return {"assets": [], "transactions": []}

    def save_crypto_data(self, crypto_data: Dict[str, Any]) -> bool:
        """
        Save crypto holdings data to file.

        Args:
            crypto_data (Dict): Crypto holdings data

        Returns:
            bool: Success or failure
        """
        try:
            with open(self.crypto_file, 'w') as f:
                json.dump(crypto_data, f, indent=2)
            return True
        except Exception as e:
            print(f"Error saving crypto data: {str(e)}")
            return False

    def load_exchange_data(self) -> Dict[str, Any]:
        """
        Load crypto exchange data from file.

        Returns:
            Dict: Crypto exchange data
        """
        try:
            with open(self.exchange_file, 'r') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            self._initialize_exchange_file()
            return {"exchanges": []}

    def save_exchange_data(self, exchange_data: Dict[str, Any]) -> bool:
        """
        Save crypto exchange data to file.

        Args:
            exchange_data (Dict): Crypto exchange data

        Returns:
            bool: Success or failure
        """
        try:
            with open(self.exchange_file, 'w') as f:
                json.dump(exchange_data, f, indent=2)
            return True
        except Exception as e:
            print(f"Error saving exchange data: {str(e)}")
            return False

    def get_crypto_assets(self) -> List[Dict[str, Any]]:
        """
        Get list of crypto assets.

        Returns:
            List[Dict]: List of crypto assets
        """
        data = self.load_crypto_data()
        return data.get("assets", [])

    def add_crypto_asset(self, asset: Dict[str, Any]) -> bool:
        """
        Add a crypto asset.

        Args:
            asset (Dict): Asset data with symbol, name, quantity, etc.

        Returns:
            bool: Success or failure
        """
        data = self.load_crypto_data()
        assets = data.get("assets", [])

        # Check if asset already exists
        for i, existing_asset in enumerate(assets):
            if existing_asset.get("symbol") == asset.get("symbol"):
                # Update existing asset
                assets[i] = asset
                data["assets"] = assets
                return self.save_crypto_data(data)

        # Add new asset
        assets.append(asset)
        data["assets"] = assets
        return self.save_crypto_data(data)

    def remove_crypto_asset(self, symbol: str) -> bool:
        """
        Remove a crypto asset.

        Args:
            symbol (str): Symbol of asset to remove

        Returns:
            bool: Success or failure
        """
        data = self.load_crypto_data()
        assets = data.get("assets", [])
        data["assets"] = [a for a in assets if a.get("symbol") != symbol]
        return self.save_crypto_data(data)

    def get_crypto_transactions(self) -> List[Dict[str, Any]]:
        """
        Get list of crypto transactions.

        Returns:
            List[Dict]: List of crypto transactions
        """
        data = self.load_crypto_data()
        return data.get("transactions", [])

    def add_crypto_transaction(self, transaction: Dict[str, Any]) -> bool:
        """
        Add a crypto transaction.

        Args:
            transaction (Dict): Transaction data

        Returns:
            bool: Success or failure
        """
        data = self.load_crypto_data()
        transactions = data.get("transactions", [])

        # Generate transaction ID if not provided
        if "id" not in transaction:
            transaction["id"] = len(transactions) + 1

        # Add timestamp if not provided
        if "timestamp" not in transaction:
            transaction["timestamp"] = datetime.datetime.now().isoformat()

        transactions.append(transaction)
        data["transactions"] = transactions

        # Update asset holdings based on transaction
        self._update_asset_holdings(transaction)

        return self.save_crypto_data(data)

    def _update_asset_holdings(self, transaction: Dict[str, Any]) -> None:
        """
        Update asset holdings based on transaction.

        Args:
            transaction (Dict): Transaction data
        """
        symbol = transaction.get("symbol")
        quantity = transaction.get("quantity", 0)
        transaction_type = transaction.get("type", "buy")

        # Load current assets
        data = self.load_crypto_data()
        assets = data.get("assets", [])

        # Find the asset
        asset_found = False
        for i, asset in enumerate(assets):
            if asset.get("symbol") == symbol:
                # Update quantity
                current_quantity = asset.get("quantity", 0)
                if transaction_type.lower() == "buy":
                    assets[i]["quantity"] = current_quantity + quantity
                else:  # sell
                    assets[i]["quantity"] = max(0, current_quantity - quantity)
                asset_found = True
                break

        # If asset not found and it's a buy, add it
        if not asset_found and transaction_type.lower() == "buy":
            # Get coin info from CoinGecko
            coin_info = self.get_coin_info(symbol)
            name = coin_info.get("name", symbol.upper()) if coin_info else symbol.upper()

            assets.append({
                "symbol": symbol,
                "name": name,
                "quantity": quantity,
                "purchase_date": transaction.get("date", datetime.datetime.now().strftime("%Y-%m-%d"))
            })

        # Save updated assets
        data["assets"] = assets
        self.save_crypto_data(data)

    def get_exchanges(self) -> List[Dict[str, Any]]:
        """
        Get list of connected crypto exchanges.

        Returns:
            List[Dict]: List of exchanges
        """
        data = self.load_exchange_data()
        return data.get("exchanges", [])

    def add_exchange(self, exchange: Dict[str, Any]) -> bool:
        """
        Add or update a crypto exchange connection.

        Args:
            exchange (Dict): Exchange data with name, api_key, etc.

        Returns:
            bool: Success or failure
        """
        data = self.load_exchange_data()
        exchanges = data.get("exchanges", [])

        # Check if exchange already exists
        for i, existing_exchange in enumerate(exchanges):
            if existing_exchange.get("name") == exchange.get("name"):
                # Update existing exchange
                exchanges[i] = exchange
                data["exchanges"] = exchanges
                return self.save_exchange_data(data)

        # Add new exchange
        exchanges.append(exchange)
        data["exchanges"] = exchanges
        return self.save_exchange_data(data)

    def remove_exchange(self, exchange_name: str) -> bool:
        """
        Remove a crypto exchange connection.

        Args:
            exchange_name (str): Name of exchange to remove

        Returns:
            bool: Success or failure
        """
        data = self.load_exchange_data()
        exchanges = data.get("exchanges", [])
        data["exchanges"] = [e for e in exchanges if e.get("name") != exchange_name]
        return self.save_exchange_data(data)

    def get_coin_list(self) -> List[Dict[str, str]]:
        """
        Get list of available cryptocurrencies.

        Returns:
            List[Dict]: List of coins with symbols and names
        """
        try:
            response = requests.get(f"{COINGECKO_API_BASE}/coins/list")
            if response.status_code == 200:
                coins = response.json()
                # Return only necessary info
                return [{"id": coin["id"], "symbol": coin["symbol"].upper(), "name": coin["name"]} 
                       for coin in coins[:1000]]  # Limit to 1000 to avoid overwhelming
            else:
                return []
        except Exception as e:
            print(f"Error fetching coin list: {str(e)}")
            return []

    def get_coin_info(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Get detailed information for a specific coin.

        Args:
            symbol (str): Coin symbol

        Returns:
            Dict: Coin information
        """
        try:
            # First, need to get the coin ID from symbol
            coin_list = self.get_coin_list()
            coin_id = None
            for coin in coin_list:
                if coin["symbol"].upper() == symbol.upper():
                    coin_id = coin["id"]
                    break

            if not coin_id:
                return None

            # Get detailed coin info
            response = requests.get(f"{COINGECKO_API_BASE}/coins/{coin_id}")
            if response.status_code == 200:
                return response.json()
            else:
                return None
        except Exception as e:
            print(f"Error fetching coin info: {str(e)}")
            return None

    def get_current_prices(self, symbols: List[str]) -> Dict[str, float]:
        """
        Get current prices for specified cryptocurrencies.

        Args:
            symbols (List[str]): List of coin symbols

        Returns:
            Dict: Symbol to price mapping
        """
        if not symbols:
            return {}

        try:
            # Convert symbols to IDs
            coin_list = self.get_coin_list()
            symbol_to_id = {}
            for coin in coin_list:
                if coin["symbol"].upper() in [s.upper() for s in symbols]:
                    symbol_to_id[coin["symbol"].upper()] = coin["id"]

            # Get prices for all found IDs
            ids = ",".join([symbol_to_id[s.upper()] for s in symbols if s.upper() in symbol_to_id])
            if not ids:
                return {}

            response = requests.get(
                f"{COINGECKO_API_BASE}/simple/price",
                params={
                    "ids": ids,
                    "vs_currencies": "usd",
                    "include_24hr_change": "true"
                }
            )

            if response.status_code == 200:
                data = response.json()
                # Convert from id -> price to symbol -> price
                result = {}
                for symbol, coin_id in symbol_to_id.items():
                    if coin_id in data:
                        result[symbol] = {
                            "price": data[coin_id]["usd"],
                            "change_24h": data[coin_id].get("usd_24h_change", 0)
                        }
                return result
            else:
                return {}
        except Exception as e:
            print(f"Error fetching prices: {str(e)}")
            return {}

    def get_price_history(self, symbol: str, days: int = 30) -> List[Dict[str, Any]]:
        """
        Get price history for a specific cryptocurrency.

        Args:
            symbol (str): Coin symbol
            days (int): Number of days of history

        Returns:
            List[Dict]: Price history data points
        """
        try:
            # First, need to get the coin ID from symbol
            coin_list = self.get_coin_list()
            coin_id = None
            for coin in coin_list:
                if coin["symbol"].upper() == symbol.upper():
                    coin_id = coin["id"]
                    break

            if not coin_id:
                return []

            # Get market chart data
            response = requests.get(
                f"{COINGECKO_API_BASE}/coins/{coin_id}/market_chart",
                params={
                    "vs_currency": "usd",
                    "days": days
                }
            )

            if response.status_code == 200:
                data = response.json()
                # Format price data
                prices = data.get("prices", [])
                return [
                    {
                        "timestamp": datetime.datetime.fromtimestamp(price[0] / 1000).strftime("%Y-%m-%d"),
                        "price": price[1]
                    }
                    for price in prices
                ]
            else:
                return []
        except Exception as e:
            print(f"Error fetching price history: {str(e)}")
            return []

    def calculate_portfolio_value(self) -> Dict[str, Any]:
        """
        Calculate the current value and performance of the crypto portfolio.

        Returns:
            Dict: Portfolio value and performance metrics
        """
        assets = self.get_crypto_assets()
        if not assets:
            return {
                "total_value": 0,
                "daily_change": 0,
                "daily_change_percent": 0,
                "assets": []
            }

        # Get current prices
        symbols = [asset["symbol"] for asset in assets]
        price_data = self.get_current_prices(symbols)

        # Calculate values for each asset
        portfolio_assets = []
        total_value = 0
        daily_change = 0

        for asset in assets:
            symbol = asset["symbol"]
            quantity = asset["quantity"]

            # Skip if no price data or quantity is zero
            if symbol not in price_data or quantity == 0:
                continue

            price = price_data[symbol]["price"]
            change_24h = price_data[symbol]["change_24h"]

            value = quantity * price
            value_change = value * (change_24h / 100)

            portfolio_assets.append({
                "symbol": symbol,
                "name": asset.get("name", symbol),
                "quantity": quantity,
                "price": price,
                "value": value,
                "change_24h": change_24h,
                "value_change_24h": value_change
            })

            total_value += value
            daily_change += value_change

        # Calculate daily change percentage
        daily_change_percent = (daily_change / total_value) * 100 if total_value > 0 else 0

        return {
            "total_value": total_value,
            "daily_change": daily_change,
            "daily_change_percent": daily_change_percent,
            "assets": portfolio_assets
        }

    def get_portfolio_allocation(self) -> List[Dict[str, Any]]:
        """
        Calculate the allocation of assets in the portfolio.

        Returns:
            List[Dict]: Asset allocation data
        """
        portfolio = self.calculate_portfolio_value()
        total_value = portfolio["total_value"]
        assets = portfolio["assets"]

        if total_value == 0 or not assets:
            return []

        # Calculate percentage for each asset
        allocation = []
        for asset in assets:
            allocation.append({
                "symbol": asset["symbol"],
                "name": asset["name"],
                "value": asset["value"],
                "percentage": (asset["value"] / total_value) * 100
            })

        # Sort by value descending
        allocation.sort(key=lambda x: x["value"], reverse=True)

        return allocation

    def get_portfolio_performance(self, days: int = 30) -> List[Dict[str, Any]]:
        """
        Calculate portfolio performance over time.

        Args:
            days (int): Number of days of history

        Returns:
            List[Dict]: Portfolio performance data points
        """
        assets = self.get_crypto_assets()
        if not assets:
            return []

        # Get price history for each asset
        symbols = [asset["symbol"] for asset in assets]

        # Combine performance data from all assets
        performance = {}

        for symbol in symbols:
            # Get asset quantity
            asset = next((a for a in assets if a["symbol"] == symbol), None)
            if not asset:
                continue

            quantity = asset["quantity"]
            if quantity == 0:
                continue

            # Get price history
            history = self.get_price_history(symbol, days)

            # Add to performance data
            for point in history:
                date = point["timestamp"]
                value = point["price"] * quantity

                if date not in performance:
                    performance[date] = 0

                performance[date] += value

        # Convert to list and sort by date
        result = [{"date": date, "value": value} for date, value in performance.items()]
        result.sort(key=lambda x: x["date"])

        return result


def crypto_finance_page(user_id: str):
    """
    Display the crypto finance page.

    Args:
        user_id (str): The ID of the current user
    """
    st.title("Crypto Finance Integration")

    # Initialize crypto manager
    crypto_manager = CryptoManager(user_id)

    # Create tabs for different cryptocurrency features
    tabs = st.tabs(["Portfolio", "Add Assets", "Transactions", "Exchanges", "Analytics"])

    # Portfolio tab
    with tabs[0]:
        st.subheader("Cryptocurrency Portfolio")

        # Calculate portfolio value
        portfolio = crypto_manager.calculate_portfolio_value()
        total_value = portfolio["total_value"]
        daily_change = portfolio["daily_change"]
        daily_change_percent = portfolio["daily_change_percent"]

        # Display portfolio metrics
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "Portfolio Value", 
                f"${total_value:,.2f}", 
                f"{daily_change_percent:+.2f}% (${daily_change:+,.2f})"
            )

        with col2:
            # Get portfolio allocation
            allocation = crypto_manager.get_portfolio_allocation()

            if allocation:
                top_asset = allocation[0]
                st.metric(
                    "Largest Holding", 
                    f"{top_asset['symbol']} (${top_asset['value']:,.2f})",
                    f"{top_asset['percentage']:.1f}% of portfolio"
                )
            else:
                st.metric("Largest Holding", "None", "0%")

        with col3:
            assets = portfolio["assets"]
            st.metric("Number of Assets", str(len(assets)), "")

        # Display portfolio assets
        if assets:
            st.subheader("Assets")

            # Create DataFrame for display
            asset_data = []
            for asset in assets:
                asset_data.append({
                    "Symbol": asset["symbol"],
                    "Name": asset["name"],
                    "Quantity": f"{asset['quantity']:.8f}".rstrip('0').rstrip('.'),
                    "Price": f"${asset['price']:,.2f}",
                    "Value": f"${asset['value']:,.2f}",
                    "24h Change": f"{asset['change_24h']:+.2f}%"
                })

            asset_df = pd.DataFrame(asset_data)
            st.dataframe(asset_df, use_container_width=True)

            # Portfolio allocation chart
            st.subheader("Portfolio Allocation")

            # Create pie chart
            if allocation:
                fig = px.pie(
                    allocation, 
                    values="value", 
                    names="symbol", 
                    title="Asset Allocation",
                    hover_data=["percentage"],
                    labels={
                        "value": "USD Value",
                        "symbol": "Asset",
                        "percentage": "Percentage"
                    }
                )

                fig.update_traces(
                    textposition="inside",
                    textinfo="percent+label",
                    hovertemplate="<b>%{label}</b><br>Value: $%{value:.2f}<br>Percentage: %{customdata[0]:.1f}%"
                )

                st.plotly_chart(fig, use_container_width=True)

            # Portfolio performance chart
            st.subheader("Portfolio Performance")

            # Time period selector
            time_period = st.selectbox(
                "Time Period",
                ["7 Days", "30 Days", "90 Days", "1 Year"],
                index=1  # Default to 30 days
            )

            # Convert to days
            if time_period == "7 Days":
                days = 7
            elif time_period == "30 Days":
                days = 30
            elif time_period == "90 Days":
                days = 90
            else:  # 1 Year
                days = 365

            # Get performance data
            performance = crypto_manager.get_portfolio_performance(days)

            if performance:
                # Create line chart
                perf_df = pd.DataFrame(performance)
                perf_df["date"] = pd.to_datetime(perf_df["date"])

                fig = px.line(
                    perf_df,
                    x="date",
                    y="value",
                    title=f"Portfolio Value ({time_period})",
                    labels={
                        "date": "Date",
                        "value": "Portfolio Value (USD)"
                    }
                )

                fig.update_layout(
                    xaxis_title="Date",
                    yaxis_title="Value (USD)",
                    hovermode="x unified"
                )

                # Format hover data
                fig.update_traces(
                    hovertemplate="$%{y:,.2f}"
                )

                st.plotly_chart(fig, use_container_width=True)

                # Calculate performance metrics
                if len(performance) > 1:
                    start_value = performance[0]["value"]
                    end_value = performance[-1]["value"]

                    absolute_change = end_value - start_value
                    percent_change = (absolute_change / start_value) * 100 if start_value > 0 else 0

                    col1, col2 = st.columns(2)

                    with col1:
                        st.metric(
                            f"Change ({time_period})", 
                            f"${absolute_change:+,.2f}", 
                            f"{percent_change:+.2f}%"
                        )

                    with col2:
                        # Simple moving average
                        if len(performance) >= 7:
                            recent_values = [p["value"] for p in performance[-7:]]
                            avg = sum(recent_values) / len(recent_values)
                            st.metric("7-Day Average", f"${avg:,.2f}")
                        else:
                            st.metric("Average Value", f"${end_value:,.2f}")
            else:
                st.info("No performance data available for the selected time period.")
        else:
            st.info("No crypto assets in your portfolio. Add assets to get started.")

    # Add Assets tab
    with tabs[1]:
        st.subheader("Add Cryptocurrency Asset")

        # Two methods to add assets
        method = st.radio("Add Method", ["Manual Entry", "Import from Exchange"])

        if method == "Manual Entry":
            with st.form("add_asset_form"):
                # Get list of cryptocurrencies
                coin_list = crypto_manager.get_coin_list()
                coin_options = [f"{coin['symbol']} ({coin['name']})" for coin in coin_list]

                # Default placeholder for search
                coin_selection = st.selectbox(
                    "Select Cryptocurrency",
                    [""] + coin_options
                )

                # Extract symbol from selection
                selected_symbol = ""
                if coin_selection:
                    selected_symbol = coin_selection.split(" ")[0]

                # Allow manual symbol entry as fallback
                symbol = st.text_input("Symbol", value=selected_symbol)
                quantity = st.number_input("Quantity", min_value=0.0, format="%.8f")
                purchase_date = st.date_input("Purchase Date", datetime.datetime.now())
                purchase_price = st.number_input("Purchase Price (USD, optional)", min_value=0.0, format="%.2f")
                notes = st.text_area("Notes (Optional)")

                submit = st.form_submit_button("Add Asset")

                if submit:
                    if symbol and quantity > 0:
                        # Create asset data
                        asset = {
                            "symbol": symbol.upper(),
                            "name": next((coin["name"] for coin in coin_list if coin["symbol"].upper() == symbol.upper()), symbol.upper()),
                            "quantity": quantity,
                            "purchase_date": purchase_date.strftime("%Y-%m-%d"),
                            "notes": notes
                        }

                        # Add optional purchase price if provided
                        if purchase_price > 0:
                            asset["purchase_price"] = purchase_price

                        # Add to manager
                        if crypto_manager.add_crypto_asset(asset):
                            # Also record as a transaction
                            transaction = {
                                "symbol": symbol.upper(),
                                "quantity": quantity,
                                "type": "buy",
                                "date": purchase_date.strftime("%Y-%m-%d"),
                                "price": purchase_price if purchase_price > 0 else None,
                                "notes": notes
                            }
                            crypto_manager.add_crypto_transaction(transaction)

                            st.success(f"Added {quantity} {symbol.upper()} to your portfolio!")
                            st.rerun()
                        else:
                            st.error("Failed to add asset. Please try again.")
                    else:
                        st.error("Please provide both symbol and quantity.")
        else:  # Import from Exchange
            st.info("To import from exchanges, first connect your exchange accounts in the Exchanges tab.")

            # Get connected exchanges
            exchanges = crypto_manager.get_exchanges()

            if exchanges:
                exchange_names = [e["name"] for e in exchanges]
                selected_exchange = st.selectbox("Select Exchange", exchange_names)

                if st.button("Import Assets"):
                    # In a real implementation, this would use the exchange API to fetch balances
                    # For this demo, we'll just show a success message
                    st.success(f"Assets imported from {selected_exchange}!")

                    # Add some example assets for demo purposes
                    if selected_exchange == "Binance":
                        demo_assets = [
                            {"symbol": "BTC", "name": "Bitcoin", "quantity": 0.05},
                            {"symbol": "ETH", "name": "Ethereum", "quantity": 2.5}
                        ]
                    elif selected_exchange == "Coinbase":
                        demo_assets = [
                            {"symbol": "SOL", "name": "Solana", "quantity": 10},
                            {"symbol": "ADA", "name": "Cardano", "quantity": 500}
                        ]
                    else:
                        demo_assets = [
                            {"symbol": "DOT", "name": "Polkadot", "quantity": 25},
                            {"symbol": "LINK", "name": "Chainlink", "quantity": 30}
                        ]

                    for asset in demo_assets:
                        asset["purchase_date"] = datetime.datetime.now().strftime("%Y-%m-%d")
                        crypto_manager.add_crypto_asset(asset)

                        # Also record as transactions
                        transaction = {
                            "symbol": asset["symbol"],
                            "quantity": asset["quantity"],
                            "type": "buy",
                            "date": asset["purchase_date"],
                            "notes": f"Imported from {selected_exchange}"
                        }
                        crypto_manager.add_crypto_transaction(transaction)

                    st.rerun()
            else:
                st.warning("No exchanges connected. Add an exchange in the Exchanges tab.")

    # Transactions tab
    with tabs[2]:
        st.subheader("Cryptocurrency Transactions")

        # Add transaction section
        with st.form("add_crypto_transaction_form"):
            col1, col2 = st.columns(2)

            with col1:
                # Get list of cryptocurrencies
                coin_list = crypto_manager.get_coin_list()
                coin_options = [f"{coin['symbol']} ({coin['name']})" for coin in coin_list]

                # Default placeholder for search
                coin_selection = st.selectbox(
                    "Select Cryptocurrency",
                    [""] + coin_options,
                    key="transaction_coin"
                )

                # Extract symbol from selection
                selected_symbol = ""
                if coin_selection:
                    selected_symbol = coin_selection.split(" ")[0]

                # Allow manual symbol entry as fallback
                symbol = st.text_input("Symbol", value=selected_symbol, key="transaction_symbol")
                quantity = st.number_input("Quantity", min_value=0.0, format="%.8f", key="transaction_quantity")

            with col2:
                transaction_type = st.selectbox("Type", ["Buy", "Sell", "Transfer In", "Transfer Out"])
                transaction_date = st.date_input("Date", datetime.datetime.now(), key="transaction_date")
                price = st.number_input("Price per Coin (USD, optional)", min_value=0.0, format="%.2f")

            exchange = st.text_input("Exchange/Wallet (Optional)")
            fee = st.number_input("Fee (USD, Optional)", min_value=0.0, format="%.2f")
            notes = st.text_area("Notes (Optional)", key="transaction_notes")

            submit_transaction = st.form_submit_button("Add Transaction")

            if submit_transaction:
                if symbol and quantity > 0:
                    # Create transaction data
                    transaction = {
                        "symbol": symbol.upper(),
                        "quantity": quantity,
                        "type": transaction_type.lower().replace(" ", "_"),
                        "date": transaction_date.strftime("%Y-%m-%d"),
                        "notes": notes
                    }

                    # Add optional fields if provided
                    if price > 0:
                        transaction["price"] = price
                    if exchange:
                        transaction["exchange"] = exchange
                    if fee > 0:
                        transaction["fee"] = fee

                    # Add to manager
                    if crypto_manager.add_crypto_transaction(transaction):
                        st.success(f"Added {transaction_type.lower()} transaction for {quantity} {symbol.upper()}!")
                        st.rerun()
                    else:
                        st.error("Failed to add transaction. Please try again.")
                else:
                    st.error("Please provide both symbol and quantity.")

        # Transaction history
        st.subheader("Transaction History")

        transactions = crypto_manager.get_crypto_transactions()

        if transactions:
            # Sort by date descending
            transactions.sort(key=lambda x: x.get("date", ""), reverse=True)

            # Create DataFrame for display
            tx_data = []
            for tx in transactions:
                tx_data.append({
                    "Date": tx.get("date", ""),
                    "Type": tx.get("type", "").replace("_", " ").title(),
                    "Symbol": tx.get("symbol", ""),
                    "Quantity": f"{tx.get('quantity', 0):.8f}".rstrip('0').rstrip('.'),
                    "Price": f"${tx.get('price', 0):,.2f}" if tx.get('price') else "-",
                    "Exchange": tx.get("exchange", "-"),
                    "Notes": tx.get("notes", "")
                })

            tx_df =pd.DataFrame(tx_data)
            st.dataframe(tx_df, use_container_width=True)
        else:
            st.info("No transactions recorded yet.")

    # Exchanges tab
    with tabs[3]:
        st.subheader("Cryptocurrency Exchanges")

        # Add exchange connection
        with st.form("add_exchange_form"):
            st.markdown("#### Connect Exchange")
            st.markdown("Add your exchange API credentials to automatically sync your holdings.")

            exchange_name = st.selectbox(
                "Exchange",
                ["Binance", "Coinbase", "Kraken", "FTX", "KuCoin", "Other"]
            )

            if exchange_name == "Other":
                custom_exchange = st.text_input("Exchange Name")
                exchange_name = custom_exchange if custom_exchange else exchange_name

            api_key = st.text_input("API Key")
            api_secret = st.text_input("API Secret", type="password")

            # Optional fields based on exchange
            if exchange_name in ["Binance", "KuCoin"]:
                passphrase = st.text_input("Passphrase/API Passphrase (if required)", type="password")
            else:
                passphrase = None

            notes = st.text_input("Description (e.g., 'Main Trading Account')")

            st.markdown("""
            > **Note**: Your API keys are stored locally and are only used to communicate directly with the exchange's API.
            > For security, use read-only API keys that do not have withdrawal permissions.
            """)

            submit_exchange = st.form_submit_button("Connect Exchange")

            if submit_exchange:
                if exchange_name and api_key and api_secret:
                    # Create exchange data
                    exchange = {
                        "name": exchange_name,
                        "api_key": api_key,
                        "api_secret": api_secret,
                        "connected_date": datetime.datetime.now().strftime("%Y-%m-%d"),
                        "notes": notes
                    }

                    # Add passphrase if provided
                    if passphrase:
                        exchange["passphrase"] = passphrase

                    # Add to manager
                    if crypto_manager.add_exchange(exchange):
                        st.success(f"Connected to {exchange_name}!")
                        st.rerun()
                    else:
                        st.error("Failed to connect exchange. Please try again.")
                else:
                    st.error("Please provide exchange name, API key, and API secret.")

        # Connected exchanges
        st.subheader("Connected Exchanges")

        exchanges = crypto_manager.get_exchanges()

        if exchanges:
            # Create DataFrame for display
            exchange_data = []
            for ex in exchanges:
                exchange_data.append({
                    "Exchange": ex.get("name", ""),
                    "Description": ex.get("notes", ""),
                    "Connected Date": ex.get("connected_date", ""),
                    "Status": "Connected"
                })

            ex_df = pd.DataFrame(exchange_data)
            st.dataframe(ex_df, use_container_width=True)

            # Option to remove exchange
            if st.button("Remove Selected Exchange"):
                st.warning("This feature would allow removing an exchange connection. Not implemented in this demo.")
        else:
            st.info("No exchanges connected yet.")

    # Analytics tab
    with tabs[4]:
        st.subheader("Cryptocurrency Analytics")

        # Get assets for analysis
        assets = crypto_manager.get_crypto_assets()

        if not assets:
            st.info("No crypto assets in your portfolio. Add assets to see analytics.")
        else:
            # Asset comparison
            st.markdown("#### Asset Performance Comparison")

            # Select assets to compare
            symbols = [asset["symbol"] for asset in assets]
            selected_symbols = st.multiselect(
                "Select Assets to Compare",
                symbols,
                default=symbols[:min(3, len(symbols))]
            )

            # Time period selector
            time_period = st.selectbox(
                "Time Period",
                ["7 Days", "30 Days", "90 Days", "1 Year"],
                index=1,  # Default to 30 days
                key="analytics_time_period"
            )

            # Convert to days
            if time_period == "7 Days":
                days = 7
            elif time_period == "30 Days":
                days = 30
            elif time_period == "90 Days":
                days = 90
            else:  # 1 Year
                days = 365

            if selected_symbols:
                # Get price history for each asset
                combined_history = {}

                for symbol in selected_symbols:
                    history = crypto_manager.get_price_history(symbol, days)

                    # Add to combined history
                    for point in history:
                        date = point["timestamp"]
                        price = point["price"]

                        if date not in combined_history:
                            combined_history[date] = {}

                        combined_history[date][symbol] = price

                # Convert to DataFrame
                if combined_history:
                    # Create DataFrame
                    history_df = pd.DataFrame([
                        {"date": date, **prices}
                        for date, prices in combined_history.items()
                    ])

                    # Sort by date
                    history_df["date"] = pd.to_datetime(history_df["date"])
                    history_df = history_df.sort_values("date")

                    # Calculate percentage change from first day
                    for symbol in selected_symbols:
                        if symbol in history_df.columns:
                            first_price = history_df[symbol].iloc[0]
                            history_df[f"{symbol}_pct"] = (history_df[symbol] / first_price - 1) * 100

                    # Create line chart for price comparison
                    st.markdown("##### Price History")

                    fig = go.Figure()

                    for symbol in selected_symbols:
                        if symbol in history_df.columns:
                            fig.add_trace(go.Scatter(
                                x=history_df["date"],
                                y=history_df[symbol],
                                mode="lines",
                                name=symbol
                            ))

                    fig.update_layout(
                        title=f"Price History ({time_period})",
                        xaxis_title="Date",
                        yaxis_title="Price (USD)",
                        hovermode="x unified"
                    )

                    st.plotly_chart(fig, use_container_width=True)

                    # Create percentage change comparison chart
                    st.markdown("##### Percentage Change")

                    fig = go.Figure()

                    for symbol in selected_symbols:
                        pct_col = f"{symbol}_pct"
                        if pct_col in history_df.columns:
                            fig.add_trace(go.Scatter(
                                x=history_df["date"],
                                y=history_df[pct_col],
                                mode="lines",
                                name=symbol
                            ))

                    fig.update_layout(
                        title=f"Percentage Change ({time_period})",
                        xaxis_title="Date",
                        yaxis_title="Change (%)",
                        hovermode="x unified"
                    )

                    # Add zero line
                    fig.add_shape(
                        type="line",
                        x0=history_df["date"].min(),
                        x1=history_df["date"].max(),
                        y0=0,
                        y1=0,
                        line=dict(color="gray", width=1, dash="dash")
                    )

                    st.plotly_chart(fig, use_container_width=True)

                    # Calculate performance metrics
                    st.markdown("##### Performance Metrics")

                    metrics = []
                    for symbol in selected_symbols:
                        if symbol in history_df.columns:
                            first_price = history_df[symbol].iloc[0]
                            last_price = history_df[symbol].iloc[-1]

                            pct_change = ((last_price / first_price) - 1) * 100

                            # Calculate volatility (standard deviation of daily returns)
                            daily_returns = history_df[symbol].pct_change().dropna()
                            volatility = daily_returns.std() * 100

                            metrics.append({
                                "Symbol": symbol,
                                "Start Price": f"${first_price:.2f}",
                                "Current Price": f"${last_price:.2f}",
                                "Change": f"{pct_change:+.2f}%",
                                "Volatility": f"{volatility:.2f}%"
                            })

                    if metrics:
                        metrics_df = pd.DataFrame(metrics)
                        st.dataframe(metrics_df, use_container_width=True)
                else:
                    st.warning("Unable to fetch price history for the selected assets.")
            else:
                st.info("Select assets to compare their performance.")

            # Correlation analysis
            st.markdown("#### Asset Correlation Analysis")
            st.markdown("This shows how the selected assets move in relation to each other.")

            if selected_symbols and len(selected_symbols) > 1 and combined_history:
                # Create DataFrame for correlation
                corr_df = pd.DataFrame([
                    {"date": date, **prices}
                    for date, prices in combined_history.items()
                ])

                # Calculate correlation matrix
                corr_matrix = corr_df[selected_symbols].corr()

                # Create heatmap
                fig = px.imshow(
                    corr_matrix,
                    labels=dict(x="Asset", y="Asset", color="Correlation"),
                    x=corr_matrix.columns,
                    y=corr_matrix.columns,
                    color_continuous_scale="RdBu_r",
                    zmin=-1,
                    zmax=1
                )

                fig.update_layout(
                    title="Asset Correlation Matrix",
                    width=600,
                    height=500
                )

                # Add correlation values as text
                for i in range(len(corr_matrix.columns)):
                    for j in range(len(corr_matrix.columns)):
                        fig.add_annotation(
                            x=i,
                            y=j,
                            text=f"{corr_matrix.iloc[j, i]:.2f}",
                            showarrow=False,
                            font=dict(color="white" if abs(corr_matrix.iloc[j, i]) > 0.5 else "black")
                        )

                st.plotly_chart(fig)

                st.markdown("""
                * **Correlation of 1.0**: Assets move perfectly together
                * **Correlation of -1.0**: Assets move perfectly opposite
                * **Correlation near 0**: Assets move independently
                """)
            elif len(selected_symbols) <= 1:
                st.info("Select at least two assets to analyze correlation.")

            # Market overview
            st.markdown("#### Crypto Market Overview")

            # Top cryptocurrencies
            try:
                # Get market data for top coins
                response = requests.get(
                    f"{COINGECKO_API_BASE}/coins/markets",
                    params={
                        "vs_currency": "usd",
                        "order": "market_cap_desc",
                        "per_page": 10,
                        "page": 1,
                        "sparkline": False,
                        "price_change_percentage": "24h"
                    }
                )

                if response.status_code == 200:
                    market_data = response.json()

                    # Create DataFrame
                    market_df = pd.DataFrame([
                        {
                            "Rank": i+1,
                            "Symbol": coin["symbol"].upper(),
                            "Name": coin["name"],
                            "Price": f"${coin['current_price']:,.2f}",
                            "24h Change": f"{coin.get('price_change_percentage_24h', 0):+.2f}%",
                            "Market Cap": f"${coin['market_cap']:,.0f}",
                            "Volume": f"${coin['total_volume']:,.0f}"
                        }
                        for i, coin in enumerate(market_data)
                    ])

                    st.dataframe(market_df, use_container_width=True)
                else:
                    st.warning("Unable to fetch market data. API rate limit may have been reached.")
            except Exception as e:
                st.warning(f"Unable to fetch market data: {str(e)}")

            # Market dominance chart
            try:
                # Calculate market dominance
                response = requests.get(
                    f"{COINGECKO_API_BASE}/global"
                )

                if response.status_code == 200:
                    global_data = response.json()["data"]
                    dominance = global_data.get("market_cap_percentage", {})

                    if dominance:
                        # Create pie chart
                        dominance_df = pd.DataFrame([
                            {"symbol": symbol.upper(), "percentage": value}
                            for symbol, value in dominance.items()
                        ])

                        fig = px.pie(
                            dominance_df,
                            values="percentage",
                            names="symbol",
                            title="Market Dominance",
                            hover_data=["percentage"]
                        )

                        fig.update_traces(
                            textposition="inside",
                            textinfo="percent+label"
                        )

                        st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.warning(f"Unable to fetch global market data: {str(e)}")
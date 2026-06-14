"""
API Integration Module for External Data Sources
Provides modular functions for Google Trends, News, Shopping, Holidays APIs
"""

import requests
import math
from datetime import datetime, timezone
from serpapi import GoogleSearch
import numpy as np


# ==================== CONFIGURATION ====================
CALENDARIFIC_API_KEY = "wnzqL8Pon8slb8csEqbQkhdqup1YVICA"
SERPAPI_KEY = "5a39c9658fc21b7327111b660c888d17d38c0f9b9b6a34c91279482d3533f1ec"

# Indian States Mapping
INDIAN_STATES = [
    "Andhra Pradesh", "Arunachal Pradesh", "Assam", "Bihar", "Chhattisgarh",
    "Goa", "Gujarat", "Haryana", "Himachal Pradesh", "Jharkhand", "Karnataka",
    "Kerala", "Madhya Pradesh", "Maharashtra", "Manipur", "Meghalaya", "Mizoram",
    "Nagaland", "Odisha", "Punjab", "Rajasthan", "Sikkim", "Tamil Nadu",
    "Telangana", "Tripura", "Uttar Pradesh", "Uttarakhand", "West Bengal"
]


# ==================== UTILITY FUNCTIONS ====================
def compute_eoq(annual_demand, ordering_cost, holding_cost):
    """
    Calculate Economic Order Quantity (EOQ)

    Args:
        annual_demand: Total annual demand for the product
        ordering_cost: Fixed cost per order
        holding_cost: Annual holding cost per unit

    Returns:
        Optimal order quantity (rounded)
    """
    if ordering_cost == 0 or holding_cost == 0:
        return 0
    return round(math.sqrt((2 * annual_demand * ordering_cost) / holding_cost))


# ==================== GOOGLE TRENDS API ====================
def get_trends_timeseries(product_name, region="IN", timeframe='today 12-m'):
    """
    Get Google Trends interest over time data for a product

    Args:
        product_name: Product name to search
        region: Country code (default: "IN" for India)
        timeframe: Time range (default: 'today 12-m')

    Returns:
        Dictionary with timeline data and averages
    """
    try:
        params = {
            "engine": "google_trends",
            "q": product_name,
            "data_type": "TIMESERIES",
            "geo": region,
            "api_key": SERPAPI_KEY
        }

        search = GoogleSearch(params)
        results = search.get_dict()
        interest_over_time = results.get("interest_over_time", {})

        # Extract timeline data
        timeline_data = interest_over_time.get("timeline_data", [])
        averages = interest_over_time.get("averages", [])

        # Calculate seasonality index
        if timeline_data:
            values = [entry["values"][0]["extracted_value"] for entry in timeline_data if entry.get("values")]
            if values:
                max_value = max(values)
                avg_value = np.mean(values)
                seasonality_index = round(max_value / avg_value, 2) if avg_value > 0 else 1.0
            else:
                seasonality_index = 1.0
        else:
            seasonality_index = 1.0

        return {
            "timeline_data": timeline_data,
            "averages": averages,
            "seasonality_index": seasonality_index,
            "success": True
        }

    except Exception as e:
        print(f"Error fetching trends timeseries for {product_name}: {str(e)}")
        return {
            "timeline_data": [],
            "averages": [],
            "seasonality_index": 1.0,
            "success": False,
            "error": str(e)
        }


def get_trends_by_region(product_name, region="IN"):
    """
    Get Google Trends interest breakdown by region (states in India)

    Args:
        product_name: Product name to search
        region: Country code (default: "IN" for India)

    Returns:
        Dictionary with regional breakdown data
    """
    try:
        params = {
            "engine": "google_trends",
            "q": product_name,
            "data_type": "GEO_MAP",
            "geo": region,
            "api_key": SERPAPI_KEY
        }

        search = GoogleSearch(params)
        results = search.get_dict()

        regional_breakdown = results.get("compared_breakdown_by_region", [])

        # Find user's state interest if available
        state_interest_map = {}
        for region_data in regional_breakdown:
            location = region_data.get("location", "")
            if region_data.get("values"):
                interest_value = region_data["values"][0].get("extracted_value", 0)
                state_interest_map[location] = interest_value

        return {
            "regional_breakdown": regional_breakdown,
            "state_interest_map": state_interest_map,
            "success": True
        }

    except Exception as e:
        print(f"Error fetching regional trends for {product_name}: {str(e)}")
        return {
            "regional_breakdown": [],
            "state_interest_map": {},
            "success": False,
            "error": str(e)
        }


def get_state_interest(product_name, state_name, region="IN"):
    """
    Get Google Trends interest for a specific state

    Args:
        product_name: Product name to search
        state_name: Name of the state (e.g., "Maharashtra")
        region: Country code (default: "IN" for India)

    Returns:
        Interest value for the state (0-100)
    """
    regional_data = get_trends_by_region(product_name, region)
    if regional_data["success"]:
        return regional_data["state_interest_map"].get(state_name, 0)
    return 0


def get_trends_visualization_data(product_name, region="IN", timeframe='today 12-m'):
    """
    Get formatted Google Trends data optimized for frontend visualization

    Args:
        product_name: Product name to search
        region: Country code (default: "IN" for India)
        timeframe: Time range (default: 'today 12-m')

    Returns:
        Dictionary with chart-ready data
    """
    try:
        timeseries_data = get_trends_timeseries(product_name, region, timeframe)

        if not timeseries_data["success"] or not timeseries_data["timeline_data"]:
            return {
                "success": False,
                "chart_data": [],
                "labels": [],
                "values": []
            }

        # Format data for charts (e.g., Chart.js, Recharts)
        chart_data = []
        labels = []
        values = []

        for entry in timeseries_data["timeline_data"]:
            date = entry.get("date", "")
            if entry.get("values") and len(entry["values"]) > 0:
                value = entry["values"][0].get("extracted_value", 0)

                chart_data.append({
                    "date": date,
                    "timestamp": entry.get("timestamp", ""),
                    "value": value
                })
                labels.append(date)
                values.append(value)

        return {
            "success": True,
            "chart_data": chart_data,  # Array of {date, timestamp, value}
            "labels": labels,  # For x-axis
            "values": values,  # For y-axis
            "seasonality_index": timeseries_data.get("seasonality_index", 1.0),
            "average": timeseries_data.get("averages", [{}])[0].get("value", 0) if timeseries_data.get("averages") else 0
        }

    except Exception as e:
        print(f"Error fetching trends visualization data for {product_name}: {str(e)}")
        return {
            "success": False,
            "chart_data": [],
            "labels": [],
            "values": [],
            "error": str(e)
        }


# ==================== NEWS SENTIMENT API ====================
def get_product_news(product_name, location="India", num_results=10):
    """
    Get recent news articles related to a product in a specific location

    Args:
        product_name: Product name to search
        location: Location/region for news (default: "India")
        num_results: Number of news results to fetch

    Returns:
        List of news articles with title, link, snippet, and date
    """
    try:
        params = {
            "engine": "google_news",
            "q": f"{product_name} {location}",
            "gl": "in",
            "hl": "en",
            "api_key": SERPAPI_KEY
        }

        search = GoogleSearch(params)
        results = search.get_dict()
        news_results = results.get("news_results", [])[:num_results]

        parsed_news = []
        for result in news_results:
            parsed_news.append({
                "title": result.get("title", ""),
                "link": result.get("link", ""),
                "snippet": result.get("snippet", ""),
                "date": result.get("date", ""),
                "source": result.get("source", {}).get("name", "")
            })

        return {
            "news": parsed_news,
            "count": len(parsed_news),
            "success": True
        }

    except Exception as e:
        print(f"Error fetching news for {product_name}: {str(e)}")
        return {
            "news": [],
            "count": 0,
            "success": False,
            "error": str(e)
        }


# ==================== SHOPPING PRICE API ====================
def get_product_pricing(product_name, location="India"):
    """
    Get current market pricing information for a product

    Args:
        product_name: Product name to search
        location: Location for shopping search (default: "India")

    Returns:
        Dictionary with pricing information and market insights
    """
    try:
        params = {
            "engine": "google_shopping",
            "q": product_name,
            "location": location,
            "google_domain": "google.co.in",
            "hl": "en",
            "gl": "in",
            "api_key": SERPAPI_KEY
        }

        search = GoogleSearch(params)
        results = search.get_dict()
        shopping_results = results.get("shopping_results", [])

        if not shopping_results:
            return {
                "avg_price": 0,
                "min_price": 0,
                "max_price": 0,
                "price_variance": 0,
                "num_sellers": 0,
                "success": False
            }

        # Extract prices
        prices = []
        for result in shopping_results:
            price_str = result.get("price", "")
            extracted_price = result.get("extracted_price", 0)
            if extracted_price:
                prices.append(extracted_price)

        if prices:
            avg_price = round(np.mean(prices), 2)
            min_price = round(min(prices), 2)
            max_price = round(max(prices), 2)
            price_variance = round(np.std(prices), 2) if len(prices) > 1 else 0
        else:
            avg_price = min_price = max_price = price_variance = 0

        return {
            "avg_price": avg_price,
            "min_price": min_price,
            "max_price": max_price,
            "price_variance": price_variance,
            "num_sellers": len(prices),
            "top_products": shopping_results[:5],
            "success": True
        }

    except Exception as e:
        print(f"Error fetching pricing for {product_name}: {str(e)}")
        return {
            "avg_price": 0,
            "min_price": 0,
            "max_price": 0,
            "price_variance": 0,
            "num_sellers": 0,
            "success": False,
            "error": str(e)
        }


# ==================== HOLIDAYS API ====================
def get_upcoming_holidays(country="IN", year=None, months_ahead=3):
    """
    Get upcoming public holidays for demand forecasting

    Args:
        country: Country code (default: "IN" for India)
        year: Year to fetch holidays (default: current year)
        months_ahead: Number of months to look ahead

    Returns:
        List of upcoming holidays with dates and types
    """
    try:
        if year is None:
            year = datetime.now().year

        params = {
            "api_key": CALENDARIFIC_API_KEY,
            "country": country,
            "year": year
        }

        response = requests.get("https://calendarific.com/api/v2/holidays", params=params)
        data = response.json()

        if response.status_code != 200 or "response" not in data:
            return {
                "holidays": [],
                "count": 0,
                "success": False,
                "error": "API request failed"
            }

        all_holidays = data["response"]["holidays"]

        # Filter upcoming holidays
        current_date = datetime.now(timezone.utc)
        end_month = current_date.month + months_ahead
        end_year = current_date.year
        if end_month > 12:
            end_year += end_month // 12
            end_month = end_month % 12
        end_date = datetime(end_year, end_month, current_date.day, tzinfo=timezone.utc)

        upcoming_holidays = []
        for holiday in all_holidays:
            holiday_date_str = holiday["date"]["iso"]
            # Parse ISO date and make it timezone-aware
            if 'T' in holiday_date_str:
                holiday_date = datetime.fromisoformat(holiday_date_str.replace('Z', '+00:00'))
            else:
                # If date only, assume midnight UTC
                holiday_date = datetime.fromisoformat(holiday_date_str).replace(tzinfo=timezone.utc)

            if current_date <= holiday_date <= end_date:
                upcoming_holidays.append({
                    "name": holiday["name"],
                    "date": holiday["date"]["iso"],
                    "type": ", ".join(holiday["type"]),
                    "description": holiday.get("description", "")
                })

        return {
            "holidays": upcoming_holidays,
            "count": len(upcoming_holidays),
            "success": True
        }

    except Exception as e:
        print(f"Error fetching holidays: {str(e)}")
        return {
            "holidays": [],
            "count": 0,
            "success": False,
            "error": str(e)
        }


# ==================== INTEGRATED INTELLIGENCE ====================
def get_comprehensive_product_intelligence(product_name, user_location=None, user_state=None):
    """
    Get comprehensive market intelligence for a product combining all data sources

    Args:
        product_name: Product name to analyze
        user_location: User's location/city (optional)
        user_state: User's state (for regional trends)

    Returns:
        Dictionary with comprehensive intelligence data
    """
    intelligence = {
        "product_name": product_name,
        "user_location": user_location,
        "user_state": user_state,
        "timestamp": datetime.now().isoformat()
    }

    # 1. Google Trends Analysis
    print(f"Fetching trends data for {product_name}...")
    trends_timeseries = get_trends_timeseries(product_name)
    intelligence["trends_timeseries"] = trends_timeseries
    intelligence["seasonality_index"] = trends_timeseries.get("seasonality_index", 1.0)

    # 2. Regional Interest
    if user_state:
        print(f"Fetching regional interest for {user_state}...")
        regional_data = get_trends_by_region(product_name)
        state_interest = regional_data["state_interest_map"].get(user_state, 0)
        intelligence["state_interest"] = state_interest
        intelligence["regional_data"] = regional_data
    else:
        intelligence["state_interest"] = 0
        intelligence["regional_data"] = {"success": False}

    # 3. Market Pricing
    print(f"Fetching pricing data for {product_name}...")
    location_query = user_state if user_state else "India"
    pricing_data = get_product_pricing(product_name, location_query)
    intelligence["pricing"] = pricing_data

    # 4. News Sentiment
    print(f"Fetching news data for {product_name}...")
    news_data = get_product_news(product_name, location_query)
    intelligence["news"] = news_data

    # 5. Upcoming Holidays
    print("Fetching upcoming holidays...")
    holidays_data = get_upcoming_holidays()
    intelligence["upcoming_holidays"] = holidays_data

    return intelligence


# ==================== TESTING ====================
if __name__ == "__main__":
    # Test the module
    product = "Laptop"
    state = "Maharashtra"

    print(f"\n{'='*60}")
    print(f"Testing API Integrations for: {product}")
    print(f"{'='*60}\n")

    # Test comprehensive intelligence
    intelligence = get_comprehensive_product_intelligence(product, user_state=state)

    print(f"\n📊 Seasonality Index: {intelligence['seasonality_index']}")
    print(f"📍 State Interest ({state}): {intelligence['state_interest']}")
    print(f"💰 Avg Market Price: ₹{intelligence['pricing']['avg_price']}")
    print(f"📰 News Articles Found: {intelligence['news']['count']}")
    print(f"🎉 Upcoming Holidays: {intelligence['upcoming_holidays']['count']}")

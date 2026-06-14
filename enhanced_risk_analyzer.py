"""
Enhanced Risk Analysis Module
Integrates external data sources for comprehensive risk and trend analysis
"""

import numpy as np
from datetime import datetime
from api_integrations import (
    get_comprehensive_product_intelligence,
    get_trends_timeseries,
    get_state_interest,
    get_product_news,
    get_product_pricing,
    get_upcoming_holidays,
    get_trends_visualization_data
)
from sentiment_analyzer import SentimentAnalyzer


class EnhancedRiskAnalyzer:
    """
    Comprehensive risk analyzer that combines:
    - Historical demand patterns
    - Google Trends data
    - News sentiment
    - Market pricing
    - Regional interest
    - Seasonality and holidays
    """

    def __init__(self):
        self.sentiment_analyzer = SentimentAnalyzer()

    def analyze_product_risk(
        self,
        product_name,
        category,
        avg_demand,
        std_demand,
        predicted_demand,
        safety_stock,
        user_state=None,
        user_location=None,
        use_external_data=True
    ):
        """
        Comprehensive risk analysis for a product

        Args:
            product_name: Name of the product
            category: Product category
            avg_demand: Historical average demand
            std_demand: Standard deviation of demand
            predicted_demand: ML-predicted demand
            safety_stock: Calculated safety stock
            user_state: User's state (for regional trends)
            user_location: User's location/city
            use_external_data: Whether to use external APIs

        Returns:
            Dictionary with risk level, trend, and detailed analysis
        """
        # Initialize base risk score
        risk_score = 0
        trend = "stable"
        demand_change_pct = 0

        # ========== FACTOR 1: Historical Demand Variability ==========
        cv = (std_demand / avg_demand) if avg_demand > 0 else 0  # Coefficient of variation

        if cv > 0.6:
            risk_score += 3
        elif cv > 0.4:
            risk_score += 2
        elif cv > 0.2:
            risk_score += 1

        # ========== FACTOR 2: Predicted Demand Trend ==========
        demand_change_pct = ((predicted_demand - avg_demand) / avg_demand * 100) if avg_demand > 0 else 0

        if predicted_demand > avg_demand * 1.15:  # 15% increase
            trend = "increasing"
            if demand_change_pct > 25:
                risk_score += 3
            else:
                risk_score += 2
        elif predicted_demand < avg_demand * 0.85:  # 15% decrease
            trend = "decreasing"
            risk_score -= 1
        else:
            trend = "stable"

        # ========== FACTOR 3: Safety Stock Adequacy ==========
        safety_ratio = safety_stock / avg_demand if avg_demand > 0 else 0
        if safety_ratio < 0.3:  # Less than 30% safety stock
            risk_score += 2
        elif safety_ratio < 0.5:
            risk_score += 1

        # ========== EXTERNAL DATA INTEGRATION ==========
        external_insights = {}
        if use_external_data:
            try:
                external_insights = self._get_external_insights(
                    product_name,
                    user_state,
                    user_location
                )

                # FACTOR 4: Google Trends Seasonality
                seasonality_index = external_insights.get("seasonality_index", 1.0)
                if seasonality_index > 1.5:
                    risk_score += 2  # High seasonality = higher risk
                    external_insights["seasonality_risk"] = "high"
                elif seasonality_index > 1.2:
                    risk_score += 1
                    external_insights["seasonality_risk"] = "medium"
                else:
                    external_insights["seasonality_risk"] = "low"

                # FACTOR 5: Regional Interest
                if user_state:
                    state_interest = external_insights.get("state_interest", 0)
                    if state_interest > 70:  # High regional interest
                        risk_score += 2
                        external_insights["regional_risk"] = "high"
                    elif state_interest > 50:
                        risk_score += 1
                        external_insights["regional_risk"] = "medium"
                    else:
                        external_insights["regional_risk"] = "low"

                # FACTOR 6: News Sentiment
                sentiment_data = external_insights.get("sentiment_analysis", {})
                sentiment_risk_modifier = self.sentiment_analyzer.get_sentiment_risk_modifier(sentiment_data)
                risk_score += sentiment_risk_modifier
                external_insights["sentiment_risk_modifier"] = sentiment_risk_modifier

                # Update trend based on sentiment if strong signal
                sentiment_trend = self.sentiment_analyzer.get_trend_from_sentiment(sentiment_data)
                if sentiment_data.get("articles_analyzed", 0) >= 3:
                    # Only override trend if we have enough data
                    if sentiment_trend == "increasing" and trend != "decreasing":
                        trend = "increasing"
                    elif sentiment_trend == "decreasing" and trend != "increasing":
                        trend = "decreasing"

                # FACTOR 7: Market Pricing Competitiveness
                pricing_data = external_insights.get("pricing", {})
                if pricing_data.get("success", False):
                    # If our price is competitive (lower than market avg), risk of stockout increases
                    external_insights["pricing_competitive"] = True

                # FACTOR 8: Upcoming Holidays
                holidays_count = external_insights.get("upcoming_holidays_count", 0)
                if holidays_count > 2:  # Multiple holidays coming
                    risk_score += 2
                    external_insights["holiday_risk"] = "high"
                elif holidays_count > 0:
                    risk_score += 1
                    external_insights["holiday_risk"] = "medium"
                else:
                    external_insights["holiday_risk"] = "low"

            except Exception as e:
                print(f"Warning: Could not fetch external data for {product_name}: {str(e)}")
                external_insights["error"] = str(e)

        # ========== DETERMINE FINAL RISK LEVEL ==========
        if risk_score >= 7:
            risk_level = "critical"
        elif risk_score >= 5:
            risk_level = "high"
        elif risk_score >= 2:
            risk_level = "medium"
        else:
            risk_level = "low"

        return {
            "risk_level": risk_level,
            "risk_score": risk_score,
            "trend": trend,
            "demand_change_pct": round(demand_change_pct, 2),
            "coefficient_variation": round(cv, 3),
            "safety_ratio": round(safety_ratio, 3),
            "external_insights": external_insights,
            "analysis_timestamp": datetime.now().isoformat()
        }

    def _get_external_insights(self, product_name, user_state, user_location):
        """
        Fetch and process external data sources

        Args:
            product_name: Name of the product
            user_state: User's state
            user_location: User's location

        Returns:
            Dictionary with processed external insights
        """
        insights = {}

        # 1. Google Trends
        trends_data = get_trends_timeseries(product_name)
        insights["seasonality_index"] = trends_data.get("seasonality_index", 1.0)
        insights["trends_success"] = trends_data.get("success", False)

        # 1b. Get visualization data for charts
        trends_viz = get_trends_visualization_data(product_name)
        insights["trends_chart"] = trends_viz if trends_viz.get("success") else None

        # 2. Regional Interest
        if user_state:
            state_interest = get_state_interest(product_name, user_state)
            insights["state_interest"] = state_interest
        else:
            insights["state_interest"] = 0

        # 3. News Sentiment
        location_query = user_state if user_state else "India"
        news_data = get_product_news(product_name, location_query, num_results=10)

        if news_data.get("success", False) and news_data.get("count", 0) > 0:
            sentiment_analysis = self.sentiment_analyzer.analyze_news_articles(news_data["news"])
            insights["sentiment_analysis"] = sentiment_analysis
            insights["news_count"] = news_data["count"]
        else:
            insights["sentiment_analysis"] = {
                "overall_sentiment": "neutral",
                "demand_signal": "stable",
                "articles_analyzed": 0
            }
            insights["news_count"] = 0

        # 4. Market Pricing
        pricing_data = get_product_pricing(product_name, location_query)
        insights["pricing"] = pricing_data

        # 5. Upcoming Holidays
        holidays_data = get_upcoming_holidays(months_ahead=3)
        insights["upcoming_holidays_count"] = holidays_data.get("count", 0)
        insights["upcoming_holidays"] = holidays_data.get("holidays", [])

        return insights

    def calculate_optimal_reorder_point(
        self,
        avg_demand,
        std_demand,
        lead_time_days=7,
        service_level=0.95,
        seasonality_index=1.0
    ):
        """
        Calculate optimal reorder point with seasonality adjustment

        Args:
            avg_demand: Average daily demand
            std_demand: Standard deviation of demand
            lead_time_days: Lead time in days
            service_level: Target service level (0-1)
            seasonality_index: Seasonality multiplier from trends

        Returns:
            Optimal reorder point
        """
        # Z-score for service level (95% = 1.65, 99% = 2.33)
        z_scores = {0.90: 1.28, 0.95: 1.65, 0.99: 2.33}
        z_score = z_scores.get(service_level, 1.65)

        # Safety stock calculation
        safety_stock = z_score * std_demand

        # Base reorder point
        reorder_point = avg_demand * lead_time_days + safety_stock

        # Adjust for seasonality
        adjusted_reorder_point = reorder_point * seasonality_index

        return round(adjusted_reorder_point, 2)

    def get_risk_explanation(self, analysis_result):
        """
        Generate human-readable explanation of risk factors

        Args:
            analysis_result: Result from analyze_product_risk()

        Returns:
            List of risk factor explanations
        """
        explanations = []
        insights = analysis_result.get("external_insights", {})

        # Demand variability
        cv = analysis_result.get("coefficient_variation", 0)
        if cv > 0.4:
            explanations.append(f"High demand variability (CV: {cv:.2f})")

        # Trend
        trend = analysis_result.get("trend", "stable")
        change_pct = analysis_result.get("demand_change_pct", 0)
        if trend == "increasing":
            explanations.append(f"Demand increasing by {change_pct:.1f}%")
        elif trend == "decreasing":
            explanations.append(f"Demand decreasing by {abs(change_pct):.1f}%")

        # Seasonality
        if insights.get("seasonality_risk") == "high":
            explanations.append("High seasonality detected in search trends")

        # Regional interest
        if insights.get("regional_risk") == "high":
            explanations.append("Very high interest in your region")

        # Sentiment
        sentiment_analysis = insights.get("sentiment_analysis", {})
        if sentiment_analysis.get("demand_signal") == "increasing":
            explanations.append("News sentiment indicates rising demand")
        elif sentiment_analysis.get("demand_signal") == "decreasing":
            explanations.append("News sentiment indicates falling demand")

        # Holidays
        if insights.get("holiday_risk") == "high":
            explanations.append(f"Multiple holidays approaching ({insights.get('upcoming_holidays_count', 0)} events)")

        return explanations


# ==================== TESTING ====================
if __name__ == "__main__":
    analyzer = EnhancedRiskAnalyzer()

    # Test with sample product data
    print("\n" + "="*60)
    print("Enhanced Risk Analysis Test")
    print("="*60 + "\n")

    test_product = {
        "product_name": "Laptop",
        "category": "Electronics",
        "avg_demand": 50,
        "std_demand": 15,
        "predicted_demand": 65,
        "safety_stock": 25,
        "user_state": "Maharashtra",
        "user_location": "Mumbai"
    }

    print(f"Analyzing: {test_product['product_name']}")
    print(f"Location: {test_product['user_location']}, {test_product['user_state']}\n")

    # Run analysis
    result = analyzer.analyze_product_risk(**test_product, use_external_data=True)

    print(f"Risk Level: {result['risk_level'].upper()}")
    print(f"Risk Score: {result['risk_score']}")
    print(f"Trend: {result['trend']}")
    print(f"Demand Change: {result['demand_change_pct']}%\n")

    # Print explanations
    explanations = analyzer.get_risk_explanation(result)
    if explanations:
        print("Risk Factors:")
        for i, explanation in enumerate(explanations, 1):
            print(f"  {i}. {explanation}")

    # Print external insights
    insights = result.get("external_insights", {})
    print(f"\nExternal Data Insights:")
    print(f"  Seasonality Index: {insights.get('seasonality_index', 'N/A')}")
    print(f"  Regional Interest: {insights.get('state_interest', 'N/A')}")
    print(f"  News Articles: {insights.get('news_count', 0)}")
    print(f"  Sentiment: {insights.get('sentiment_analysis', {}).get('overall_sentiment', 'N/A')}")
    print(f"  Upcoming Holidays: {insights.get('upcoming_holidays_count', 0)}")

"""
Sentiment Analysis Module for News and Product Reviews
Analyzes text sentiment to gauge market perception and demand signals
"""

from textblob import TextBlob
import re


class SentimentAnalyzer:
    """
    Analyzer for determining sentiment from text data
    Uses TextBlob for polarity and subjectivity analysis
    """

    POSITIVE_KEYWORDS = [
        "popular", "trending", "demand", "bestseller", "sold out", "buy",
        "shortage", "limited stock", "high demand", "rush", "surge",
        "increase", "growth", "rising", "booming", "hot item"
    ]

    NEGATIVE_KEYWORDS = [
        "decline", "drop", "falling", "decrease", "slow", "weak demand",
        "surplus", "overstocked", "clearance", "discount", "markdown",
        "slump", "downturn", "reduced interest"
    ]

    def __init__(self):
        pass

    def analyze_text(self, text):
        """
        Analyze sentiment of a single text

        Args:
            text: Text to analyze

        Returns:
            Dictionary with sentiment scores and classification
        """
        if not text or not isinstance(text, str):
            return {
                "polarity": 0.0,
                "subjectivity": 0.0,
                "sentiment": "neutral",
                "confidence": 0.0
            }

        # Clean text
        text_clean = self._clean_text(text)

        # Get TextBlob sentiment
        blob = TextBlob(text_clean)
        polarity = blob.sentiment.polarity  # -1 to 1
        subjectivity = blob.sentiment.subjectivity  # 0 to 1

        # Classify sentiment
        if polarity > 0.1:
            sentiment = "positive"
        elif polarity < -0.1:
            sentiment = "negative"
        else:
            sentiment = "neutral"

        # Calculate confidence based on polarity strength
        confidence = abs(polarity)

        return {
            "polarity": round(polarity, 3),
            "subjectivity": round(subjectivity, 3),
            "sentiment": sentiment,
            "confidence": round(confidence, 3)
        }

    def analyze_news_articles(self, news_articles):
        """
        Analyze sentiment across multiple news articles

        Args:
            news_articles: List of news article dictionaries with 'title' and 'snippet'

        Returns:
            Dictionary with aggregated sentiment analysis
        """
        if not news_articles:
            return {
                "overall_sentiment": "neutral",
                "avg_polarity": 0.0,
                "avg_subjectivity": 0.0,
                "positive_count": 0,
                "negative_count": 0,
                "neutral_count": 0,
                "demand_signal": "stable",
                "articles_analyzed": 0
            }

        sentiments = []
        positive_count = 0
        negative_count = 0
        neutral_count = 0

        for article in news_articles:
            # Combine title and snippet for analysis
            text = f"{article.get('title', '')} {article.get('snippet', '')}"
            sentiment_result = self.analyze_text(text)

            sentiments.append(sentiment_result)

            if sentiment_result["sentiment"] == "positive":
                positive_count += 1
            elif sentiment_result["sentiment"] == "negative":
                negative_count += 1
            else:
                neutral_count += 1

        # Calculate averages
        avg_polarity = sum(s["polarity"] for s in sentiments) / len(sentiments)
        avg_subjectivity = sum(s["subjectivity"] for s in sentiments) / len(sentiments)

        # Determine overall sentiment
        if avg_polarity > 0.15:
            overall_sentiment = "positive"
        elif avg_polarity < -0.15:
            overall_sentiment = "negative"
        else:
            overall_sentiment = "neutral"

        # Determine demand signal
        demand_signal = self._determine_demand_signal(news_articles, avg_polarity)

        return {
            "overall_sentiment": overall_sentiment,
            "avg_polarity": round(avg_polarity, 3),
            "avg_subjectivity": round(avg_subjectivity, 3),
            "positive_count": positive_count,
            "negative_count": negative_count,
            "neutral_count": neutral_count,
            "demand_signal": demand_signal,
            "articles_analyzed": len(news_articles)
        }

    def _determine_demand_signal(self, articles, avg_polarity):
        """
        Determine demand signal based on news content and keywords

        Args:
            articles: List of news articles
            avg_polarity: Average polarity score

        Returns:
            Demand signal: "increasing", "decreasing", or "stable"
        """
        positive_keyword_count = 0
        negative_keyword_count = 0

        for article in articles:
            text = f"{article.get('title', '')} {article.get('snippet', '')}".lower()

            # Count positive keywords
            for keyword in self.POSITIVE_KEYWORDS:
                if keyword in text:
                    positive_keyword_count += 1

            # Count negative keywords
            for keyword in self.NEGATIVE_KEYWORDS:
                if keyword in text:
                    negative_keyword_count += 1

        # Determine signal based on keyword counts and polarity
        keyword_diff = positive_keyword_count - negative_keyword_count

        if keyword_diff > 2 or avg_polarity > 0.2:
            return "increasing"
        elif keyword_diff < -2 or avg_polarity < -0.2:
            return "decreasing"
        else:
            return "stable"

    def _clean_text(self, text):
        """
        Clean and preprocess text for sentiment analysis

        Args:
            text: Raw text

        Returns:
            Cleaned text
        """
        # Remove URLs
        text = re.sub(r'http\S+|www\S+', '', text)

        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s.,!?-]', '', text)

        # Remove extra whitespace
        text = ' '.join(text.split())

        return text

    def get_sentiment_risk_modifier(self, sentiment_analysis):
        """
        Calculate risk score modifier based on sentiment analysis

        Args:
            sentiment_analysis: Dictionary from analyze_news_articles()

        Returns:
            Risk modifier score (-2 to +3)
        """
        demand_signal = sentiment_analysis.get("demand_signal", "stable")
        avg_polarity = sentiment_analysis.get("avg_polarity", 0.0)

        risk_modifier = 0

        # High positive sentiment and increasing demand = higher stockout risk
        if demand_signal == "increasing":
            if avg_polarity > 0.3:
                risk_modifier = 3
            elif avg_polarity > 0.15:
                risk_modifier = 2
            else:
                risk_modifier = 1

        # Negative sentiment and decreasing demand = lower stockout risk
        elif demand_signal == "decreasing":
            if avg_polarity < -0.3:
                risk_modifier = -2
            elif avg_polarity < -0.15:
                risk_modifier = -1

        return risk_modifier

    def get_trend_from_sentiment(self, sentiment_analysis):
        """
        Determine product trend based on sentiment analysis

        Args:
            sentiment_analysis: Dictionary from analyze_news_articles()

        Returns:
            Trend: "increasing", "decreasing", or "stable"
        """
        return sentiment_analysis.get("demand_signal", "stable")


# ==================== TESTING ====================
if __name__ == "__main__":
    analyzer = SentimentAnalyzer()

    # Test with sample news articles
    sample_news = [
        {
            "title": "iPhone sales surge amid high demand",
            "snippet": "Apple reports record iPhone sales as customers rush to buy the latest model."
        },
        {
            "title": "Laptop market sees declining interest",
            "snippet": "PC manufacturers report slow sales and excess inventory as demand weakens."
        },
        {
            "title": "Smartphone prices remain stable",
            "snippet": "Market analysts predict steady smartphone prices throughout the quarter."
        }
    ]

    print("\n" + "="*60)
    print("Sentiment Analysis Test")
    print("="*60 + "\n")

    # Analyze individual articles
    for i, article in enumerate(sample_news, 1):
        print(f"Article {i}: {article['title']}")
        result = analyzer.analyze_text(f"{article['title']} {article['snippet']}")
        print(f"  Sentiment: {result['sentiment']}")
        print(f"  Polarity: {result['polarity']}")
        print(f"  Confidence: {result['confidence']}\n")

    # Analyze all articles together
    overall = analyzer.analyze_news_articles(sample_news)
    print("\n" + "-"*60)
    print("Overall Analysis:")
    print("-"*60)
    print(f"Overall Sentiment: {overall['overall_sentiment']}")
    print(f"Average Polarity: {overall['avg_polarity']}")
    print(f"Demand Signal: {overall['demand_signal']}")
    print(f"Positive Articles: {overall['positive_count']}")
    print(f"Negative Articles: {overall['negative_count']}")
    print(f"Neutral Articles: {overall['neutral_count']}")
    print(f"Risk Modifier: {analyzer.get_sentiment_risk_modifier(overall)}")

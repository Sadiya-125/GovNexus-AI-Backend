"""
Report Generation Module
Generates PDF and HTML reports for EDA and Optimization Analysis
"""

import os
import io
import base64
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO


class ReportGenerator:
    """
    Generate comprehensive reports for inventory analysis
    """

    def __init__(self):
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        sns.set_style("whitegrid")

    def generate_eda_report(self, df, output_format='html'):
        """
        Generate Exploratory Data Analysis report

        Args:
            df: DataFrame with orders data
            output_format: 'html' or 'pdf'

        Returns:
            Dictionary with report data
        """
        report_data = {
            "title": "Inventory Data Analysis Report",
            "generated_at": datetime.now().isoformat(),
            "summary_stats": self._get_summary_statistics(df),
            "charts": self._generate_eda_charts(df),
            "insights": self._generate_eda_insights(df)
        }

        if output_format == 'html':
            html_content = self._create_eda_html(report_data)
            return {
                "format": "html",
                "content": html_content,
                "filename": f"eda_report_{self.timestamp}.html"
            }
        else:
            # For PDF, return HTML that can be converted by frontend or weasyprint
            html_content = self._create_eda_html(report_data)
            return {
                "format": "html",  # Frontend can convert to PDF
                "content": html_content,
                "filename": f"eda_report_{self.timestamp}.html"
            }

    def generate_optimization_report(self, recommendations, summary, output_format='html'):
        """
        Generate Optimization Recommendations report

        Args:
            recommendations: List of product recommendations
            summary: Summary statistics
            output_format: 'html' or 'pdf'

        Returns:
            Dictionary with report data
        """
        report_data = {
            "title": "Inventory Optimization Report",
            "generated_at": datetime.now().isoformat(),
            "summary": summary,
            "recommendations": recommendations,
            "charts": self._generate_optimization_charts(recommendations),
            "insights": self._generate_optimization_insights(recommendations, summary)
        }

        if output_format == 'html':
            html_content = self._create_optimization_html(report_data)
            return {
                "format": "html",
                "content": html_content,
                "filename": f"optimization_report_{self.timestamp}.html"
            }
        else:
            html_content = self._create_optimization_html(report_data)
            return {
                "format": "html",
                "content": html_content,
                "filename": f"optimization_report_{self.timestamp}.html"
            }

    def _get_summary_statistics(self, df):
        """Generate summary statistics from DataFrame"""
        return {
            "total_orders": len(df),
            "total_products": df['product_name'].nunique() if 'product_name' in df.columns else 0,
            "total_categories": df['product_category'].nunique() if 'product_category' in df.columns else 0,
            "total_revenue": float(df['sales'].sum()) if 'sales' in df.columns else 0,
            "total_profit": float(df['profit'].sum()) if 'profit' in df.columns else 0,
            "avg_order_qty": float(df['orderQty'].mean()) if 'orderQty' in df.columns else 0,
            "date_range": {
                "start": df['orderDate'].min().isoformat() if 'orderDate' in df.columns else None,
                "end": df['orderDate'].max().isoformat() if 'orderDate' in df.columns else None
            }
        }

    def _generate_eda_charts(self, df):
        """Generate charts for EDA report"""
        charts = {}

        # 1. Sales by Category
        if 'product_category' in df.columns and 'sales' in df.columns:
            charts['sales_by_category'] = self._create_sales_by_category_chart(df)

        # 2. Top Products by Revenue
        if 'product_name' in df.columns and 'sales' in df.columns:
            charts['top_products'] = self._create_top_products_chart(df)

        # 3. Order Quantity Distribution
        if 'orderQty' in df.columns:
            charts['qty_distribution'] = self._create_qty_distribution_chart(df)

        # 4. Sales Trend Over Time
        if 'orderDate' in df.columns and 'sales' in df.columns:
            charts['sales_trend'] = self._create_sales_trend_chart(df)

        return charts

    def _generate_optimization_charts(self, recommendations):
        """Generate charts for optimization report"""
        charts = {}

        # 1. Risk Distribution
        charts['risk_distribution'] = self._create_risk_distribution_chart(recommendations)

        # 2. Top 10 High Risk Products
        charts['high_risk_products'] = self._create_high_risk_products_chart(recommendations)

        # 3. Demand Trend Distribution
        charts['demand_trends'] = self._create_demand_trend_chart(recommendations)

        return charts

    def _create_sales_by_category_chart(self, df):
        """Create sales by category bar chart"""
        fig, ax = plt.subplots(figsize=(10, 6))
        category_sales = df.groupby('product_category')['sales'].sum().sort_values(ascending=False).head(10)

        category_sales.plot(kind='bar', ax=ax, color='steelblue')
        ax.set_title('Top 10 Categories by Sales', fontsize=14, fontweight='bold')
        ax.set_xlabel('Category')
        ax.set_ylabel('Sales (₹)')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()

        return self._fig_to_base64(fig)

    def _create_top_products_chart(self, df):
        """Create top products bar chart"""
        fig, ax = plt.subplots(figsize=(12, 6))
        product_sales = df.groupby('product_name')['sales'].sum().sort_values(ascending=False).head(15)

        product_sales.plot(kind='barh', ax=ax, color='green')
        ax.set_title('Top 15 Products by Revenue', fontsize=14, fontweight='bold')
        ax.set_xlabel('Sales (₹)')
        ax.set_ylabel('Product')
        plt.tight_layout()

        return self._fig_to_base64(fig)

    def _create_qty_distribution_chart(self, df):
        """Create order quantity distribution histogram"""
        fig, ax = plt.subplots(figsize=(10, 6))

        ax.hist(df['orderQty'], bins=30, color='orange', edgecolor='black', alpha=0.7)
        ax.set_title('Order Quantity Distribution', fontsize=14, fontweight='bold')
        ax.set_xlabel('Order Quantity')
        ax.set_ylabel('Frequency')
        plt.tight_layout()

        return self._fig_to_base64(fig)

    def _create_sales_trend_chart(self, df):
        """Create sales trend over time line chart"""
        fig, ax = plt.subplots(figsize=(12, 6))

        df_copy = df.copy()
        df_copy['orderDate'] = pd.to_datetime(df_copy['orderDate'])
        daily_sales = df_copy.groupby(df_copy['orderDate'].dt.date)['sales'].sum()

        ax.plot(daily_sales.index, daily_sales.values, color='blue', linewidth=2)
        ax.set_title('Sales Trend Over Time', fontsize=14, fontweight='bold')
        ax.set_xlabel('Date')
        ax.set_ylabel('Daily Sales (₹)')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()

        return self._fig_to_base64(fig)

    def _create_risk_distribution_chart(self, recommendations):
        """Create risk level distribution pie chart"""
        fig, ax = plt.subplots(figsize=(8, 8))

        risk_counts = {}
        for rec in recommendations:
            risk_level = rec.get('risk_level', 'unknown')
            risk_counts[risk_level] = risk_counts.get(risk_level, 0) + 1

        colors = {
            'critical': '#d32f2f',
            'high': '#ff6f00',
            'medium': '#fbc02d',
            'low': '#388e3c'
        }
        chart_colors = [colors.get(level, '#757575') for level in risk_counts.keys()]

        ax.pie(risk_counts.values(), labels=risk_counts.keys(), autopct='%1.1f%%',
               colors=chart_colors, startangle=90)
        ax.set_title('Risk Level Distribution', fontsize=14, fontweight='bold')

        return self._fig_to_base64(fig)

    def _create_high_risk_products_chart(self, recommendations):
        """Create chart for high-risk products"""
        fig, ax = plt.subplots(figsize=(12, 6))

        # Filter high and critical risk products
        high_risk = [r for r in recommendations if r.get('risk_level') in ['high', 'critical']][:10]

        if not high_risk:
            ax.text(0.5, 0.5, 'No high-risk products found', ha='center', va='center')
            return self._fig_to_base64(fig)

        products = [r['product_name'][:30] for r in high_risk]  # Truncate long names
        scores = [r.get('risk_score', 0) for r in high_risk]
        colors_list = ['#d32f2f' if r.get('risk_level') == 'critical' else '#ff6f00' for r in high_risk]

        ax.barh(products, scores, color=colors_list)
        ax.set_title('Top 10 High-Risk Products', fontsize=14, fontweight='bold')
        ax.set_xlabel('Risk Score')
        ax.set_ylabel('Product')
        plt.tight_layout()

        return self._fig_to_base64(fig)

    def _create_demand_trend_chart(self, recommendations):
        """Create demand trend distribution chart"""
        fig, ax = plt.subplots(figsize=(8, 6))

        trend_counts = {}
        for rec in recommendations:
            trend = rec.get('demand_trend', 'unknown')
            trend_counts[trend] = trend_counts.get(trend, 0) + 1

        colors_map = {
            'increasing': '#4caf50',
            'decreasing': '#f44336',
            'stable': '#2196f3'
        }
        chart_colors = [colors_map.get(trend, '#757575') for trend in trend_counts.keys()]

        ax.bar(trend_counts.keys(), trend_counts.values(), color=chart_colors)
        ax.set_title('Demand Trend Distribution', fontsize=14, fontweight='bold')
        ax.set_xlabel('Trend')
        ax.set_ylabel('Number of Products')
        plt.tight_layout()

        return self._fig_to_base64(fig)

    def _fig_to_base64(self, fig):
        """Convert matplotlib figure to base64 string"""
        buffer = BytesIO()
        fig.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
        buffer.seek(0)
        img_base64 = base64.b64encode(buffer.read()).decode('utf-8')
        plt.close(fig)
        return f"data:image/png;base64,{img_base64}"

    def _generate_eda_insights(self, df):
        """Generate key insights from EDA"""
        insights = []

        # Top selling category
        if 'product_category' in df.columns and 'sales' in df.columns:
            top_category = df.groupby('product_category')['sales'].sum().idxmax()
            top_category_sales = df.groupby('product_category')['sales'].sum().max()
            insights.append(f"Top selling category: {top_category} (₹{top_category_sales:,.2f})")

        # Most ordered product
        if 'product_name' in df.columns and 'orderQty' in df.columns:
            top_product = df.groupby('product_name')['orderQty'].sum().idxmax()
            insights.append(f"Most ordered product: {top_product}")

        # Average profit margin
        if 'profit' in df.columns and 'sales' in df.columns:
            total_profit = df['profit'].sum()
            total_sales = df['sales'].sum()
            margin = (total_profit / total_sales * 100) if total_sales > 0 else 0
            insights.append(f"Average profit margin: {margin:.2f}%")

        return insights

    def _generate_optimization_insights(self, recommendations, summary):
        """Generate key insights from optimization"""
        insights = []

        # Risk summary
        critical_count = summary.get('critical_risk_products', 0)
        high_count = summary.get('high_risk_products', 0)
        if critical_count > 0:
            insights.append(f"⚠️ {critical_count} products require immediate attention (Critical Risk)")
        if high_count > 0:
            insights.append(f"⚡ {high_count} products are at high risk of stockout")

        # Top risk products
        high_risk_products = [r for r in recommendations if r.get('risk_level') in ['critical', 'high']][:3]
        if high_risk_products:
            product_names = ', '.join([r['product_name'] for r in high_risk_products])
            insights.append(f"Top priority products: {product_names}")

        # ML model info
        if summary.get('ml_model_used'):
            model_name = summary['ml_model_used']
            r2_score = summary.get('ml_model_r2_score', 0)
            insights.append(f"Predictions powered by {model_name} (R² Score: {r2_score:.3f})")

        # Enhanced analysis
        if summary.get('enhanced_analysis_enabled'):
            insights.append("✓ Enhanced risk analysis enabled with external market data")

        return insights

    def _create_eda_html(self, report_data):
        """Create HTML content for EDA report"""
        stats = report_data['summary_stats']
        charts = report_data['charts']
        insights = report_data['insights']

        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>{report_data['title']}</title>
            <style>
                body {{
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                    margin: 20px;
                    background-color: #f5f5f5;
                }}
                .container {{
                    max-width: 1200px;
                    margin: 0 auto;
                    background: white;
                    padding: 30px;
                    border-radius: 8px;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                }}
                h1 {{
                    color: #1976d2;
                    border-bottom: 3px solid #1976d2;
                    padding-bottom: 10px;
                }}
                h2 {{
                    color: #424242;
                    margin-top: 30px;
                }}
                .stats-grid {{
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
                    gap: 20px;
                    margin: 20px 0;
                }}
                .stat-card {{
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    color: white;
                    padding: 20px;
                    border-radius: 8px;
                    box-shadow: 0 2px 8px rgba(0,0,0,0.1);
                }}
                .stat-card h3 {{
                    margin: 0;
                    font-size: 14px;
                    opacity: 0.9;
                }}
                .stat-card .value {{
                    font-size: 28px;
                    font-weight: bold;
                    margin-top: 10px;
                }}
                .chart {{
                    margin: 20px 0;
                    text-align: center;
                }}
                .chart img {{
                    max-width: 100%;
                    border-radius: 8px;
                    box-shadow: 0 2px 8px rgba(0,0,0,0.1);
                }}
                .insights {{
                    background: #e3f2fd;
                    padding: 20px;
                    border-radius: 8px;
                    border-left: 4px solid #1976d2;
                }}
                .insights li {{
                    margin: 10px 0;
                    font-size: 16px;
                }}
                .footer {{
                    margin-top: 40px;
                    text-align: center;
                    color: #757575;
                    font-size: 12px;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>{report_data['title']}</h1>
                <p><strong>Generated:</strong> {report_data['generated_at']}</p>

                <h2>Summary Statistics</h2>
                <div class="stats-grid">
                    <div class="stat-card">
                        <h3>Total Orders</h3>
                        <div class="value">{stats['total_orders']:,}</div>
                    </div>
                    <div class="stat-card">
                        <h3>Total Products</h3>
                        <div class="value">{stats['total_products']:,}</div>
                    </div>
                    <div class="stat-card">
                        <h3>Total Revenue</h3>
                        <div class="value">₹{stats['total_revenue']:,.2f}</div>
                    </div>
                    <div class="stat-card">
                        <h3>Total Profit</h3>
                        <div class="value">₹{stats['total_profit']:,.2f}</div>
                    </div>
                </div>

                <h2>Key Insights</h2>
                <div class="insights">
                    <ul>
                        {''.join([f'<li>{insight}</li>' for insight in insights])}
                    </ul>
                </div>

                <h2>Data Visualizations</h2>
        """

        # Add charts
        for chart_name, chart_data in charts.items():
            html += f"""
                <div class="chart">
                    <img src="{chart_data}" alt="{chart_name}">
                </div>
            """

        html += """
                <div class="footer">
                    <p>Generated by Inventory Optimization System | Hack Revolution 2025</p>
                </div>
            </div>
        </body>
        </html>
        """

        return html

    def _create_optimization_html(self, report_data):
        """Create HTML content for optimization report"""
        summary = report_data['summary']
        recommendations = report_data['recommendations']
        charts = report_data['charts']
        insights = report_data['insights']

        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>{report_data['title']}</title>
            <style>
                body {{
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                    margin: 20px;
                    background-color: #f5f5f5;
                }}
                .container {{
                    max-width: 1400px;
                    margin: 0 auto;
                    background: white;
                    padding: 30px;
                    border-radius: 8px;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                }}
                h1 {{
                    color: #1976d2;
                    border-bottom: 3px solid #1976d2;
                    padding-bottom: 10px;
                }}
                h2 {{
                    color: #424242;
                    margin-top: 30px;
                }}
                .stats-grid {{
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                    gap: 20px;
                    margin: 20px 0;
                }}
                .stat-card {{
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    color: white;
                    padding: 20px;
                    border-radius: 8px;
                    text-align: center;
                }}
                .stat-card.critical {{
                    background: linear-gradient(135deg, #d32f2f 0%, #c62828 100%);
                }}
                .stat-card.high {{
                    background: linear-gradient(135deg, #ff6f00 0%, #e65100 100%);
                }}
                .stat-card.medium {{
                    background: linear-gradient(135deg, #fbc02d 0%, #f9a825 100%);
                }}
                .stat-card.low {{
                    background: linear-gradient(135deg, #388e3c 0%, #2e7d32 100%);
                }}
                .stat-card h3 {{
                    margin: 0;
                    font-size: 14px;
                    opacity: 0.9;
                }}
                .stat-card .value {{
                    font-size: 32px;
                    font-weight: bold;
                    margin-top: 10px;
                }}
                .chart {{
                    margin: 20px 0;
                    text-align: center;
                }}
                .chart img {{
                    max-width: 100%;
                    border-radius: 8px;
                    box-shadow: 0 2px 8px rgba(0,0,0,0.1);
                }}
                .insights {{
                    background: #fff3e0;
                    padding: 20px;
                    border-radius: 8px;
                    border-left: 4px solid #ff6f00;
                }}
                .insights li {{
                    margin: 10px 0;
                    font-size: 16px;
                }}
                table {{
                    width: 100%;
                    border-collapse: collapse;
                    margin-top: 20px;
                }}
                th, td {{
                    padding: 12px;
                    text-align: left;
                    border-bottom: 1px solid #ddd;
                }}
                th {{
                    background-color: #1976d2;
                    color: white;
                    font-weight: bold;
                }}
                tr:hover {{
                    background-color: #f5f5f5;
                }}
                .risk-badge {{
                    padding: 4px 12px;
                    border-radius: 12px;
                    font-size: 12px;
                    font-weight: bold;
                    color: white;
                }}
                .risk-critical {{
                    background-color: #d32f2f;
                }}
                .risk-high {{
                    background-color: #ff6f00;
                }}
                .risk-medium {{
                    background-color: #fbc02d;
                }}
                .risk-low {{
                    background-color: #388e3c;
                }}
                .footer {{
                    margin-top: 40px;
                    text-align: center;
                    color: #757575;
                    font-size: 12px;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>{report_data['title']}</h1>
                <p><strong>Generated:</strong> {report_data['generated_at']}</p>
                {f'<p><strong>Location:</strong> {summary.get("user_location", "N/A")}, {summary.get("user_state", "N/A")}</p>' if summary.get('user_state') else ''}

                <h2>Risk Summary</h2>
                <div class="stats-grid">
                    <div class="stat-card critical">
                        <h3>Critical Risk</h3>
                        <div class="value">{summary.get('critical_risk_products', 0)}</div>
                    </div>
                    <div class="stat-card high">
                        <h3>High Risk</h3>
                        <div class="value">{summary.get('high_risk_products', 0)}</div>
                    </div>
                    <div class="stat-card medium">
                        <h3>Medium Risk</h3>
                        <div class="value">{summary.get('medium_risk_products', 0)}</div>
                    </div>
                    <div class="stat-card low">
                        <h3>Low Risk</h3>
                        <div class="value">{summary.get('low_risk_products', 0)}</div>
                    </div>
                </div>

                <h2>Key Insights</h2>
                <div class="insights">
                    <ul>
                        {''.join([f'<li>{insight}</li>' for insight in insights])}
                    </ul>
                </div>

                <h2>Visualizations</h2>
        """

        # Add charts
        for chart_name, chart_data in charts.items():
            html += f"""
                <div class="chart">
                    <img src="{chart_data}" alt="{chart_name}">
                </div>
            """

        # Add recommendations table
        html += """
                <h2>Top Recommendations</h2>
                <table>
                    <thead>
                        <tr>
                            <th>Product</th>
                            <th>Category</th>
                            <th>Risk Level</th>
                            <th>Trend</th>
                            <th>Current Demand</th>
                            <th>Predicted Demand</th>
                            <th>Optimal Qty</th>
                            <th>Reorder Point</th>
                        </tr>
                    </thead>
                    <tbody>
        """

        # Show top 20 recommendations
        for rec in recommendations[:20]:
            risk_class = f"risk-{rec.get('risk_level', 'low')}"
            html += f"""
                        <tr>
                            <td>{rec.get('product_name', 'N/A')}</td>
                            <td>{rec.get('category', 'N/A')}</td>
                            <td><span class="risk-badge {risk_class}">{rec.get('risk_level', 'low').upper()}</span></td>
                            <td>{rec.get('demand_trend', 'stable')}</td>
                            <td>{rec.get('current_avg_demand', 0):.2f}</td>
                            <td>{rec.get('predicted_demand', 0):.2f}</td>
                            <td>{rec.get('optimal_order_qty', 0)}</td>
                            <td>{rec.get('reorder_point', 0):.2f}</td>
                        </tr>
            """

        html += """
                    </tbody>
                </table>

                <div class="footer">
                    <p>Generated by Inventory Optimization System | Hack Revolution 2025</p>
                </div>
            </div>
        </body>
        </html>
        """

        return html


# ==================== TESTING ====================
if __name__ == "__main__":
    # Test report generation
    print("\n" + "="*60)
    print("Testing Report Generator")
    print("="*60 + "\n")

    generator = ReportGenerator()

    # Test with sample data
    sample_df = pd.DataFrame({
        'product_name': ['Product A', 'Product B', 'Product C'] * 10,
        'product_category': ['Electronics', 'Clothing', 'Food'] * 10,
        'orderQty': np.random.randint(1, 50, 30),
        'sales': np.random.uniform(100, 1000, 30),
        'profit': np.random.uniform(10, 200, 30),
        'orderDate': pd.date_range(start='2024-01-01', periods=30, freq='D')
    })

    eda_report = generator.generate_eda_report(sample_df)
    print(f"EDA Report generated: {eda_report['filename']}")
    print(f"Report size: {len(eda_report['content'])} bytes")

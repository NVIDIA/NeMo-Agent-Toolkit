"""
Data visualization tools for retail sales analysis.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Optional
from pydantic import Field
import io
import base64

from aiq.builder.builder import Builder
from aiq.cli.register_workflow import register_function
from aiq.data_models.function import FunctionBaseConfig


class VisualizeSalesTrendConfig(FunctionBaseConfig, name="visualize_sales_trend"):
    """Visualize sales trend over time."""
    start_date: str = Field(description="Start date in YYYY-MM-DD format")
    end_date: str = Field(description="End date in YYYY-MM-DD format")
    product_name: Optional[str] = Field(
        default=None, description="Optional product name to filter by"
    )


@register_function(config_type=VisualizeSalesTrendConfig)
async def visualize_sales_trend_function(config: VisualizeSalesTrendConfig, builder: Builder):
    """Create a visualization of sales trends over time."""
    
    async def _visualize_sales_trend(start_date: str, end_date: str, product_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Create a line chart showing sales trends over time.
        
        Args:
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            product_name: Optional product name to filter by
            
        Returns:
            Dictionary containing chart data and image
        """
        # Generate sample data
        date_range = pd.date_range(start=start_date, end=end_date, freq='D')
        np.random.seed(42)
        
        # Create sample sales data with trend
        sales_data = []
        base_sales = 1000
        for i, date in enumerate(date_range):
            # Add trend, seasonality, and noise
            trend = i * 2  # Small upward trend
            seasonality = 200 * np.sin(2 * np.pi * i / 7)  # Weekly seasonality
            noise = np.random.normal(0, 100)
            sales = max(0, base_sales + trend + seasonality + noise)
            
            sales_data.append({
                'date': date,
                'sales': sales,
                'product': product_name or 'All Products'
            })
        
        df = pd.DataFrame(sales_data)
        
        # Create the plot
        plt.figure(figsize=(12, 6))
        plt.plot(df['date'], df['sales'], linewidth=2, marker='o', markersize=4)
        title = f'Sales Trend: {start_date} to {end_date}'
        if product_name:
            title += f' - {product_name}'
        plt.title(title)
        plt.xlabel('Date')
        plt.ylabel('Sales Amount')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Save plot to base64 string
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode()
        plt.close()
        
        return {
            'chart_type': 'sales_trend',
            'date_range': f"{start_date} to {end_date}",
            'product_name': product_name or 'All Products',
            'data_points': len(sales_data),
            'average_sales': round(df['sales'].mean(), 2),
            'total_sales': round(df['sales'].sum(), 2),
            'image_base64': image_base64,
            'chart_data': sales_data[:10]  # Return first 10 data points
        }
    
    yield _visualize_sales_trend


class CompareStoreTrendsConfig(FunctionBaseConfig, name="compare_store_trends_visually"):
    """Compare sales trends between different stores."""
    store_names: List[str] = Field(description="List of store names to compare")
    start_date: str = Field(description="Start date in YYYY-MM-DD format")
    end_date: str = Field(description="End date in YYYY-MM-DD format")


@register_function(config_type=CompareStoreTrendsConfig)
async def compare_store_trends_visually_function(config: CompareStoreTrendsConfig, builder: Builder):
    """Create a visualization comparing sales trends between stores."""
    
    async def _compare_store_trends_visually(store_names: List[str], start_date: str, end_date: str) -> Dict[str, Any]:
        """
        Create a multi-line chart comparing sales trends between stores.
        
        Args:
            store_names: List of store names to compare
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            
        Returns:
            Dictionary containing comparison chart data and image
        """
        date_range = pd.date_range(start=start_date, end=end_date, freq='D')
        all_data = []
        
        # Generate sample data for each store
        for store_idx, store_name in enumerate(store_names):
            np.random.seed(42 + store_idx)  # Different seed for each store
            base_sales = 800 + store_idx * 200  # Different base sales for each store
            
            for i, date in enumerate(date_range):
                trend = i * (1 + store_idx * 0.5)  # Different trends
                seasonality = (150 + store_idx * 50) * np.sin(2 * np.pi * i / 7)
                noise = np.random.normal(0, 80)
                sales = max(0, base_sales + trend + seasonality + noise)
                
                all_data.append({
                    'date': date,
                    'sales': sales,
                    'store': store_name
                })
        
        df = pd.DataFrame(all_data)
        
        # Create the comparison plot
        plt.figure(figsize=(14, 8))
        for store in store_names:
            store_data = df[df['store'] == store]
            plt.plot(
                store_data['date'], store_data['sales'], 
                linewidth=2, marker='o', markersize=3, label=store
            )
        
        plt.title(f'Store Sales Comparison: {start_date} to {end_date}')
        plt.xlabel('Date')
        plt.ylabel('Sales Amount')
        plt.legend()
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Save plot to base64 string
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode()
        plt.close()
        
        # Calculate summary statistics
        store_summaries = []
        for store in store_names:
            store_data = df[df['store'] == store]
            store_summaries.append({
                'store_name': store,
                'average_sales': round(store_data['sales'].mean(), 2),
                'total_sales': round(store_data['sales'].sum(), 2),
                'max_sales': round(store_data['sales'].max(), 2),
                'min_sales': round(store_data['sales'].min(), 2)
            })
        
        return {
            'chart_type': 'store_comparison',
            'stores_compared': store_names,
            'date_range': f"{start_date} to {end_date}",
            'image_base64': image_base64,
            'store_summaries': store_summaries
        }
    
    yield _compare_store_trends_visually


class PlotAverageDailyRevenueConfig(FunctionBaseConfig, name="plot_average_daily_revenue"):
    """Plot average daily revenue by day of week."""
    start_date: str = Field(description="Start date in YYYY-MM-DD format")
    end_date: str = Field(description="End date in YYYY-MM-DD format")


@register_function(config_type=PlotAverageDailyRevenueConfig)
async def plot_average_daily_revenue_function(config: PlotAverageDailyRevenueConfig, builder: Builder):
    """Create a bar chart showing average daily revenue by day of week."""
    
    async def _plot_average_daily_revenue(start_date: str, end_date: str) -> Dict[str, Any]:
        """
        Create a bar chart showing average revenue by day of the week.
        
        Args:
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            
        Returns:
            Dictionary containing revenue chart data and image
        """
        date_range = pd.date_range(start=start_date, end=end_date, freq='D')
        np.random.seed(42)
        
        # Generate sample revenue data
        revenue_data = []
        for date in date_range:
            day_of_week = date.strftime('%A')
            # Weekend typically has higher sales
            base_revenue = 1200 if day_of_week in ['Saturday', 'Sunday'] else 800
            noise = np.random.normal(0, 150)
            revenue = max(0, base_revenue + noise)
            
            revenue_data.append({
                'date': date,
                'day_of_week': day_of_week,
                'revenue': revenue
            })
        
        df = pd.DataFrame(revenue_data)
        
        # Calculate average revenue by day of week
        avg_revenue_by_day = df.groupby('day_of_week')['revenue'].mean().reindex([
            'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'
        ])
        
        # Create the bar chart
        plt.figure(figsize=(10, 6))
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2']
        bars = plt.bar(avg_revenue_by_day.index, avg_revenue_by_day.values, color=colors)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            plt.text(
                bar.get_x() + bar.get_width()/2., height + 10,
                f'${height:.0f}', ha='center', va='bottom'
            )
        
        title = f'Average Daily Revenue by Day of Week ({start_date} to {end_date})'
        plt.title(title)
        plt.xlabel('Day of Week')
        plt.ylabel('Average Revenue ($)')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        
        # Save plot to base64 string
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode()
        plt.close()
        
        return {
            'chart_type': 'average_daily_revenue',
            'date_range': f"{start_date} to {end_date}",
            'avg_revenue_by_day': avg_revenue_by_day.to_dict(),
            'highest_day': avg_revenue_by_day.idxmax(),
            'lowest_day': avg_revenue_by_day.idxmin(),
            'overall_average': round(avg_revenue_by_day.mean(), 2),
            'image_base64': image_base64
        }
    
    yield _plot_average_daily_revenue


class GraphSummarizerConfig(FunctionBaseConfig, name="graph_summarizer"):
    """Analyze and summarize chart data."""
    chart_data: Dict[str, Any] = Field(description="Chart data to summarize")
    chart_type: str = Field(description="Type of chart (e.g., 'line', 'bar', 'comparison')")


@register_function(config_type=GraphSummarizerConfig)
async def graph_summarizer_function(config: GraphSummarizerConfig, builder: Builder):
    """Analyze chart data and provide natural language summaries."""
    
    async def _graph_summarizer(chart_data: Dict[str, Any], chart_type: str) -> Dict[str, Any]:
        """
        Analyze chart data and provide insights and summaries.
        
        Args:
            chart_data: The chart data to analyze
            chart_type: Type of chart being analyzed
            
        Returns:
            Dictionary containing analysis and insights
        """
        insights = []
        summary = ""
        
        if chart_type == "sales_trend":
            if 'average_sales' in chart_data and 'total_sales' in chart_data:
                avg_sales = chart_data['average_sales']
                total_sales = chart_data['total_sales']
                
                insights.append(f"Average daily sales: ${avg_sales:,.2f}")
                insights.append(f"Total sales for period: ${total_sales:,.2f}")
                
                if avg_sales > 1000:
                    insights.append("Sales performance is above average")
                else:
                    insights.append("Sales performance may need improvement")
                
                summary = f"The sales trend shows an average of ${avg_sales:,.2f} per day over the analyzed period."
        
        elif chart_type == "store_comparison":
            if 'store_summaries' in chart_data:
                summaries = chart_data['store_summaries']
                best_store = max(summaries, key=lambda x: x['average_sales'])
                worst_store = min(summaries, key=lambda x: x['average_sales'])
                
                best_msg = f"Best performing store: {best_store['store_name']} (${best_store['average_sales']:,.2f}/day)"
                worst_msg = f"Lowest performing store: {worst_store['store_name']} (${worst_store['average_sales']:,.2f}/day)"
                insights.append(best_msg)
                insights.append(worst_msg)
                
                performance_gap = best_store['average_sales'] - worst_store['average_sales']
                insights.append(f"Performance gap: ${performance_gap:,.2f}/day")
                
                summary = (f"Store comparison shows {best_store['store_name']} leading with "
                          f"${best_store['average_sales']:,.2f} average daily sales.")
        
        elif chart_type == "average_daily_revenue":
            if 'avg_revenue_by_day' in chart_data:
                revenue_data = chart_data['avg_revenue_by_day']
                highest_day = chart_data.get('highest_day', 'Unknown')
                lowest_day = chart_data.get('lowest_day', 'Unknown')
                
                insights.append(f"Highest revenue day: {highest_day}")
                insights.append(f"Lowest revenue day: {lowest_day}")
                
                weekend_avg = (revenue_data.get('Saturday', 0) + revenue_data.get('Sunday', 0)) / 2
                weekday_avg = sum(revenue_data.get(day, 0) for day in ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']) / 5
                
                if weekend_avg > weekday_avg:
                    insights.append("Weekend sales outperform weekday sales")
                else:
                    insights.append("Weekday sales are stronger than weekend sales")
                
                summary = f"Revenue analysis shows {highest_day} as the strongest day and {lowest_day} as the weakest."
        
        return {
            'chart_type': chart_type,
            'summary': summary,
            'key_insights': insights,
            'recommendations': [
                "Monitor trends regularly for early detection of changes",
                "Focus marketing efforts on underperforming periods",
                "Consider promotional strategies for low-performance days"
            ],
            'data_quality': "Good" if len(insights) > 2 else "Limited"
        }
    
    yield _graph_summarizer 
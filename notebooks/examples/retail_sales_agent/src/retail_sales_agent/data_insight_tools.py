"""
Data insight tools for retail sales analysis.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any
from pydantic import Field

from aiq.builder.builder import Builder
from aiq.cli.register_workflow import register_function
from aiq.data_models.function import FunctionBaseConfig


class GetDailySalesByProductConfig(FunctionBaseConfig, name="get_daily_sales_by_product"):
    """Get daily sales data by product."""
    product_name: str = Field(description="Name of the product to analyze")
    start_date: str = Field(description="Start date in YYYY-MM-DD format")
    end_date: str = Field(description="End date in YYYY-MM-DD format")


@register_function(config_type=GetDailySalesByProductConfig)
async def get_daily_sales_by_product_function(config: GetDailySalesByProductConfig, builder: Builder):
    """Get daily sales data for a specific product."""
    
    async def _get_daily_sales_by_product(product_name: str, start_date: str, end_date: str) -> Dict[str, Any]:
        """
        Retrieve daily sales data for a specific product within a date range.
        
        Args:
            product_name: Name of the product
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            
        Returns:
            Dictionary containing daily sales data
        """
        # This would typically connect to a real database
        # For now, return sample data
        date_range = pd.date_range(start=start_date, end=end_date, freq='D')
        
        # Generate sample sales data
        np.random.seed(42)  # For reproducible results
        daily_sales = []
        
        for date in date_range:
            sales_amount = np.random.randint(100, 1000)
            quantity_sold = np.random.randint(10, 100)
            daily_sales.append({
                'date': date.strftime('%Y-%m-%d'),
                'product_name': product_name,
                'sales_amount': sales_amount,
                'quantity_sold': quantity_sold
            })
        
        return {
            'product_name': product_name,
            'date_range': f"{start_date} to {end_date}",
            'daily_sales': daily_sales,
            'total_sales': sum(item['sales_amount'] for item in daily_sales),
            'total_quantity': sum(item['quantity_sold'] for item in daily_sales)
        }
    
    yield _get_daily_sales_by_product


class GetTotalSalesPerDayConfig(FunctionBaseConfig, name="get_total_sales_per_day"):
    """Get total sales across all products per day."""
    start_date: str = Field(description="Start date in YYYY-MM-DD format")
    end_date: str = Field(description="End date in YYYY-MM-DD format")


@register_function(config_type=GetTotalSalesPerDayConfig)
async def get_total_sales_per_day_function(config: GetTotalSalesPerDayConfig, builder: Builder):
    """Get total sales across all products per day."""
    
    async def _get_total_sales_per_day(start_date: str, end_date: str) -> Dict[str, Any]:
        """
        Retrieve total sales data across all products for each day in the date range.
        
        Args:
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            
        Returns:
            Dictionary containing daily total sales data
        """
        date_range = pd.date_range(start=start_date, end=end_date, freq='D')
        
        # Generate sample total sales data
        np.random.seed(42)
        daily_totals = []
        
        for date in date_range:
            total_sales = np.random.randint(5000, 25000)
            total_transactions = np.random.randint(50, 300)
            daily_totals.append({
                'date': date.strftime('%Y-%m-%d'),
                'total_sales': total_sales,
                'total_transactions': total_transactions,
                'average_transaction_value': round(total_sales / total_transactions, 2)
            })
        
        return {
            'date_range': f"{start_date} to {end_date}",
            'daily_totals': daily_totals,
            'period_total_sales': sum(item['total_sales'] for item in daily_totals),
            'period_total_transactions': sum(item['total_transactions'] for item in daily_totals)
        }
    
    yield _get_total_sales_per_day


class DetectOutliersIQRConfig(FunctionBaseConfig, name="detect_outliers_iqr"):
    """Detect outliers in sales data using IQR method."""
    data_source: str = Field(description="Source of data (e.g., 'daily_sales', 'product_sales')")
    metric: str = Field(description="Metric to analyze for outliers (e.g., 'sales_amount', 'quantity')")


@register_function(config_type=DetectOutliersIQRConfig)
async def detect_outliers_iqr_function(config: DetectOutliersIQRConfig, builder: Builder):
    """Detect outliers in sales data using the Interquartile Range (IQR) method."""
    
    async def _detect_outliers_iqr(data_source: str, metric: str) -> Dict[str, Any]:
        """
        Detect outliers in sales data using the IQR method.
        
        Args:
            data_source: Source of data to analyze
            metric: Specific metric to check for outliers
            
        Returns:
            Dictionary containing outlier analysis results
        """
        # Generate sample data for outlier detection
        np.random.seed(42)
        sample_data = np.random.normal(1000, 200, 100)  # Normal distribution with some outliers
        
        # Add some outliers
        outlier_indices = np.random.choice(100, 5, replace=False)
        for idx in outlier_indices:
            sample_data[idx] = np.random.choice([2500, 3000, 200, 150])  # Add high and low outliers
        
        # Calculate IQR
        q1 = np.percentile(sample_data, 25)
        q3 = np.percentile(sample_data, 75)
        iqr = q3 - q1
        
        # Define outlier bounds
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        
        # Identify outliers
        outliers = []
        for i, value in enumerate(sample_data):
            if value < lower_bound or value > upper_bound:
                outliers.append({
                    'index': i,
                    'value': round(value, 2),
                    'type': 'high' if value > upper_bound else 'low'
                })
        
        return {
            'data_source': data_source,
            'metric': metric,
            'total_data_points': len(sample_data),
            'outliers_found': len(outliers),
            'outlier_percentage': round((len(outliers) / len(sample_data)) * 100, 2),
            'statistics': {
                'q1': round(q1, 2),
                'q3': round(q3, 2),
                'iqr': round(iqr, 2),
                'lower_bound': round(lower_bound, 2),
                'upper_bound': round(upper_bound, 2),
                'mean': round(np.mean(sample_data), 2),
                'std': round(np.std(sample_data), 2)
            },
            'outliers': outliers[:10]  # Return first 10 outliers
        }
    
    yield _detect_outliers_iqr 
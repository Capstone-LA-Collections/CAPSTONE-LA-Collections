"""
Dashboard Data Loader
Centralized data loading functions for dashboard components

This module provides easy-to-use functions to load transformed dimensional data
for dashboard visualization and analytics.
"""

import pandas as pd
import os
import sys
from datetime import datetime, timedelta
import numpy as np

# Add parent directories to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def get_data_path():
    """Get the path to transformed data directory"""
    return os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'Transformed')

class DashboardDataLoader:
    """
    Main data loader class for dashboard components
    Provides clean, ready-to-use data for visualizations
    """
    
    def __init__(self):
        self.data_path = get_data_path()
        self._cache = {}
        print(f"📊 Dashboard Data Loader initialized")
        print(f"📁 Data path: {self.data_path}")
    
    def load_dimensional_data(self, refresh_cache=False):
        """
        Load all dimensional tables with proper data types and relationships
        
        Args:
            refresh_cache (bool): Force reload data from files
            
        Returns:
            dict: Dictionary containing all dimensional tables
        """
        if 'dimensional_data' in self._cache and not refresh_cache:
            print("📋 Using cached dimensional data")
            return self._cache['dimensional_data']
        
        print("📥 Loading dimensional data...")
        data = {}
        
        # Load dimension tables
        tables = {
            'dim_time': 'dim_time.csv',
            'dim_product': 'dim_product.csv', 
            'dim_product_variant': 'dim_product_variant.csv',
            'dim_order': 'dim_order.csv',
            'dim_customer': 'dim_customer.csv',
            'fact_orders': 'fact_orders.csv',
            'fact_sales_aggregate': 'fact_sales_aggregate.csv'
        }
        
        for table_name, filename in tables.items():
            file_path = os.path.join(self.data_path, filename)
            if os.path.exists(file_path):
                try:
                    df = pd.read_csv(file_path)
                    
                    # Apply data type conversions
                    df = self._apply_data_types(df, table_name)
                    
                    data[table_name] = df
                    print(f"✅ Loaded {table_name}: {len(df):,} records")
                except Exception as e:
                    print(f"❌ Error loading {table_name}: {e}")
                    data[table_name] = pd.DataFrame()
            else:
                print(f"⚠️ File not found: {filename}")
                data[table_name] = pd.DataFrame()
        
        # Cache the data
        self._cache['dimensional_data'] = data
        print(f"💾 Cached dimensional data with {sum(len(df) for df in data.values()):,} total records")
        
        return data
    
    def _apply_data_types(self, df, table_name):
        """Apply proper data types to dimensional tables"""
        
        # Common date columns
        date_columns = ['order_date', 'updated_at', 'created_at', 'last_updated']
        for col in date_columns:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')
        
        # Table-specific conversions
        if table_name == 'dim_time':
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
        
        elif table_name in ['dim_product', 'dim_product_variant']:
            price_cols = ['current_price', 'original_price']
            for col in price_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
        
        elif table_name == 'dim_order':
            numeric_cols = ['price_total', 'insurance_premium_and_fees', 'total_item_count']
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
        
        elif table_name in ['fact_orders', 'fact_sales_aggregate']:
            numeric_cols = ['quantity', 'unit_price', 'paid_price', 'total_amount', 'voucher_amount', 'shipping_fee']
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
        
        return df
    
    def get_sales_summary(self, start_date=None, end_date=None, platform=None):
        """
        Get sales summary data for dashboard KPIs
        
        Args:
            start_date (str): Start date in YYYY-MM-DD format
            end_date (str): End date in YYYY-MM-DD format
            platform (str): 'Lazada' or 'Shopee' or None for both
            
        Returns:
            dict: Sales summary metrics
        """
        data = self.load_dimensional_data()
        sales_df = data['fact_sales_aggregate']
        
        if sales_df.empty:
            return {}
        
        # Apply filters
        filtered_df = sales_df.copy()
        
        if start_date:
            filtered_df = filtered_df[filtered_df['order_date'] >= start_date]
        if end_date:
            filtered_df = filtered_df[filtered_df['order_date'] <= end_date]
        if platform:
            platform_key = 1 if platform.lower() == 'lazada' else 2
            filtered_df = filtered_df[filtered_df['platform_key'] == platform_key]
        
        # Calculate metrics
        summary = {
            'total_revenue': filtered_df['total_amount'].sum(),
            'total_orders': filtered_df['orders_key'].nunique(),
            'total_items_sold': filtered_df['quantity'].sum(),
            'average_order_value': filtered_df['total_amount'].sum() / filtered_df['orders_key'].nunique() if len(filtered_df) > 0 else 0,
            'unique_customers': filtered_df['customer_key'].nunique(),
            'unique_products': filtered_df['product_key'].nunique(),
            'date_range': {
                'start': filtered_df['order_date'].min() if len(filtered_df) > 0 else None,
                'end': filtered_df['order_date'].max() if len(filtered_df) > 0 else None
            }
        }
        
        return summary
    
    def get_revenue_by_time(self, period='monthly', platform=None):
        """
        Get revenue trends by time period
        
        Args:
            period (str): 'daily', 'weekly', 'monthly', 'yearly'
            platform (str): 'Lazada' or 'Shopee' or None for both
            
        Returns:
            pd.DataFrame: Revenue by time period
        """
        data = self.load_dimensional_data()
        sales_df = data['fact_sales_aggregate']
        
        if sales_df.empty:
            return pd.DataFrame()
        
        # Apply platform filter
        if platform:
            platform_key = 1 if platform.lower() == 'lazada' else 2
            sales_df = sales_df[sales_df['platform_key'] == platform_key]
        
        # Group by time period
        if period == 'daily':
            grouped = sales_df.groupby('order_date')
        elif period == 'weekly':
            sales_df['week'] = sales_df['order_date'].dt.to_period('W')
            grouped = sales_df.groupby('week')
        elif period == 'monthly':
            sales_df['month'] = sales_df['order_date'].dt.to_period('M')
            grouped = sales_df.groupby('month')
        elif period == 'yearly':
            sales_df['year'] = sales_df['order_date'].dt.to_period('Y')
            grouped = sales_df.groupby('year')
        else:
            raise ValueError("Period must be 'daily', 'weekly', 'monthly', or 'yearly'")
        
        result = grouped.agg({
            'total_amount': 'sum',
            'orders_key': 'nunique',
            'quantity': 'sum'
        }).reset_index()
        
        result.columns = [period.rstrip('ly'), 'revenue', 'orders', 'items_sold']
        return result
    
    def get_product_performance(self, top_n=20, metric='revenue'):
        """
        Get top performing products by revenue or quantity
        
        Args:
            top_n (int): Number of top products to return
            metric (str): 'revenue' or 'quantity'
            
        Returns:
            pd.DataFrame: Top products with performance metrics
        """
        data = self.load_dimensional_data()
        sales_df = data['fact_sales_aggregate']
        products_df = data['dim_product']
        
        if sales_df.empty or products_df.empty:
            return pd.DataFrame()
        
        # Aggregate by product
        product_agg = sales_df.groupby('product_key').agg({
            'total_amount': 'sum',
            'quantity': 'sum',
            'orders_key': 'nunique'
        }).reset_index()
        
        # Join with product details
        result = product_agg.merge(
            products_df[['product_key', 'product_name', 'product_category', 'platform_key']], 
            on='product_key', 
            how='left'
        )
        
        # Sort by metric
        if metric == 'revenue':
            result = result.sort_values('total_amount', ascending=False)
        else:
            result = result.sort_values('quantity', ascending=False)
        
        # Add platform names
        result['platform'] = result['platform_key'].map({1: 'Lazada', 2: 'Shopee'})
        
        return result.head(top_n)
    
    def get_platform_comparison(self):
        """
        Get comparison metrics between Lazada and Shopee
        
        Returns:
            pd.DataFrame: Platform comparison metrics
        """
        data = self.load_dimensional_data()
        sales_df = data['fact_sales_aggregate']
        
        if sales_df.empty:
            return pd.DataFrame()
        
        comparison = sales_df.groupby('platform_key').agg({
            'total_amount': ['sum', 'mean'],
            'quantity': 'sum',
            'orders_key': 'nunique',
            'customer_key': 'nunique',
            'product_key': 'nunique'
        }).round(2)
        
        # Flatten column names
        comparison.columns = ['_'.join(col).strip() for col in comparison.columns]
        comparison = comparison.reset_index()
        
        # Add platform names
        comparison['platform'] = comparison['platform_key'].map({1: 'Lazada', 2: 'Shopee'})
        
        return comparison
    
    def get_customer_segments(self, top_n=10):
        """
        Get customer segmentation by purchase behavior
        
        Args:
            top_n (int): Number of top customers to return
            
        Returns:
            pd.DataFrame: Customer segments with metrics
        """
        data = self.load_dimensional_data()
        sales_df = data['fact_sales_aggregate']
        customers_df = data['dim_customer']
        
        if sales_df.empty or customers_df.empty:
            return pd.DataFrame()
        
        # Aggregate by customer
        customer_agg = sales_df.groupby('customer_key').agg({
            'total_amount': ['sum', 'mean', 'count'],
            'quantity': 'sum',
            'orders_key': 'nunique'
        }).round(2)
        
        # Flatten column names
        customer_agg.columns = ['total_spent', 'avg_order_value', 'total_transactions', 'total_items', 'unique_orders']
        customer_agg = customer_agg.reset_index()
        
        # Join with customer details
        result = customer_agg.merge(
            customers_df[['customer_key', 'customer_name', 'customer_email', 'shipping_city', 'platform_key']], 
            on='customer_key', 
            how='left'
        )
        
        # Add platform names
        result['platform'] = result['platform_key'].map({1: 'Lazada', 2: 'Shopee'})
        
        # Sort by total spent
        result = result.sort_values('total_spent', ascending=False)
        
        return result.head(top_n)
    
    def export_dashboard_data(self, output_dir=None):
        """
        Export prepared data files for dashboard consumption
        
        Args:
            output_dir (str): Output directory (defaults to Data_Prepared)
            
        Returns:
            dict: Paths to exported files
        """
        if output_dir is None:
            output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Data_Prepared')
        
        os.makedirs(output_dir, exist_ok=True)
        
        exported_files = {}
        
        try:
            # Export sales summary
            sales_summary = self.get_sales_summary()
            if sales_summary:
                import json
                with open(os.path.join(output_dir, 'sales_summary.json'), 'w') as f:
                    json.dump(sales_summary, f, indent=2, default=str)
                exported_files['sales_summary'] = os.path.join(output_dir, 'sales_summary.json')
            
            # Export revenue trends
            revenue_monthly = self.get_revenue_by_time('monthly')
            if not revenue_monthly.empty:
                revenue_monthly.to_csv(os.path.join(output_dir, 'revenue_by_month.csv'), index=False)
                exported_files['revenue_monthly'] = os.path.join(output_dir, 'revenue_by_month.csv')
            
            # Export product performance
            top_products = self.get_product_performance(50, 'revenue')
            if not top_products.empty:
                top_products.to_csv(os.path.join(output_dir, 'top_products.csv'), index=False)
                exported_files['top_products'] = os.path.join(output_dir, 'top_products.csv')
            
            # Export platform comparison
            platform_comparison = self.get_platform_comparison()
            if not platform_comparison.empty:
                platform_comparison.to_csv(os.path.join(output_dir, 'platform_comparison.csv'), index=False)
                exported_files['platform_comparison'] = os.path.join(output_dir, 'platform_comparison.csv')
            
            # Export customer segments
            customer_segments = self.get_customer_segments(100)
            if not customer_segments.empty:
                customer_segments.to_csv(os.path.join(output_dir, 'customer_segments.csv'), index=False)
                exported_files['customer_segments'] = os.path.join(output_dir, 'customer_segments.csv')
            
            print(f"✅ Exported {len(exported_files)} dashboard data files to: {output_dir}")
            
            return exported_files
            
        except Exception as e:
            print(f"❌ Error exporting dashboard data: {e}")
            return {}

# Convenience functions for direct usage
def load_dashboard_data():
    """Quick function to load all dimensional data"""
    loader = DashboardDataLoader()
    return loader.load_dimensional_data()

def get_quick_summary():
    """Quick function to get sales summary"""
    loader = DashboardDataLoader()
    return loader.get_sales_summary()

if __name__ == "__main__":
    # Test the data loader
    print("🧪 Testing Dashboard Data Loader...")
    
    loader = DashboardDataLoader()
    
    # Test loading dimensional data
    data = loader.load_dimensional_data()
    print(f"\n📊 Loaded {len(data)} tables:")
    for table_name, df in data.items():
        print(f"   - {table_name}: {len(df):,} records")
    
    # Test sales summary
    summary = loader.get_sales_summary()
    if summary:
        print(f"\n💰 Sales Summary:")
        print(f"   - Total Revenue: ₱{summary.get('total_revenue', 0):,.2f}")
        print(f"   - Total Orders: {summary.get('total_orders', 0):,}")
        print(f"   - Average Order Value: ₱{summary.get('average_order_value', 0):.2f}")
    
    # Export dashboard data
    exported = loader.export_dashboard_data()
    print(f"\n📁 Exported Files:")
    for name, path in exported.items():
        print(f"   - {name}: {os.path.basename(path)}")
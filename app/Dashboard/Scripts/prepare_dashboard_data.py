"""
Data Preparation Script for Dashboard
Processes and prepares all necessary data for dashboard consumption
"""

import os
import sys
import json
import pandas as pd
from datetime import datetime

# Add parent directories to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Data_Loading.dashboard_data_loader import DashboardDataLoader

def prepare_all_dashboard_data():
    """
    Comprehensive data preparation for dashboard
    Creates all necessary data files and summaries
    """
    print("🔄 Starting Dashboard Data Preparation...")
    print(f"⏰ Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Initialize data loader
    loader = DashboardDataLoader()
    
    # Load all dimensional data
    print("\n📥 Loading dimensional data...")
    data = loader.load_dimensional_data(refresh_cache=True)
    
    # Check data availability
    required_tables = ['fact_sales_aggregate', 'dim_product', 'dim_order', 'dim_customer']
    missing_tables = [table for table in required_tables if data[table].empty]
    
    if missing_tables:
        print(f"❌ Missing required tables: {missing_tables}")
        print("❌ Cannot proceed with dashboard preparation")
        return False
    
    print("✅ All required tables available")
    
    # Prepare output directory
    output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'Data_Prepared')
    os.makedirs(output_dir, exist_ok=True)
    
    prepared_files = {}
    
    try:
        # 1. Overall Sales Summary
        print("\n📊 Preparing sales summary...")
        sales_summary = loader.get_sales_summary()
        if sales_summary:
            import json
            summary_file = os.path.join(output_dir, 'sales_summary.json')
            with open(summary_file, 'w') as f:
                json.dump(sales_summary, f, indent=2, default=str)
            prepared_files['sales_summary'] = summary_file
            print(f"   ✅ Sales summary: ₱{sales_summary['total_revenue']:,.2f} revenue")
        
        # 2. Platform-specific summaries
        print("\n🏪 Preparing platform summaries...")
        for platform in ['Lazada', 'Shopee']:
            platform_summary = loader.get_sales_summary(platform=platform)
            if platform_summary:
                summary_file = os.path.join(output_dir, f'sales_summary_{platform.lower()}.json')
                with open(summary_file, 'w') as f:
                    json.dump(platform_summary, f, indent=2, default=str)
                prepared_files[f'sales_summary_{platform.lower()}'] = summary_file
                print(f"   ✅ {platform}: ₱{platform_summary['total_revenue']:,.2f} revenue")
        
        # 3. Revenue trends by different periods
        print("\n📈 Preparing revenue trends...")
        for period in ['daily', 'weekly', 'monthly', 'yearly']:
            try:
                revenue_trend = loader.get_revenue_by_time(period)
                if not revenue_trend.empty:
                    trend_file = os.path.join(output_dir, f'revenue_by_{period}.csv')
                    revenue_trend.to_csv(trend_file, index=False)
                    prepared_files[f'revenue_{period}'] = trend_file
                    print(f"   ✅ Revenue by {period}: {len(revenue_trend)} periods")
            except Exception as e:
                print(f"   ⚠️ Error preparing {period} trends: {e}")
        
        # 4. Product performance analytics
        print("\n🛍️ Preparing product analytics...")
        
        # Top products by revenue
        top_products_revenue = loader.get_product_performance(50, 'revenue')
        if not top_products_revenue.empty:
            products_file = os.path.join(output_dir, 'top_products_by_revenue.csv')
            top_products_revenue.to_csv(products_file, index=False)
            prepared_files['top_products_revenue'] = products_file
            print(f"   ✅ Top 50 products by revenue")
        
        # Top products by quantity
        top_products_quantity = loader.get_product_performance(50, 'quantity')
        if not top_products_quantity.empty:
            products_file = os.path.join(output_dir, 'top_products_by_quantity.csv')
            top_products_quantity.to_csv(products_file, index=False)
            prepared_files['top_products_quantity'] = products_file
            print(f"   ✅ Top 50 products by quantity")
        
        # 5. Platform comparison
        print("\n⚖️ Preparing platform comparison...")
        platform_comparison = loader.get_platform_comparison()
        if not platform_comparison.empty:
            comparison_file = os.path.join(output_dir, 'platform_comparison.csv')
            platform_comparison.to_csv(comparison_file, index=False)
            prepared_files['platform_comparison'] = comparison_file
            print(f"   ✅ Platform comparison analysis")
        
        # 6. Customer analytics
        print("\n👥 Preparing customer analytics...")
        customer_segments = loader.get_customer_segments(200)
        if not customer_segments.empty:
            customers_file = os.path.join(output_dir, 'customer_segments.csv')
            customer_segments.to_csv(customers_file, index=False)
            prepared_files['customer_segments'] = customers_file
            print(f"   ✅ Top 200 customer segments")
        
        # 7. Category analysis
        print("\n🏷️ Preparing category analysis...")
        sales_df = data['fact_sales_aggregate']
        products_df = data['dim_product']
        
        if not sales_df.empty and not products_df.empty:
            # Join sales with products for category analysis
            sales_with_category = sales_df.merge(
                products_df[['product_key', 'product_category', 'platform_key']], 
                on='product_key', 
                how='left'
            )
            
            category_analysis = sales_with_category.groupby(['product_category', 'platform_key']).agg({
                'total_amount': 'sum',
                'quantity': 'sum',
                'orders_key': 'nunique'
            }).reset_index()
            
            category_analysis['platform'] = category_analysis['platform_key'].map({1: 'Lazada', 2: 'Shopee'})
            category_analysis = category_analysis.sort_values('total_amount', ascending=False)
            
            category_file = os.path.join(output_dir, 'category_analysis.csv')
            category_analysis.to_csv(category_file, index=False)
            prepared_files['category_analysis'] = category_file
            print(f"   ✅ Category performance analysis")
        
        # 8. Time-based analytics
        print("\n⏰ Preparing time-based analytics...")
        
        # Monthly revenue with year-over-year comparison
        if not sales_df.empty:
            monthly_revenue = sales_df.copy()
            monthly_revenue['year'] = monthly_revenue['order_date'].dt.year
            monthly_revenue['month'] = monthly_revenue['order_date'].dt.month
            monthly_revenue['month_name'] = monthly_revenue['order_date'].dt.strftime('%B')
            
            monthly_summary = monthly_revenue.groupby(['year', 'month', 'month_name']).agg({
                'total_amount': 'sum',
                'orders_key': 'nunique',
                'quantity': 'sum'
            }).reset_index()
            
            monthly_file = os.path.join(output_dir, 'monthly_revenue_breakdown.csv')
            monthly_summary.to_csv(monthly_file, index=False)
            prepared_files['monthly_breakdown'] = monthly_file
            print(f"   ✅ Monthly revenue breakdown")
        
        # 9. Create data manifest
        print("\n📋 Creating data manifest...")
        manifest = {
            'generated_at': datetime.now().isoformat(),
            'total_files': len(prepared_files),
            'files': prepared_files,
            'data_summary': {
                'total_records': sum(len(df) for df in data.values()),
                'tables_loaded': list(data.keys()),
                'date_range': {
                    'start': str(sales_df['order_date'].min()) if not sales_df.empty else None,
                    'end': str(sales_df['order_date'].max()) if not sales_df.empty else None
                }
            }
        }
        
        manifest_file = os.path.join(output_dir, 'data_manifest.json')
        with open(manifest_file, 'w') as f:
            json.dump(manifest, f, indent=2)
        
        print(f"\n✅ Dashboard data preparation completed!")
        print(f"📁 Output directory: {output_dir}")
        print(f"📄 Files prepared: {len(prepared_files)}")
        print(f"📋 Data manifest: data_manifest.json")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during data preparation: {e}")
        import traceback
        traceback.print_exc()
        return False

def prepare_real_time_summary():
    """
    Quick preparation of real-time dashboard summary
    Creates a lightweight summary file for dashboard overview
    """
    print("⚡ Preparing real-time dashboard summary...")
    
    loader = DashboardDataLoader()
    
    try:
        # Load only the essential data
        data = loader.load_dimensional_data()
        sales_df = data['fact_sales_aggregate']
        
        if sales_df.empty:
            print("❌ No sales data available")
            return False
        
        # Calculate current metrics
        today = datetime.now().date()
        this_month_start = today.replace(day=1)
        last_30_days = today - pd.Timedelta(days=30)
        
        # Real-time summary
        summary = {
            'last_updated': datetime.now().isoformat(),
            'overall': loader.get_sales_summary(),
            'last_30_days': loader.get_sales_summary(start_date=str(last_30_days)),
            'this_month': loader.get_sales_summary(start_date=str(this_month_start)),
            'platform_breakdown': {
                'lazada': loader.get_sales_summary(platform='Lazada'),
                'shopee': loader.get_sales_summary(platform='Shopee')
            },
            'quick_stats': {
                'total_orders_count': len(data['dim_order']),
                'total_customers_count': len(data['dim_customer']),
                'total_products_count': len(data['dim_product']),
                'latest_order_date': str(sales_df['order_date'].max()),
                'data_freshness': 'Real-time'
            }
        }
        
        # Save real-time summary
        output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'Data_Prepared')
        os.makedirs(output_dir, exist_ok=True)
        
        summary_file = os.path.join(output_dir, 'realtime_summary.json')
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        print(f"✅ Real-time summary saved: {os.path.basename(summary_file)}")
        print(f"💰 Total Revenue: ₱{summary['overall']['total_revenue']:,.2f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error preparing real-time summary: {e}")
        return False

if __name__ == "__main__":
    print("🔄 Running Dashboard Data Preparation...")
    
    # Run full preparation
    success = prepare_all_dashboard_data()
    
    if success:
        # Also prepare real-time summary
        prepare_real_time_summary()
        print("\n🎉 All dashboard data preparation completed successfully!")
    else:
        print("\n❌ Dashboard data preparation failed!")
    
    print(f"⏰ Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
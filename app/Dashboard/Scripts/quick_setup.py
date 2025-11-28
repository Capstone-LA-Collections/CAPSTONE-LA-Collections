"""
Quick Dashboard Setup and Test Script
Run this to quickly prepare all dashboard data and test the data loader
"""

import os
import sys

# Add paths for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
dashboard_dir = os.path.dirname(current_dir)
app_dir = os.path.dirname(dashboard_dir)
root_dir = os.path.dirname(app_dir)

sys.path.append(root_dir)
sys.path.append(app_dir)
sys.path.append(dashboard_dir)

def run_dashboard_setup():
    """
    Complete dashboard setup and data preparation
    """
    print("🚀 LA Collections Dashboard Setup")
    print("=" * 50)
    
    try:
        # Import dashboard modules
        from Data_Loading.dashboard_data_loader import DashboardDataLoader
        from Scripts.prepare_dashboard_data import prepare_all_dashboard_data, prepare_real_time_summary
        
        print("📦 Dashboard modules imported successfully")
        
        # Test data loader
        print("\n🧪 Testing data loader...")
        loader = DashboardDataLoader()
        data = loader.load_dimensional_data()
        
        if not data or all(df.empty for df in data.values()):
            print("❌ No data available - please run harmonization first")
            print("\n💡 To prepare data, run these commands:")
            print("   1. python app/Transformation/harmonize_dim_time.py")
            print("   2. python app/Transformation/harmonize_dim_product.py") 
            print("   3. python app/Transformation/harmonize_dim_order.py")
            print("   4. python app/Transformation/harmonize_dim_customer.py")
            print("   5. python app/Transformation/harmonize_fact_orders.py")
            print("   6. python app/Transformation/harmonize_fact_sales_aggregate.py")
            return False
        
        print("✅ Data loader working correctly")
        
        # Show data summary
        print("\n📊 Data Summary:")
        for table_name, df in data.items():
            if not df.empty:
                print(f"   ✅ {table_name}: {len(df):,} records")
            else:
                print(f"   ⚠️ {table_name}: No data")
        
        # Test sales summary
        print("\n💰 Testing sales summary...")
        summary = loader.get_sales_summary()
        if summary:
            print(f"   ✅ Total Revenue: ₱{summary['total_revenue']:,.2f}")
            print(f"   ✅ Total Orders: {summary['total_orders']:,}")
            print(f"   ✅ Average Order Value: ₱{summary['average_order_value']:.2f}")
        
        # Prepare all dashboard data
        print("\n🔄 Preparing all dashboard data...")
        success = prepare_all_dashboard_data()
        
        if success:
            print("✅ Dashboard data prepared successfully")
            
            # Prepare real-time summary
            print("\n⚡ Preparing real-time summary...")
            prepare_real_time_summary()
            
            # Show prepared files
            data_prepared_dir = os.path.join(dashboard_dir, 'Data_Prepared')
            if os.path.exists(data_prepared_dir):
                files = [f for f in os.listdir(data_prepared_dir) if f.endswith(('.csv', '.json'))]
                print(f"\n📁 Dashboard data files prepared ({len(files)} files):")
                for file in sorted(files):
                    file_path = os.path.join(data_prepared_dir, file)
                    size = os.path.getsize(file_path) / 1024  # KB
                    print(f"   📄 {file} ({size:.1f} KB)")
            
            print(f"\n🎉 Dashboard setup completed successfully!")
            print(f"📁 Data location: {data_prepared_dir}")
            print(f"💡 You can now use these files for your dashboard!")
            
            return True
            
        else:
            print("❌ Dashboard data preparation failed")
            return False
            
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("💡 Make sure you're running this from the correct directory")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_dashboard_loader():
    """
    Quick test of dashboard data loader functionality
    """
    print("\n🧪 Quick Dashboard Loader Test")
    print("-" * 30)
    
    try:
        from Data_Loading.dashboard_data_loader import DashboardDataLoader
        
        loader = DashboardDataLoader()
        
        # Test various functions
        print("📊 Testing sales summary...")
        summary = loader.get_sales_summary()
        if summary:
            print(f"   Revenue: ₱{summary.get('total_revenue', 0):,.2f}")
        
        print("📈 Testing revenue trends...")
        revenue_monthly = loader.get_revenue_by_time('monthly')
        print(f"   Monthly data points: {len(revenue_monthly)}")
        
        print("🛍️ Testing product performance...")
        top_products = loader.get_product_performance(10)
        print(f"   Top products retrieved: {len(top_products)}")
        
        print("⚖️ Testing platform comparison...")
        platform_comp = loader.get_platform_comparison()
        print(f"   Platform comparison data: {len(platform_comp)} platforms")
        
        print("✅ All dashboard loader tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Dashboard loader test failed: {e}")
        return False

if __name__ == "__main__":
    print("🔧 Running Quick Dashboard Setup...")
    
    # Run full setup
    setup_success = run_dashboard_setup()
    
    if setup_success:
        # Run quick test
        test_dashboard_loader()
    
    print(f"\n📁 Dashboard folder structure:")
    print(f"   📂 Dashboard/")
    print(f"   ├── 📂 Data_Loading/")
    print(f"   │   └── 📄 dashboard_data_loader.py")
    print(f"   ├── 📂 Data_Prepared/")
    print(f"   │   └── 📄 (generated dashboard files)")
    print(f"   └── 📂 Scripts/")
    print(f"       ├── 📄 prepare_dashboard_data.py")
    print(f"       └── 📄 quick_setup.py")
    
    print(f"\n💡 To use in your dashboard:")
    print(f"   from app.Dashboard.Data_Loading.dashboard_data_loader import DashboardDataLoader")
    print(f"   loader = DashboardDataLoader()")
    print(f"   data = loader.load_dimensional_data()")
# 📊 LA Collections Dashboard Module

This folder contains all the data loading and preparation scripts specifically designed for dashboard development and visualization.

## 📁 Folder Structure

```
Dashboard/
├── 📂 Data_Loading/           # Core data loading modules
│   └── dashboard_data_loader.py   # Main data loader class
├── 📂 Data_Prepared/         # Generated dashboard data files
│   ├── sales_summary.json        # Overall sales metrics
│   ├── revenue_by_month.csv       # Monthly revenue trends
│   ├── top_products.csv           # Product performance data
│   ├── platform_comparison.csv    # Lazada vs Shopee metrics
│   ├── customer_segments.csv      # Customer analytics
│   └── realtime_summary.json     # Real-time dashboard summary
├── 📂 Scripts/               # Data preparation scripts
│   ├── prepare_dashboard_data.py  # Full data preparation
│   └── quick_setup.py             # Quick setup and testing
└── README.md                # This file
```

## 🚀 Quick Start

### 1. Setup Dashboard Data

Run the quick setup script to prepare all dashboard data:

```bash
cd app/Dashboard/Scripts
python quick_setup.py
```

This will:

- ✅ Test the data loader
- ✅ Prepare all dashboard data files
- ✅ Generate real-time summaries
- ✅ Create data manifest

### 2. Use in Your Dashboard

```python
# Import the dashboard data loader
from app.Dashboard.Data_Loading.dashboard_data_loader import DashboardDataLoader

# Initialize the loader
loader = DashboardDataLoader()

# Load all dimensional data
data = loader.load_dimensional_data()

# Get quick sales summary
summary = loader.get_sales_summary()
print(f"Total Revenue: ₱{summary['total_revenue']:,.2f}")

# Get revenue trends
revenue_monthly = loader.get_revenue_by_time('monthly')

# Get top products
top_products = loader.get_product_performance(20, 'revenue')

# Platform comparison
platform_comparison = loader.get_platform_comparison()
```

## 📊 Available Data Functions

### Sales Analytics

- `get_sales_summary(start_date, end_date, platform)` - Overall sales metrics
- `get_revenue_by_time(period, platform)` - Revenue trends by time period
- `get_platform_comparison()` - Lazada vs Shopee comparison

### Product Analytics

- `get_product_performance(top_n, metric)` - Top products by revenue/quantity
- Category analysis in prepared files

### Customer Analytics

- `get_customer_segments(top_n)` - Customer segmentation analysis
- Customer lifetime value metrics

### Time-based Analytics

- Daily, weekly, monthly, yearly revenue trends
- Year-over-year comparisons
- Seasonal analysis

## 📄 Prepared Data Files

After running the preparation scripts, you'll have these files ready for dashboard consumption:

### JSON Files (for APIs/summary data)

- `sales_summary.json` - Overall sales metrics
- `sales_summary_lazada.json` - Lazada-specific metrics
- `sales_summary_shopee.json` - Shopee-specific metrics
- `realtime_summary.json` - Real-time dashboard overview
- `data_manifest.json` - Data preparation metadata

### CSV Files (for charts/visualizations)

- `revenue_by_daily.csv` - Daily revenue trends
- `revenue_by_weekly.csv` - Weekly revenue trends
- `revenue_by_monthly.csv` - Monthly revenue trends
- `revenue_by_yearly.csv` - Yearly revenue trends
- `top_products_by_revenue.csv` - Top 50 products by revenue
- `top_products_by_quantity.csv` - Top 50 products by quantity
- `platform_comparison.csv` - Platform metrics comparison
- `customer_segments.csv` - Top 200 customer segments
- `category_analysis.csv` - Product category performance
- `monthly_revenue_breakdown.csv` - Detailed monthly breakdown

## 🔄 Data Refresh

To refresh dashboard data with latest information:

```python
# Refresh all data
from app.Dashboard.Scripts.prepare_dashboard_data import prepare_all_dashboard_data
prepare_all_dashboard_data()

# Or just refresh real-time summary
from app.Dashboard.Scripts.prepare_dashboard_data import prepare_real_time_summary
prepare_real_time_summary()
```

## 💡 Tips for Dashboard Development

1. **Use prepared files** - The CSV files are optimized for dashboard consumption
2. **Cache data** - The data loader includes caching to improve performance
3. **Real-time updates** - Use `realtime_summary.json` for live dashboard metrics
4. **Platform filtering** - All functions support platform-specific filtering
5. **Date filtering** - Most functions support date range filtering

## 🛠️ Troubleshooting

### No data available

Make sure you've run all harmonization scripts first:

```bash
# Run these in order:
python app/Transformation/harmonize_dim_time.py
python app/Transformation/harmonize_dim_product.py
python app/Transformation/harmonize_dim_order.py
python app/Transformation/harmonize_dim_customer.py
python app/Transformation/harmonize_fact_orders.py
python app/Transformation/harmonize_fact_sales_aggregate.py
```

### Import errors

Make sure you're running from the correct directory and paths are set up properly.

### Empty data files

Check that the source dimensional data files exist in `app/Transformed/`.

## 📱 Dashboard Integration Examples

### Web Dashboard (Flask/FastAPI)

```python
from flask import Flask, jsonify
from app.Dashboard.Data_Loading.dashboard_data_loader import DashboardDataLoader

app = Flask(__name__)
loader = DashboardDataLoader()

@app.route('/api/sales-summary')
def sales_summary():
    return jsonify(loader.get_sales_summary())

@app.route('/api/revenue-trends')
def revenue_trends():
    trends = loader.get_revenue_by_time('monthly')
    return trends.to_json(orient='records')
```

### Streamlit Dashboard

```python
import streamlit as st
from app.Dashboard.Data_Loading.dashboard_data_loader import DashboardDataLoader

loader = DashboardDataLoader()
summary = loader.get_sales_summary()

st.metric("Total Revenue", f"₱{summary['total_revenue']:,.2f}")
st.metric("Total Orders", f"{summary['total_orders']:,}")

# Load and display charts
revenue_data = loader.get_revenue_by_time('monthly')
st.line_chart(revenue_data.set_index('month')['revenue'])
```

---

## 📞 Need Help?

This dashboard module is designed to make it easy to load and prepare data for any type of dashboard or visualization tool. If you need additional data transformations or have specific requirements, you can extend the `DashboardDataLoader` class or modify the preparation scripts.

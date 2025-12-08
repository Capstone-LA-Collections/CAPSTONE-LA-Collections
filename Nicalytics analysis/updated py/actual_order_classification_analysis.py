import pandas as pd

# Load the actual data
df = pd.read_csv('order_classification.csv')

print("="*80)
print("📊 ACTUAL ORDER CLASSIFICATION ANALYSIS")
print("="*80)

# Basic info
print(f"\nTotal Orders Analyzed: {len(df):,}")
print(f"Date Range: {df['order_date'].min()} to {df['order_date'].max()}")

# Segment breakdown
segment_summary = df.groupby('customer_segment').agg({
    'orders_key': 'count',
    'order_value': ['sum', 'mean'],
    'items_in_order': ['sum', 'mean'],
    'order_discount_pct': 'mean',
    'used_voucher': 'sum',
    'is_cod': 'sum',
    'is_sale_day': 'sum'
}).round(2)

segment_summary.columns = ['Order_Count', 'Total_Revenue', 'Avg_AOV', 
                            'Total_Units', 'Avg_Units_per_Order', 
                            'Avg_Discount_Pct', 'Voucher_Orders', 
                            'COD_Orders', 'Sale_Day_Orders']

print("\n" + "="*80)
print("SEGMENT BREAKDOWN")
print("="*80)
print(segment_summary)

# Calculate percentages
total_orders = len(df)
total_revenue = df['order_value'].sum()
total_units = df['items_in_order'].sum()

print("\n" + "="*80)
print("PERCENTAGE DISTRIBUTION")
print("="*80)

for segment in df['customer_segment'].unique():
    seg_data = df[df['customer_segment'] == segment]
    
    print(f"\n{segment.upper()} SEGMENT:")
    print(f"  Orders:   {len(seg_data):>6,} ({len(seg_data)/total_orders*100:>5.1f}%)")
    print(f"  Revenue:  ₱{seg_data['order_value'].sum():>10,.0f} ({seg_data['order_value'].sum()/total_revenue*100:>5.1f}%)")
    print(f"  Units:    {seg_data['items_in_order'].sum():>6,} ({seg_data['items_in_order'].sum()/total_units*100:>5.1f}%)")
    print(f"  Avg AOV:  ₱{seg_data['order_value'].mean():>6.2f}")
    print(f"  Avg Units: {seg_data['items_in_order'].mean():>5.2f} units/order")
    print(f"  Avg Discount: {seg_data['order_discount_pct'].mean():>5.2f}%")
    print(f"  COD Usage:    {seg_data['is_cod'].sum()/len(seg_data)*100:>5.1f}%")

# Revenue efficiency
print("\n" + "="*80)
print("PROFITABILITY ANALYSIS")
print("="*80)

for segment in df['customer_segment'].unique():
    seg_data = df[df['customer_segment'] == segment]
    
    gross_revenue = seg_data['order_value'].sum()
    avg_discount = seg_data['order_discount_pct'].mean() / 100
    net_revenue = gross_revenue * (1 - avg_discount)
    
    # Operational costs (estimated)
    cod_rate = seg_data['is_cod'].sum() / len(seg_data)
    avg_cost_per_order = 35 + (cod_rate * 20)  # Base + COD premium
    total_costs = len(seg_data) * avg_cost_per_order
    
    net_profit = net_revenue - total_costs
    
    print(f"\n{segment.upper()}:")
    print(f"  Gross Revenue:    ₱{gross_revenue:>10,.0f}")
    print(f"  Net Revenue:      ₱{net_revenue:>10,.0f} ({net_revenue/gross_revenue*100:.1f}% retention)")
    print(f"  Operating Costs:  ₱{total_costs:>10,.0f}")
    print(f"  Net Profit:       ₱{net_profit:>10,.0f}")
    print(f"  Profit Margin:    {net_profit/gross_revenue*100:>5.1f}%")
    print(f"  Profit per Order: ₱{net_profit/len(seg_data):>6.2f}")

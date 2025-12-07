import numpy as np
from scipy.optimize import minimize

# Historical segment response to discount rates
def calculate_segment_orders(discount_rate):
    """
    Based on your regression results:
    - Regular buyers: Inversely related to discounts
    - Promotional buyers: Directly related to discounts
    """
    # Term inside exponent (ensures it's not negative before raising to 0.8 power)
    exponent_base = 1 - (discount_rate - 0.04) / 0.10
    
    # Regular segment (decreases with discounts)
    # FIX: Use np.maximum to ensure the base is not negative, avoiding complex numbers
    regular_orders = 180 * np.maximum(0, exponent_base) ** 0.8
    regular_orders = max(regular_orders, 50)  # Floor for orders
    
    # Promotional segment (increases with discounts)
    promo_orders = 80 + (discount_rate - 0.04) * 3000
    promo_orders = min(promo_orders, 250)  # Ceiling
    
    return regular_orders, promo_orders

def calculate_monthly_performance(discount_rate):
    """
    Calculate total monthly performance at given discount rate
    """
    # Ensure discount_rate is a float/scalar when passed from minimization
    if isinstance(discount_rate, np.ndarray):
        discount_rate = discount_rate.item()
        
    regular_orders, promo_orders = calculate_segment_orders(discount_rate)
    
    # Regular segment metrics
    regular_aov = 543
    regular_units_per_order = 2.2
    regular_margin = 0.968
    regular_cost_per_order = 35  # Lower fulfillment cost (digital payment)
    
    # Promotional segment metrics
    promo_aov = 363
    promo_units_per_order = 4.1
    promo_margin = 1 - discount_rate  # Margin = what's left after discounts
    promo_cost_per_order = 55  # Higher fulfillment cost (COD)
    
    # Calculate performance
    regular_revenue = regular_orders * regular_aov
    promo_revenue = promo_orders * promo_aov
    total_revenue = regular_revenue + promo_revenue
    
    regular_profit = (regular_revenue * regular_margin) - (regular_orders * regular_cost_per_order)
    promo_profit = (promo_revenue * promo_margin) - (promo_orders * promo_cost_per_order)
    total_profit = regular_profit + promo_profit
    
    total_orders = regular_orders + promo_orders
    total_units = (regular_orders * regular_units_per_order) + (promo_orders * promo_units_per_order)
    
    return {
        'discount_rate': discount_rate,
        'total_revenue': total_revenue,
        'total_profit': total_profit,
        'total_orders': total_orders,
        'total_units': total_units,
        'regular_orders': regular_orders,
        'promo_orders': promo_orders,
        'profit_margin': total_profit / total_revenue if total_revenue > 0 else 0
    }

# The rest of your script (Testing and Optimization) remains the same.
# Use the corrected calculate_segment_orders function and rerun your full script.
# Test different discount rates
print("\n" + "="*80)
print("📊 DISCOUNT RATE SENSITIVITY ANALYSIS")
print("="*80)

discount_rates = [0.03, 0.05, 0.07, 0.09, 0.11, 0.13, 0.15]

results = []
for rate in discount_rates:
    result = calculate_monthly_performance(rate)
    results.append(result)
    
    print(f"\nDiscount Rate: {rate*100:.0f}%")
    print(f"  Revenue:  ₱{result['total_revenue']:>10,.0f}")
    print(f"  Profit:   ₱{result['total_profit']:>10,.0f}  ({result['profit_margin']*100:.1f}% margin)")
    print(f"  Orders:   {result['total_orders']:>10,.0f}  (Regular: {result['regular_orders']:.0f}, Promo: {result['promo_orders']:.0f})")
    print(f"  Units:    {result['total_units']:>10,.0f}")

# Find optimal discount rate
print("\n" + "="*80)
print("🎯 OPTIMIZATION: MAXIMIZE PROFIT SUBJECT TO REVENUE TARGET")
print("="*80)

def objective(discount_rate):
    """Minimize negative profit (= maximize profit)"""
    result = calculate_monthly_performance(discount_rate[0])
    return -result['total_profit']

def revenue_constraint(discount_rate):
    """Revenue must be >= ₱2.33M/month (₱28M annual)"""
    result = calculate_monthly_performance(discount_rate[0])
    return result['total_revenue'] - 2_333_333

# Optimize
bounds = [(0.03, 0.20)]  # Discount rate between 3% and 20%
constraints = [{'type': 'ineq', 'fun': revenue_constraint}]

result = minimize(
    objective,
    x0=[0.07],  # Initial guess: 7%
    bounds=bounds,
    constraints=constraints,
    method='SLSQP'
)

optimal_rate = result.x[0]
optimal_performance = calculate_monthly_performance(optimal_rate)

print(f"\n✅ OPTIMAL DISCOUNT RATE: {optimal_rate*100:.1f}%")
print(f"\nExpected Monthly Performance:")
print(f"  Revenue:        ₱{optimal_performance['total_revenue']:,.0f}")
print(f"  Profit:         ₱{optimal_performance['total_profit']:,.0f}")
print(f"  Profit Margin:  {optimal_performance['profit_margin']*100:.1f}%")
print(f"  Total Orders:   {optimal_performance['total_orders']:.0f}")
print(f"    • Regular:    {optimal_performance['regular_orders']:.0f} ({optimal_performance['regular_orders']/optimal_performance['total_orders']*100:.0f}%)")
print(f"    • Promo:      {optimal_performance['promo_orders']:.0f} ({optimal_performance['promo_orders']/optimal_performance['total_orders']*100:.0f}%)")
print(f"  Total Units:    {optimal_performance['total_units']:.0f}")

print(f"\nAnnual Projections:")
print(f"  Revenue:  ₱{optimal_performance['total_revenue'] * 12:,.0f}")
print(f"  Profit:   ₱{optimal_performance['total_profit'] * 12:,.0f}")
import pandas as pd
import numpy as np
from scipy.stats import pearsonr, spearmanr
import matplotlib.pyplot as plt
import seaborn as sns
from pingouin import partial_corr
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

# Load data from SQL query
df = pd.read_csv('lazada_monthly_factors.csv')

# Clean data - remove any rows with nulls
df = df.dropna()

print(f"Dataset size: {len(df)} months of Lazada data")
print(f"Date range: {df['month_date'].min()} to {df['month_date'].max()}")
print("\n" + "="*80)

# ============================================
# CORRELATION ANALYSIS: Each Factor vs Performance
# ============================================

# Define factors and outcomes
factors = {
    'Factor 1: Promotional Intensity': 'factor1_promotional_intensity',
    'Factor 2: Operational Efficiency': 'factor2_operational_efficiency',
    'Factor 3: Payment Mix': 'factor3_payment_mix',
    'Factor 4: Customer Value': 'factor4_customer_value',
    'Factor 5: Concentration': 'factor5_concentration'
}

outcomes = {
    'Net Sales (₱)': 'net_sales',
    'Total Orders': 'total_orders',
    'Net Margin (%)': 'net_margin'
}

# Create correlation matrix
print("\n📊 PEARSON CORRELATION ANALYSIS")
print("="*80)

correlation_results = []

for factor_name, factor_col in factors.items():
    print(f"\n{factor_name}")
    print("-" * 60)
    
    for outcome_name, outcome_col in outcomes.items():
        # Calculate Pearson correlation
        r, p_value = pearsonr(df[factor_col], df[outcome_col])
        
        # Interpret correlation strength
        if abs(r) >= 0.9:
            strength = "Very Strong"
        elif abs(r) >= 0.7:
            strength = "Strong"
        elif abs(r) >= 0.5:
            strength = "Moderate"
        elif abs(r) >= 0.3:
            strength = "Weak"
        else:
            strength = "Very Weak"
        
        # Interpret direction
        direction = "Positive" if r > 0 else "Negative"
        
        # Statistical significance
        significant = "✓ Significant" if p_value < 0.05 else "✗ Not Significant"
        
        print(f"  vs {outcome_name}:")
        print(f"    r = {r:+.3f} ({direction}, {strength})")
        print(f"    p-value = {p_value:.4f} {significant}")
        
        # Store results
        correlation_results.append({
            'Factor': factor_name,
            'Outcome': outcome_name,
            'Correlation (r)': r,
            'P-value': p_value,
            'Significant': p_value < 0.05,
            'Strength': strength,
            'Direction': direction
        })

# Convert to DataFrame for easy viewing
results_df = pd.DataFrame(correlation_results)

# ============================================
# HYPOTHESIS TESTING: Validate Expected Correlations
# ============================================
print("\n\n🎯 HYPOTHESIS VALIDATION")
print("="*80)

expected_correlations = {
    'Factor 1: Promotional Intensity': {
        'outcome': 'Net Margin (%)',
        'expected_r': -0.98,
        'expected_direction': 'Negative',
        'hypothesis': 'Higher discounts → Lower margins (1:1 inverse)'
    },
    'Factor 2: Operational Efficiency': {
        'outcome': 'Total Orders',
        'expected_r': +0.89,
        'expected_direction': 'Positive',
        'hypothesis': 'Higher fulfillment → More completed orders'
    },
    'Factor 3: Payment Mix': {
        'outcome': 'Net Margin (%)',
        'expected_r': +0.72,
        'expected_direction': 'Positive',
        'hypothesis': 'More digital payments → Lower costs → Higher margins'
    },
    'Factor 4: Customer Value': {
        'outcome': 'Net Sales (₱)',
        'expected_r': +0.84,
        'expected_direction': 'Positive',
        'hypothesis': 'Higher AOV → Higher revenue'
    },
    'Factor 5: Concentration': {
        'outcome': 'Net Sales (₱)',
        'expected_r': -0.76,
        'expected_direction': 'Negative',
        'hypothesis': 'Higher concentration → Constrained growth'
    }
}

hypothesis_results = []

for factor_name, expectations in expected_correlations.items():
    # Find actual correlation
    actual = results_df[
        (results_df['Factor'] == factor_name) & 
        (results_df['Outcome'] == expectations['outcome'])
    ].iloc[0]
    
    actual_r = actual['Correlation (r)']
    actual_sig = actual['Significant']
    expected_r = expectations['expected_r']
    
    # Check if matches expectation
    direction_match = (
        (expected_r > 0 and actual_r > 0) or 
        (expected_r < 0 and actual_r < 0)
    )
    
    strength_match = abs(actual_r) >= (abs(expected_r) - 0.15)  # Allow 0.15 tolerance
    
    validated = direction_match and strength_match and actual_sig
    
    print(f"\n{factor_name}")
    print(f"  Hypothesis: {expectations['hypothesis']}")
    print(f"  Expected: r ≈ {expected_r:+.2f}")
    print(f"  Actual: r = {actual_r:+.3f}")
    print(f"  Result: {'✓ VALIDATED' if validated else '✗ NOT VALIDATED'}")
    
    hypothesis_results.append({
        'Factor': factor_name,
        'Expected_r': expected_r,
        'Actual_r': actual_r,
        'Direction_Match': direction_match,
        'Strength_Match': strength_match,
        'Significant': actual_sig,
        'Validated': validated
    })

hypothesis_df = pd.DataFrame(hypothesis_results)

# ============================================
# VISUALIZATION: Correlation Heatmap
# ============================================
print("\n\n📈 Generating Correlation Heatmap...")

# Create correlation matrix
factor_cols = list(factors.values())
outcome_cols = list(outcomes.values())

# Calculate full correlation matrix
corr_matrix = df[factor_cols + outcome_cols].corr()

# Extract just factor-outcome correlations
factor_outcome_corr = corr_matrix.loc[factor_cols, outcome_cols]

# Rename for readability
factor_outcome_corr.index = [f"F{i+1}: {name.split(':')[1].strip()}" 
                               for i, name in enumerate(factors.keys())]
factor_outcome_corr.columns = list(outcomes.keys())

# Create heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(
    factor_outcome_corr, 
    annot=True, 
    fmt='.3f',
    cmap='RdYlGn',
    center=0,
    vmin=-1,
    vmax=1,
    cbar_kws={'label': 'Pearson Correlation (r)'}
)
plt.title('Correlation: 5 Factors vs Performance Outcomes (Lazada)', 
          fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('lazada_factor_correlation_heatmap.png', dpi=300)
print("  ✓ Saved: lazada_factor_correlation_heatmap.png")

# ============================================
# SCATTER PLOTS: Visual Proof
# ============================================
print("\n📊 Generating Scatter Plots...")

fig, axes = plt.subplots(3, 2, figsize=(15, 18))
axes = axes.flatten()

plot_configs = [
    ('factor1_promotional_intensity', 'net_margin', 'Factor 1: Discount Rate vs Net Margin'),
    ('factor2_operational_efficiency', 'total_orders', 'Factor 2: Fulfillment Rate vs Orders'),
    ('factor3_payment_mix', 'net_margin', 'Factor 3: Digital Adoption vs Net Margin'),
    ('factor4_customer_value', 'net_sales', 'Factor 4: AOV vs Net Sales'),
    ('factor5_concentration', 'net_sales', 'Factor 5: Concentration vs Net Sales'),
    ('factor1_promotional_intensity', 'total_orders', 'Discount Rate vs Orders (Growth Lever)')]

for idx, (x_col, y_col, title) in enumerate(plot_configs):
    if idx >= len(axes):
        break
    
    ax = axes[idx]
    
    # Scatter plot
    ax.scatter(df[x_col], df[y_col], alpha=0.6, s=50)
    
    # Fit line
    z = np.polyfit(df[x_col], df[y_col], 1)
    p = np.poly1d(z)
    ax.plot(df[x_col], p(df[x_col]), "r--", linewidth=2)
    
    # Correlation
    r, p_val = pearsonr(df[x_col], df[y_col])
    
    # Labels
    ax.set_xlabel(x_col.replace('_', ' ').title())
    ax.set_ylabel(y_col.replace('_', ' ').title())
    ax.set_title(f"{title}\nr = {r:+.3f}, p = {p_val:.4f}")
    ax.grid(True, alpha=0.3)

# Hide extra subplot
if len(plot_configs) < len(axes):
    axes[-1].set_visible(False)

plt.tight_layout()
plt.savefig('lazada_factor_scatter_plots.png', dpi=300)
print("  ✓ Saved: lazada_factor_scatter_plots.png")

# ============================================
# SUMMARY REPORT
# ============================================
print("\n\n" + "="*80)
print("📋 SUMMARY REPORT: CORRELATION ANALYSIS")
print("="*80)

# Count validated factors
validated_count = hypothesis_df['Validated'].sum()
print(f"\n✓ Validated Factors: {validated_count}/5 ({validated_count/5*100:.0f}%)")

# Show validation status
print("\nValidation Status:")
for _, row in hypothesis_df.iterrows():
    status = "✓" if row['Validated'] else "✗"
    print(f"  {status} {row['Factor']}: r = {row['Actual_r']:+.3f} (expected {row['Expected_r']:+.2f})")

# Strongest correlations
print("\n🏆 Strongest Correlations:")
top_3 = results_df[results_df['Significant']].nlargest(3, 'Correlation (r)', keep='all')
for idx, row in top_3.iterrows():
    print(f"  {row['Factor']} → {row['Outcome']}: r = {row['Correlation (r)']:+.3f}")

# Weakest correlations (but still significant)
print("\n⚠️  Weakest Correlations (but significant):")
bottom_3 = results_df[results_df['Significant']].nsmallest(3, 'Correlation (r)', keep='all')
for idx, row in bottom_3.iterrows():
    print(f"  {row['Factor']} → {row['Outcome']}: r = {row['Correlation (r)']:+.3f}")

# Non-significant correlations (if any)
non_sig = results_df[~results_df['Significant']]
if len(non_sig) > 0:
    print("\n❌ Non-Significant Correlations:")
    for idx, row in non_sig.iterrows():
        print(f"  {row['Factor']} → {row['Outcome']}: r = {row['Correlation (r)']:+.3f}, p = {row['P-value']:.4f}")

# ============================================
# EXPORT RESULTS
# ============================================
print("\n\n💾 Exporting Results...")

# Save correlation results
results_df.to_csv('lazada_correlation_results.csv', index=False)
print("  ✓ Saved: lazada_correlation_results.csv")

# Save hypothesis validation
hypothesis_df.to_csv('lazada_hypothesis_validation.csv', index=False)
print("  ✓ Saved: lazada_hypothesis_validation.csv")

# Save correlation matrix
factor_outcome_corr.to_csv('lazada_correlation_matrix.csv')
print("  ✓ Saved: lazada_correlation_matrix.csv")

print("\n" + "="*80)
print("✅ CORRELATION ANALYSIS COMPLETE")
print("="*80)

# ============================================
# SPEARMAN CORRELATION (for non-linear relationships)
# ============================================
print("\n\n📊 SPEARMAN CORRELATION (Non-Linear)")
print("="*80)

spearman_results = []

for factor_name, factor_col in factors.items():
    for outcome_name, outcome_col in outcomes.items():
        # Spearman correlation (rank-based, captures non-linear)
        rho, p_value = spearmanr(df[factor_col], df[outcome_col])
        
        print(f"{factor_name} vs {outcome_name}:")
        print(f"  ρ (rho) = {rho:+.3f}, p = {p_value:.4f}")
        
        spearman_results.append({
            'Factor': factor_name,
            'Outcome': outcome_name,
            'Spearman_rho': rho,
            'P-value': p_value
        })

spearman_df = pd.DataFrame(spearman_results)
spearman_df.to_csv('lazada_spearman_correlation.csv', index=False)



# ============================================
# PARTIAL CORRELATION
# ============================================
print("\n\n🔬 PARTIAL CORRELATION ANALYSIS")
print("="*80)
print("(Controlling for other factors)")

# For each factor, calculate partial correlation with orders
# controlling for all other factors
partial_results = []

for factor_name, factor_col in factors.items():
    # List of other factors to control for
    control_factors = [col for col in factor_cols if col != factor_col]
    
    # Calculate partial correlation
    partial_r = partial_corr(
        data=df,
        x=factor_col,
        y='total_orders',
        covar=control_factors
    )
    
    r_value = partial_r['r'].values[0]
    p_value = partial_r['p-val'].values[0]
    
    print(f"\n{factor_name}")
    print(f"  Partial r = {r_value:+.3f}")
    print(f"  p-value = {p_value:.4f}")
    print(f"  {'✓ Significant' if p_value < 0.05 else '✗ Not Significant'}")
    
    partial_results.append({
        'Factor': factor_name,
        'Partial_r': r_value,
        'P-value': p_value,
        'Significant': p_value < 0.05
    })

partial_df = pd.DataFrame(partial_results)
partial_df.to_csv('lazada_partial_correlation.csv', index=False)

print("\n📊 Partial Correlation Summary:")
print(partial_df.to_string(index=False))


#=====================================
# Combined effect of all 5 factors and their relative importance
#=====================================
X = df[['factor1_promotional_intensity', 'factor2_operational_efficiency', 
        'factor3_payment_mix', 'factor4_customer_value', 'factor5_concentration']]
y = df['total_orders']

# Standardize to compare beta coefficients
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Fit model
model = LinearRegression()
model.fit(X_scaled, y)

# Results
print("\n Combined effect of all 5 factors and their relative importance")
print(f"R² = {model.score(X_scaled, y):.3f}")  # How much variance explained
print("\nStandardized Beta Coefficients:")
for name, coef in zip(X.columns, model.coef_):
    print(f"  {name}: {coef:.3f}")
    
print("\n\n" + "="*80)
print("📦 REGRESSION ANALYSIS: TOTAL QUANTITY SOLD")
print("="*80)

# Check if we have quantity data
if 'total_units' in df.columns or 'total_items_sold' in df.columns:
    # Use whichever column name exists
    quantity_col = 'total_units' if 'total_units' in df.columns else 'total_items_sold'
    
    # Prepare features (same 5 factors)
    X_quantity = df[['factor1_promotional_intensity', 'factor2_operational_efficiency', 
                     'factor3_payment_mix', 'factor4_customer_value', 'factor5_concentration']]
    y_quantity = df[quantity_col]
    
    # Standardize features
    scaler_qty = StandardScaler()
    X_quantity_scaled = scaler_qty.fit_transform(X_quantity)
    
    # Fit model
    model_quantity = LinearRegression()
    model_quantity.fit(X_quantity_scaled, y_quantity)
    
    # Calculate metrics
    r2_qty = model_quantity.score(X_quantity_scaled, y_quantity)
    
    print(f"\n📊 MODEL PERFORMANCE:")
    print(f"  R² = {r2_qty:.3f} ({r2_qty*100:.1f}% of variance explained)")
    
    print("\n📈 STANDARDIZED BETA COEFFICIENTS:")
    print("-" * 70)
    
    # Create results dataframe
    qty_coefficients = []
    for name, coef in zip(X_quantity.columns, model_quantity.coef_):
        factor_name = name.replace('factor1_', 'F1: ').replace('factor2_', 'F2: ') \
                          .replace('factor3_', 'F3: ').replace('factor4_', 'F4: ') \
                          .replace('factor5_', 'F5: ').replace('_', ' ').title()
        
        print(f"  {factor_name:45s}: {coef:+10.3f}")
        
        qty_coefficients.append({
            'Factor': factor_name,
            'Beta': coef,
            'Abs_Beta': abs(coef)
        })
    
    qty_coef_df = pd.DataFrame(qty_coefficients).sort_values('Abs_Beta', ascending=False)
    
    # Ranking
    print("\n🏆 FACTOR IMPORTANCE RANKING (by absolute beta):")
    print("-" * 70)
    for idx, row in qty_coef_df.iterrows():
        rank = qty_coef_df.index.get_loc(idx) + 1
        direction = "↑ Positive" if row['Beta'] > 0 else "↓ Negative"
        print(f"  #{rank}. {row['Factor']:40s}: {row['Beta']:+10.3f} ({direction})")
    
    # Compare with Orders regression
    print("\n\n" + "="*80)
    print("📊 COMPARISON: ORDERS vs QUANTITY DRIVERS")
    print("="*80)
    
    comparison_data = []
    for i, (name, order_coef) in enumerate(zip(X.columns, model.coef_)):
        qty_coef = model_quantity.coef_[i]
        factor_short = name.replace('factor1_', 'F1').replace('factor2_', 'F2') \
                          .replace('factor3_', 'F3').replace('factor4_', 'F4') \
                          .replace('factor5_', 'F5').replace('_', ' ')
        
        comparison_data.append({
            'Factor': factor_short,
            'Orders_Beta': order_coef,
            'Quantity_Beta': qty_coef,
            'Ratio_Qty_to_Orders': qty_coef / order_coef if order_coef != 0 else np.inf
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    
    print("\n" + comparison_df.to_string(index=False))
    
    # Interpretation
    print("\n\n💡 KEY INSIGHTS:")
    print("-" * 70)
    
    # Find dominant factor for quantity
    max_qty_idx = qty_coef_df.iloc[0].name
    dominant_qty_factor = qty_coef_df.iloc[0]['Factor']
    dominant_qty_beta = qty_coef_df.iloc[0]['Beta']
    
    # Find dominant factor for orders
    order_coefficients = [{'Factor': name, 'Beta': coef, 'Abs_Beta': abs(coef)} 
                          for name, coef in zip(X.columns, model.coef_)]
    order_coef_df = pd.DataFrame(order_coefficients).sort_values('Abs_Beta', ascending=False)
    dominant_order_factor = order_coef_df.iloc[0]['Factor'].replace('factor4_customer_value', 'Customer Value')
    dominant_order_beta = order_coef_df.iloc[0]['Beta']
    
    print(f"\n1. DOMINANT DRIVER FOR ORDERS:")
    print(f"   {dominant_order_factor}: β = {dominant_order_beta:+.3f}")
    
    print(f"\n2. DOMINANT DRIVER FOR QUANTITY:")
    print(f"   {dominant_qty_factor}: β = {dominant_qty_beta:+.3f}")
    
    # Units per order calculation
    if dominant_qty_factor == dominant_order_factor.replace('customer_value', 'Customer Value'):
        print(f"\n3. SAME FACTOR DRIVES BOTH:")
        print(f"   {dominant_qty_factor} is the key driver for both order count AND quantity")
        print(f"   This means higher {dominant_qty_factor.split(':')[1].strip()} leads to:")
        print(f"   - More orders ({dominant_order_beta:+.1f} per SD increase)")
        print(f"   - More units ({dominant_qty_beta:+.1f} per SD increase)")
        print(f"   - Ratio: {dominant_qty_beta/dominant_order_beta:.2f}x more units than orders")
    else:
        print(f"\n3. DIFFERENT DRIVERS:")
        print(f"   Orders driven by: {dominant_order_factor}")
        print(f"   Quantity driven by: {dominant_qty_factor}")
        print(f"   Strategy: Optimize these factors separately for maximum impact")
    
    # Units per order insight
    print(f"\n4. UNITS PER ORDER IMPLICATIONS:")
    units_per_order_current = df[quantity_col].sum() / df['total_orders'].sum()
    print(f"   Current units per order: {units_per_order_current:.2f}")
    
    # Which factor increases basket size most?
    basket_impact = []
    for i, factor in enumerate(X.columns):
        order_beta = model.coef_[i]
        qty_beta = model_quantity.coef_[i]
        
        if order_beta != 0:
            basket_multiplier = qty_beta / order_beta
            basket_impact.append({
                'Factor': factor.replace('factor', 'F').replace('_', ' '),
                'Basket_Multiplier': basket_multiplier,
                'Interpretation': 'Increases basket size' if basket_multiplier > 1 else 'Decreases basket size'
            })
    
    basket_df = pd.DataFrame(basket_impact).sort_values('Basket_Multiplier', ascending=False)
    
    print("\n5. BASKET SIZE IMPACT (Quantity/Order Ratio):")
    print("-" * 70)
    for _, row in basket_df.iterrows():
        symbol = "📈" if row['Basket_Multiplier'] > 1 else "📉"
        print(f"   {symbol} {row['Factor']:40s}: {row['Basket_Multiplier']:+.3f}x ({row['Interpretation']})")
    
    # Export results
    print("\n\n💾 EXPORTING QUANTITY REGRESSION RESULTS...")
    qty_coef_df.to_csv('lazada_quantity_regression_coefficients.csv', index=False)
    comparison_df.to_csv('lazada_orders_vs_quantity_comparison.csv', index=False)
    basket_df.to_csv('lazada_basket_size_impact.csv', index=False)
    print("  ✓ Saved: lazada_quantity_regression_coefficients.csv")
    print("  ✓ Saved: lazada_orders_vs_quantity_comparison.csv")
    print("  ✓ Saved: lazada_basket_size_impact.csv")
    
    # ============================================
    # VISUALIZATION: Beta Coefficient Comparison
    # ============================================
    print("\n\n📊 Generating Beta Coefficient Comparison Chart...")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Chart 1: Orders Beta Coefficients
    order_plot_df = order_coef_df.sort_values('Beta')
    colors_orders = ['red' if x < 0 else 'green' for x in order_plot_df['Beta']]
    
    ax1.barh(range(len(order_plot_df)), order_plot_df['Beta'], color=colors_orders, alpha=0.7)
    ax1.set_yticks(range(len(order_plot_df)))
    ax1.set_yticklabels([f.replace('factor', 'F').replace('_', ' ').title() 
                          for f in order_plot_df['Factor']])
    ax1.set_xlabel('Standardized Beta Coefficient', fontsize=12, fontweight='bold')
    ax1.set_title('Factors Driving TOTAL ORDERS\n(Higher β = Stronger Impact)', 
                  fontsize=13, fontweight='bold')
    ax1.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
    ax1.grid(axis='x', alpha=0.3)
    
    # Add value labels
    for i, v in enumerate(order_plot_df['Beta']):
        ax1.text(v + (100 if v > 0 else -100), i, f'{v:.0f}', 
                va='center', ha='left' if v > 0 else 'right', fontweight='bold')
    
    # Chart 2: Quantity Beta Coefficients
    qty_plot_df = qty_coef_df.sort_values('Beta')
    colors_qty = ['red' if x < 0 else 'green' for x in qty_plot_df['Beta']]
    
    ax2.barh(range(len(qty_plot_df)), qty_plot_df['Beta'], color=colors_qty, alpha=0.7)
    ax2.set_yticks(range(len(qty_plot_df)))
    ax2.set_yticklabels([f.split(':')[1].strip() if ':' in f else f 
                         for f in qty_plot_df['Factor']])
    ax2.set_xlabel('Standardized Beta Coefficient', fontsize=12, fontweight='bold')
    ax2.set_title('Factors Driving TOTAL QUANTITY\n(Higher β = Stronger Impact)', 
                  fontsize=13, fontweight='bold')
    ax2.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
    ax2.grid(axis='x', alpha=0.3)
    
    # Add value labels
    for i, v in enumerate(qty_plot_df['Beta']):
        ax2.text(v + (100 if v > 0 else -100), i, f'{v:.0f}', 
                va='center', ha='left' if v > 0 else 'right', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('lazada_orders_vs_quantity_beta_comparison.png', dpi=300, bbox_inches='tight')
    print("  ✓ Saved: lazada_orders_vs_quantity_beta_comparison.png")
    
    # ============================================
    # STRATEGIC SUMMARY
    # ============================================
    print("\n\n" + "="*80)
    print("🎯 STRATEGIC SUMMARY: ORDERS vs QUANTITY")
    print("="*80)
    
    print(f"\n📊 MODEL PERFORMANCE COMPARISON:")
    print(f"   Orders Model R²:   {model.score(X_scaled, y):.3f} ({model.score(X_scaled, y)*100:.1f}% explained)")
    print(f"   Quantity Model R²: {r2_qty:.3f} ({r2_qty*100:.1f}% explained)")
    
    if r2_qty > 0.95:
        print(f"   ✅ Both models show excellent fit (>95%)")
    elif r2_qty > 0.90:
        print(f"   ✅ Both models show strong fit (>90%)")
    else:
        print(f"   ⚠️  Quantity model has lower explanatory power")
    
    print(f"\n🎯 STRATEGIC IMPLICATIONS:")
    
    # Determine strategy based on coefficients
    if abs(dominant_qty_beta - dominant_order_beta) / dominant_order_beta < 0.1:
        print(f"\n   UNIFIED STRATEGY:")
        print(f"   Both orders AND quantity are driven by the same factor: {dominant_qty_factor}")
        print(f"   → Focus 80% effort on optimizing {dominant_qty_factor.split(':')[1].strip()}")
        print(f"   → This will simultaneously increase order count AND basket size")
    else:
        print(f"\n   DUAL STRATEGY:")
        print(f"   Orders driven by: {dominant_order_factor} (β = {dominant_order_beta:.1f})")
        print(f"   Quantity driven by: {dominant_qty_factor} (β = {dominant_qty_beta:.1f})")
        print(f"   → Optimize different factors for order volume vs basket size")
    
    # Revenue calculation
    print(f"\n💰 REVENUE IMPACT:")
    avg_aov = df['factor4_customer_value'].mean()
    avg_price_per_unit = avg_aov / units_per_order_current
    
    print(f"   Current metrics:")
    print(f"   - Average Order Value: ₱{avg_aov:.2f}")
    print(f"   - Units per Order: {units_per_order_current:.2f}")
    print(f"   - Price per Unit: ₱{avg_price_per_unit:.2f}")
    
    print(f"\n   To maximize revenue:")
    if dominant_qty_beta / dominant_order_beta > 1.5:
        print(f"   → Prioritize QUANTITY growth (basket size)")
        print(f"   → Focus on: {dominant_qty_factor.split(':')[1].strip()}")
        print(f"   → Each unit of {dominant_qty_factor.split(':')[1].strip()} yields")
        print(f"     {dominant_qty_beta:.0f} more units vs {dominant_order_beta:.0f} more orders")
    else:
        print(f"   → Prioritize ORDER growth (transaction count)")
        print(f"   → Focus on: {dominant_order_factor}")
        print(f"   → More efficient to grow orders than basket size")
    
else:
    print("\n⚠️  Warning: Could not find quantity/units column in dataset")
    print("Available columns:", df.columns.tolist())

print("\n" + "="*80)
print("✅ QUANTITY REGRESSION ANALYSIS COMPLETE")
print("="*80)

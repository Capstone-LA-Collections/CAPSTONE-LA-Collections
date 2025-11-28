"""
Verification script to confirm Shopee category matching is working properly
"""

import pandas as pd
import os

def verify_category_fix():
    """Verify that Shopee categories are properly matched"""
    
    print("🔍 Verifying Shopee category matching fix...")
    
    # Load the harmonized product data
    transformed_dir = os.path.join('app', 'Transformed')
    product_file = os.path.join(transformed_dir, 'dim_product.csv')
    
    if not os.path.exists(product_file):
        print("❌ Product file not found. Please run harmonization first.")
        return
    
    df = pd.read_csv(product_file)
    
    # Analyze Shopee categories
    shopee_products = df[df['platform_key'] == 2]
    
    print(f"\n📊 Shopee Product Category Analysis:")
    print(f"   - Total Shopee products: {len(shopee_products)}")
    
    # Check for properly matched categories (should NOT have 'Category_' prefix)
    properly_matched = shopee_products[~shopee_products['product_category'].str.startswith('Category_', na=False)]
    fallback_categories = shopee_products[shopee_products['product_category'].str.startswith('Category_', na=False)]
    none_categories = shopee_products[shopee_products['product_category'].isna()]
    
    print(f"   - Properly matched categories: {len(properly_matched)}")
    print(f"   - Fallback categories (Category_XXX): {len(fallback_categories)}")
    print(f"   - None/missing categories: {len(none_categories)}")
    
    # Show unique category names
    unique_categories = shopee_products['product_category'].dropna().unique()
    print(f"   - Total unique categories: {len(unique_categories)}")
    
    print(f"\n📋 Sample Shopee Categories (properly matched):")
    for i, category in enumerate(sorted(unique_categories)[:10]):
        if not str(category).startswith('Category_'):
            print(f"   ✅ {category}")
    
    if len(fallback_categories) > 0:
        print(f"\n⚠️ Products with fallback categories:")
        print(fallback_categories[['product_item_id', 'product_name', 'product_category']].head())
        
        # Show some actual category_ids from raw data to help debug
        print(f"\n🔍 Checking some category_ids from fallback categories:")
        for _, row in fallback_categories.head(3).iterrows():
            category_str = row['product_category']
            if category_str and category_str.startswith('Category_'):
                category_id = category_str.replace('Category_', '')
                print(f"   - Product {row['product_item_id']} has category_id: {category_id}")
    
    # Calculate success rate
    success_rate = (len(properly_matched) / len(shopee_products)) * 100 if len(shopee_products) > 0 else 0
    
    print(f"\n📈 Category Matching Success Rate: {success_rate:.1f}%")
    
    if success_rate >= 90:
        print("✅ EXCELLENT: Category matching is working very well!")
    elif success_rate >= 70:
        print("✅ GOOD: Category matching is working well!")
    elif success_rate >= 50:
        print("⚠️ MODERATE: Category matching has room for improvement")
    else:
        print("❌ POOR: Category matching needs significant improvement")
    
    return success_rate

if __name__ == "__main__":
    verify_category_fix()
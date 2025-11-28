"""
Simplified Customer Dimension Harmonization
==========================================

This module processes Lazada and Shopee orders and extracts customer information to create
the Dim_Customer dimension table. Strictly follows DIM_CUSTOMER_COLUMNS from config.

Key Requirements from Schema:
- customer_key: Internal surrogate ID (sequential)
- platform_customer_id: For Lazada = buyer_id, For Shopee = buyer_username (raw)
- buyer_segment: Calculated as 'New Buyer' or 'Returning Buyer'
- total_orders: Count of orders per platform_customer_id
- customer_since: Earliest order_date for each platform_customer_id
- last_order_date: Latest order_date for each platform_customer_id
- platform_key: Always 1 for Lazada, 2 for Shopee

For Shopee null usernames: Creates single "Deleted User" record
"""

import pandas as pd
import json
import numpy as np
from datetime import datetime, date
import os
import sys
import re

# Add the parent directory to sys.path to import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    DIM_CUSTOMER_COLUMNS,
    get_empty_dataframe,
    apply_data_types
)


def load_lazada_orders_raw():
    """Load raw Lazada orders from JSON file"""
    json_path = os.path.join(os.path.dirname(__file__), '..', 'Staging', 'lazada_orders_raw.json')
    
    try:
        with open(json_path, 'r', encoding='utf-8') as file:
            orders_data = json.load(file)
        print(f"✓ Loaded {len(orders_data)} Lazada orders")
        return orders_data
    except FileNotFoundError:
        print(f"⚠️ Lazada orders file not found: {json_path}")
        return []
    except json.JSONDecodeError as e:
        print(f"❌ Error parsing Lazada orders JSON: {e}")
        return []


def load_shopee_orders_raw():
    """Load raw Shopee orders from JSON file (yearly directories + fallback)"""
    staging_dir = os.path.join(os.path.dirname(__file__), '..', 'Staging')
    yearly_path = os.path.join(staging_dir, 'Shopee Staging Yearly')
    
    orders_data = []
    
    # Load from yearly directories
    if os.path.exists(yearly_path):
        print("📁 Loading Shopee orders from yearly directories...")
        for year_folder in sorted(os.listdir(yearly_path)):
            if year_folder.startswith('Shopee20'):
                year_path = os.path.join(yearly_path, year_folder)
                if os.path.isdir(year_path):
                    orders_file = os.path.join(year_path, 'shopee_orders_raw.json')
                    if os.path.exists(orders_file):
                        try:
                            with open(orders_file, 'r', encoding='utf-8') as file:
                                year_orders = json.load(file)
                            orders_data.extend(year_orders)
                            print(f"✅ Loaded {len(year_orders)} orders from {year_folder}")
                        except Exception as e:
                            print(f"⚠️ Error loading orders from {year_folder}: {e}")
        print(f"📊 Total from yearly directories: {len(orders_data)} orders")
    
    # Fallback to main staging directory if no yearly data found
    if not orders_data:
        json_path = os.path.join(staging_dir, 'shopee_orders_raw.json')
        try:
            with open(json_path, 'r', encoding='utf-8') as file:
                orders_data = json.load(file)
            print(f"✅ Fallback: Loaded {len(orders_data)} Shopee orders from main staging")
        except FileNotFoundError:
            print(f"⚠️ Shopee orders file not found: {json_path}")
        except json.JSONDecodeError as e:
            print(f"❌ Error parsing Shopee orders JSON: {e}")
    
    return orders_data


def extract_customers_from_lazada_orders(orders_data):
    """
    Extract unique customers from Lazada orders using buyer_id directly as platform_customer_id
    Strictly follows DIM_CUSTOMER_COLUMNS from config - no extra columns.
    
    Args:
        orders_data (list): Raw Lazada orders from API
        
    Returns:
        DataFrame: Harmonized customer dimension records following config schema
    """
    # Build buyer_id lookup from order items
    print("📋 Building buyer_id lookup from lazada_multiple_order_items_raw.json...")
    staging_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'Staging')
    order_items_file = os.path.join(staging_dir, 'lazada_multiple_order_items_raw.json')
    
    buyer_id_lookup = {}
    all_buyer_ids = set()
    
    if os.path.exists(order_items_file):
        with open(order_items_file, 'r', encoding='utf-8') as f:
            order_items_data = json.load(f)
        
        for order in order_items_data:
            order_id = str(order.get('order_id', ''))
            order_items = order.get('order_items', [])
            if order_items and len(order_items) > 0:
                buyer_id = order_items[0].get('buyer_id')
                if buyer_id:
                    buyer_id_lookup[order_id] = str(buyer_id)
                    all_buyer_ids.add(str(buyer_id))
        
        print(f"   ✓ Built lookup for {len(buyer_id_lookup)} orders with buyer_id")
        print(f"   ✓ Found {len(all_buyer_ids)} unique buyer_ids in order items")
    
    customer_stats = {}  # Track order statistics per customer
    processed_orders = set()  # Track processed orders to avoid duplicates
    
    # Process orders and extract customers using buyer_id only
    for order in orders_data:
        try:
            order_id = str(order.get('order_id', ''))
            if order_id in processed_orders:
                continue
            processed_orders.add(order_id)
            
            # Get buyer_id from order items lookup (this is our platform_customer_id)
            buyer_id = buyer_id_lookup.get(order_id)
            if not buyer_id:
                continue  # Skip orders without buyer_id
                
            platform_customer_id = str(buyer_id)  # Use buyer_id directly as platform_customer_id
            
            # Extract order date
            created_at = order.get('created_at', '')
            order_date = None
            if created_at:
                try:
                    order_date = datetime.strptime(created_at.split(' ')[0], '%Y-%m-%d').date()
                except ValueError:
                    order_date = date.today()
            else:
                order_date = date.today()
            
            # Track customer statistics
            if platform_customer_id not in customer_stats:
                customer_stats[platform_customer_id] = {
                    'order_count': 0,
                    'earliest_date': order_date,
                    'latest_date': order_date
                }
            
            # Update statistics
            customer_stats[platform_customer_id]['order_count'] += 1
            if order_date < customer_stats[platform_customer_id]['earliest_date']:
                customer_stats[platform_customer_id]['earliest_date'] = order_date
            if order_date > customer_stats[platform_customer_id]['latest_date']:
                customer_stats[platform_customer_id]['latest_date'] = order_date
            
        except Exception as e:
            print(f"⚠️ Error processing order {order.get('order_id', 'unknown')}: {e}")
            continue
    
    # Add orphaned buyer_ids that don't appear in orders
    processed_customer_ids = set(customer_stats.keys())
    orphaned_buyer_ids = all_buyer_ids - processed_customer_ids
    if orphaned_buyer_ids:
        print(f"📋 Adding {len(orphaned_buyer_ids)} orphaned buyer_ids from order items...")
        for buyer_id in orphaned_buyer_ids:
            customer_stats[buyer_id] = {
                'order_count': 1,  # Assume at least 1 order
                'earliest_date': date(2020, 1, 1),  # Default date
                'latest_date': date(2020, 1, 1)
            }
    
    # Create customer records following DIM_CUSTOMER_COLUMNS exactly
    customer_records = []
    for platform_customer_id, stats in customer_stats.items():
        # Determine buyer segment
        buyer_segment = "New Buyer" if stats['order_count'] == 1 else "Returning Buyer"
        
        customer_record = {
            'platform_customer_id': platform_customer_id,
            'buyer_segment': buyer_segment,
            'total_orders': stats['order_count'],
            'customer_since': stats['earliest_date'],
            'last_order_date': stats['latest_date'],
            'platform_key': 1  # Lazada platform
        }
        
        customer_records.append(customer_record)
    
    # Create DataFrame with only config columns
    customers_df = pd.DataFrame(customer_records)
    
    print(f"✓ Extracted {len(customers_df)} unique Lazada customers")
    
    return customers_df


def extract_customers_from_shopee_orders(orders_data):
    """
    Extract customer information from Shopee orders data using buyer_username directly as platform_customer_id
    Strictly follows DIM_CUSTOMER_COLUMNS from config - no extra columns.
    
    Args:
        orders_data (list): List of Shopee order records
        
    Returns:
        pd.DataFrame: DataFrame with customer information following config schema
    """
    customer_stats = {}  # Track order statistics per customer
    
    for order in orders_data:
        # Use buyer_username directly as platform_customer_id (no generation, no cleaning)
        buyer_username = order.get('buyer_username')  # Note: it's buyer_username in orders, buyer_user_name in payment details
        
        if not buyer_username or pd.isna(buyer_username) or str(buyer_username).strip() == '':
            # For privacy-deleted or missing usernames we keep a single 'Deleted User' placeholder.
            # Per request, Deleted User should have total_orders = 0 and no dates (NaT). We must
            # not count orders against this placeholder.
            platform_customer_id = "Deleted User"
            # Ensure an entry exists with zero counts and NaT dates, but do NOT increment order_count
            if platform_customer_id not in customer_stats:
                customer_stats[platform_customer_id] = {
                    'order_count': 0,
                    'earliest_date': pd.NaT,
                    'latest_date': pd.NaT
                }
            # Skip counting this order toward Deleted User
            continue
        else:
            platform_customer_id = str(buyer_username).strip()  # Use as-is, no transformation

        # Parse order date
        create_time = order.get('create_time', 0)
        if create_time:
            try:
                order_date = datetime.fromtimestamp(create_time).date()
            except (ValueError, TypeError, OSError):
                order_date = date.today()
        else:
            order_date = date.today()

        # Track customer statistics
        if platform_customer_id not in customer_stats:
            customer_stats[platform_customer_id] = {
                'order_count': 0,
                'earliest_date': order_date,
                'latest_date': order_date
            }

        # Update statistics
        customer_stats[platform_customer_id]['order_count'] += 1
        if order_date < customer_stats[platform_customer_id]['earliest_date']:
            customer_stats[platform_customer_id]['earliest_date'] = order_date
        if order_date > customer_stats[platform_customer_id]['latest_date']:
            customer_stats[platform_customer_id]['latest_date'] = order_date
    
    # Create customer records following DIM_CUSTOMER_COLUMNS exactly
    customers = []
    for customer_id, stats in customer_stats.items():
        # Determine buyer segment - special case for Deleted User
        if customer_id == "Deleted User":
            buyer_segment = "Unknown"
        else:
            buyer_segment = "New Buyer" if stats['order_count'] == 1 else "Returning Buyer"
        
        customer_record = {
            'platform_customer_id': customer_id,
            'buyer_segment': buyer_segment,
            'total_orders': stats['order_count'],
            'customer_since': stats['earliest_date'],
            'last_order_date': stats['latest_date'],
            'platform_key': 2  # Shopee platform
        }
        
        customers.append(customer_record)
    
    # Convert to DataFrame with only config columns
    customers_df = pd.DataFrame(customers)
    
    print(f"✓ Extracted {len(customers_df)} unique Shopee customers from {len(orders_data)} orders")
    print(f"  - Real customers: {len([c for c in customers_df['platform_customer_id'] if c != 'Deleted User'])}")
    print(f"  - Deleted users: {len([c for c in customers_df['platform_customer_id'] if c == 'Deleted User'])}")
    
    return customers_df


def save_dim_customer(df, filename='dim_customer.csv'):
    """
    Save customer dimension DataFrame to CSV file
    Strictly follows DIM_CUSTOMER_COLUMNS from config
    
    Args:
        df (DataFrame): Customer dimension data
        filename (str): Output filename
    """
    if df.empty:
        print("⚠️ No customer data to save")
        return
    
    # Ensure we only have the required columns from config
    required_columns = DIM_CUSTOMER_COLUMNS
    
    # Make sure all required columns exist (they should from our functions above)
    for col in required_columns:
        if col not in df.columns:
            print(f"⚠️ Missing required column: {col}")
            return
    
    # Keep only the required columns in the correct order
    df_final = df[required_columns].copy()
    
    # Apply data types
    df_final = apply_data_types(df_final, 'dim_customer')
    
    # Save to file
    output_path = os.path.join(os.path.dirname(__file__), '..', 'Transformed', filename)
    df_final.to_csv(output_path, index=False)
    
    print(f"✅ Saved {len(df_final)} customers to {filename}")
    print(f"📊 Columns: {list(df_final.columns)}")
    print(f"📊 Platform breakdown:")
    print(f"   - Lazada (platform_key=1): {len(df_final[df_final['platform_key'] == 1])}")
    print(f"   - Shopee (platform_key=2): {len(df_final[df_final['platform_key'] == 2])}")
    print(f"   - Deleted User records: {len(df_final[df_final['platform_customer_id'] == 'Deleted User'])}")


def harmonize_dim_customer():
    """
    Main function to harmonize customer dimension from both platforms
    Strictly follows DIM_CUSTOMER_COLUMNS from config schema
    """
    print("🔄 Starting customer dimension harmonization...")
    print(f"📋 Required columns: {DIM_CUSTOMER_COLUMNS}")
    
    # Initialize empty DataFrame with proper schema
    customers_df = get_empty_dataframe('dim_customer')
    
    # Process Lazada customers
    print("📋 Processing Lazada customers...")
    lazada_orders = load_lazada_orders_raw()
    if lazada_orders:
        lazada_customers = extract_customers_from_lazada_orders(lazada_orders)
        customers_df = pd.concat([customers_df, lazada_customers], ignore_index=True)
    
    # Process Shopee customers
    print("📋 Processing Shopee customers...")
    shopee_orders = load_shopee_orders_raw()
    if shopee_orders:
        shopee_customers = extract_customers_from_shopee_orders(shopee_orders)
        customers_df = pd.concat([customers_df, shopee_customers], ignore_index=True)
    
    # Generate customer_key (sequential numbering)
    if not customers_df.empty:
        # Sort by platform_key, then by platform_customer_id for consistent ordering
        customers_df = customers_df.sort_values(['platform_key', 'platform_customer_id']).reset_index(drop=True)
        
        # Generate sequential customer_key starting from 1 with platform decimals
        customers_df['customer_key'] = range(1, len(customers_df) + 1)
        customers_df['customer_key'] = customers_df['customer_key'].astype(str) + '.' + customers_df['platform_key'].astype(str)
        
        # Apply proper data types
        customers_df = apply_data_types(customers_df, 'dim_customer')
        
        print(f"✅ Successfully harmonized {len(customers_df)} customers")
        print(f"   - Lazada customers: {len(customers_df[customers_df['platform_key'] == 1])}")
        print(f"   - Shopee customers: {len(customers_df[customers_df['platform_key'] == 2])}")
        
        # Validate and save
        save_dim_customer(customers_df)
        
        return customers_df
    else:
        print("⚠️ No customer data found")
        return customers_df


if __name__ == "__main__":
    harmonize_dim_customer()
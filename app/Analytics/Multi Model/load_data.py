import os
import sys
import pandas as pd
from datetime import date, timedelta, datetime
from supabase import create_client, Client
from supabase.client import ClientOptions 

# --- Setup ---
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..', '..')) 

if project_root not in sys.path:
    sys.path.append(project_root)

try:
    from config import SUPABASE_URL, SUPABASE_KEY
except ImportError:
    print("Warning: config.py not found. Using placeholders.")
    SUPABASE_URL = "http://placeholder.url"
    SUPABASE_KEY = "placeholder-key"
    
MAX_ROWS = 100000
MAX_RETRIES = 3 
_CACHED_SALES_DATA = None 

# Updated to match the NEW SQL function output
RPC_COLUMNS = [
    'ds',              # Date
    'y',               # Revenue
    'total_orders',
    'total_units',
    'aov',
    'promo_intensity',
    'units_per_order'
]

def get_supabase_client() -> Client:
    """Initializes and returns the Supabase client."""
    supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY, options=ClientOptions(schema="la_collections"))
    return supabase

def _fill_missing_dates(df: pd.DataFrame, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Generates a complete date range between start and end.
    Fills missing days (zero sales) with 0.
    """
    # 1. Convert inputs to datetime
    s_date = pd.to_datetime(start_date)
    # Subtract 1 day from end_date because SQL query uses '< end_date' (exclusive),
    # but pandas date_range is inclusive.
    e_date = pd.to_datetime(end_date) - timedelta(days=1)
    
    # 2. Create Master Date Index
    full_idx = pd.date_range(start=s_date, end=e_date, freq='D')
    
    # 3. Prepare DataFrame for Reindexing
    if not df.empty:
        # Ensure ds is datetime
        df['ds'] = pd.to_datetime(df['ds'])
        df = df.set_index('ds')
    
    # 4. Reindex and Fill
    # This adds rows for missing dates and fills them with 0
    df_filled = df.reindex(full_idx).fillna(0)
    
    # 5. Reset index to bring 'ds' back as a column
    df_filled.index.name = 'ds'
    return df_filled.reset_index()

def _process_dataframe(data: list) -> pd.DataFrame:
    """Standard processing steps for data frames returned by Supabase RPCs."""
    df = pd.DataFrame(data)

    if df.empty:
        return pd.DataFrame(columns=RPC_COLUMNS)

    # Ensure numeric columns are actually numeric
    numeric_cols = ['y', 'total_orders', 'total_units', 'aov', 'promo_intensity', 'units_per_order']
    for col in numeric_cols:
        if col in df.columns:
            # Coerce errors converts non-numbers to NaN, then fillna(0) turns them to 0.0
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    if 'ds' in df.columns:
        df['ds'] = pd.to_datetime(df['ds'])
        df = df.sort_values('ds').reset_index(drop=True)
    
    return df


def load_base_sales_data(start_date='2022-10-05', end_date='2025-10-31', platform='lazada'):
    """
    Fetches daily sales metrics, fills missing dates with 0, and caches the result.
    
    Args:
        start_date (str): YYYY-MM-DD
        end_date (str): YYYY-MM-DD. If None, defaults to Tomorrow (to include today's data).
        platform (str): Platform filter (default: 'lazada')
    """
    global _CACHED_SALES_DATA
    
    # If end_date is not provided, default to tomorrow (so < end_date covers today)
    if end_date is None:
        end_date = (date.today() + timedelta(days=1)).strftime('%Y-%m-%d')

    # Check Cache (Simple check - assumes params haven't changed. 
    # For robust caching, you might want to include params in the cache key)
    if _CACHED_SALES_DATA is not None and not _CACHED_SALES_DATA.empty:
        print("[INFO] Returning cached data.")
        return _CACHED_SALES_DATA.copy() 
    
    supabase = get_supabase_client()
    
    for attempt in range(MAX_RETRIES):
        try:
            print(f"STEP 1.1: Executing RPC: get_daily_sales_metrics_v2 ({start_date} to {end_date})...")
            
            # CALLING THE NEW SQL FUNCTION
            response = (
                supabase.rpc(
                    'get_daily_sales_metrics_v2', 
                    {
                        'start_date': start_date,
                        'end_date': end_date,
                        'p_platform_name': platform
                    }
                )
                .execute() # .range() is usually not needed for aggregates unless dataset is massive
            )

            # 1. Basic DataFrame creation
            df_raw = _process_dataframe(response.data)
            
            # 2. Fill Missing Dates (Zero Sales Days)
            df_filled = _fill_missing_dates(df_raw, start_date, end_date)
            
            # 3. Final Column Selection
            # Ensure all RPC columns exist (even if empty)
            for col in RPC_COLUMNS:
                if col not in df_filled.columns:
                    df_filled[col] = 0
            
            df_final = df_filled[RPC_COLUMNS].copy()
            
            print(f"[SUCCESS] Data loaded. Rows: {len(df_final)} (including zero-filled days).")
            
            _CACHED_SALES_DATA = df_final.copy() 
            return df_final
            
        except Exception as e:
            error_message = str(e)
            print(f"Supabase Query Error (Attempt {attempt+1}/{MAX_RETRIES}): {error_message}")
            if attempt == MAX_RETRIES - 1:
                return pd.DataFrame() 

    return pd.DataFrame()

# --- ADD THIS TO load_data.py ---

def load_transactional_data(start_date, end_date, platform='lazada'):
    """
    Fetches line-item level transactions using the 'get_sales_transactions' RPC.
    Used for basket analysis (e.g., how many orders are > 600).
    """
    supabase = get_supabase_client()
    
    try:
        print(f"   [Data] Fetching transactions from {start_date} to {end_date}...")
        
        response = (
            supabase.rpc(
                'get_sales_transactions', 
                {
                    'start_date': start_date, 
                    'end_date': end_date,
                    'p_platform_name': platform
                }
            ).execute()
        )
        
        df = pd.DataFrame(response.data)
        
        if df.empty:
            return pd.DataFrame()

        # Type conversion for critical math columns
        df['line_item_sales'] = pd.to_numeric(df['line_item_sales'], errors='coerce').fillna(0)
        df['order_date'] = pd.to_datetime(df['order_date'])
        
        return df

    except Exception as e:
        print(f"Error fetching transactions: {str(e)}")
        return pd.DataFrame()
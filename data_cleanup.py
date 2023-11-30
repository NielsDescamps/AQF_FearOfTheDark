import pandas as pd

def get_dfs_per_expiration_date(df,expiration_dates,price_dates,ticker):
    """
    Returns a didctionary containing one or more dataframes of option data related to a given ticker

    Input:
    - base dataframe containing the entire dataset 
    - 
    """

    # 1. Filter out unnecessary columns and rename others
    columns_to_retain = ['The Date of this Price',
        'Expiration Date of the Option',
        'C=Call, P=Put',
        'Strike Price of the Option Times 1000',
        'Highest Closing Bid Across All Exchanges',
        'Lowest  Closing Ask Across All Exchanges',
        'Implied Volatility of the Option',
        'Ticker Symbol',
        'Description of the Issuing Company',
        '(A)merican, (E)uropean, or ?']
    df = df[columns_to_retain]
    
    df['Strike Price of the Option Times 1000'] = df['Strike Price of the Option Times 1000']/1000
    
    new_column_names = ['PriceDate',
                        'ExpirationDate',
                        'Type',
                        'Strike',
                        'Bid',
                        'Ask',
                        'ImpliedVol',
                        'Ticker',
                        'Company',
                        'EorA']
    
    df.columns = new_column_names
    date_columns = ['PriceDate','ExpirationDate']
    for column in date_columns:
        df[column] = df[column].astype(str)
    
    filtered_df1 = pd.DataFrame()
    for specific_date in price_dates:
        date_filter = df['PriceDate'] == specific_date
        filtered_df1 = pd.concat([filtered_df1, df[date_filter]])
    
    # print('df before filtering on expiration_date',filtered_df1)
    
    dfs = dict.fromkeys(expiration_dates,None)
    
    for expiration_date in expiration_dates:
        date_filter = filtered_df1['ExpirationDate'] == expiration_date
        filtered_df = pd.DataFrame()
        filtered_df = pd.concat([filtered_df, filtered_df1[date_filter]])

        filtered_df['PriceDate'] = pd.to_datetime(filtered_df['PriceDate'], format='%d.%m.%Y')
        filtered_df['ExpirationDate'] = pd.to_datetime(filtered_df['ExpirationDate'], format='%d.%m.%Y')

        filtered_df = filtered_df[filtered_df['Ticker'] == ticker]
        dfs[expiration_date] = filtered_df
        # print('filtered df: ',filtered_df)
    
    return dfs

def calc_params(dfs,exp_date,S0,r,q):
    """
    Takes dataframe containing option data for one price date and expiration date 
    - calculate the Midquote and store in seperate column
    - calculate time to maturity [years - 250 days]

    Returns reduced dataframe with only relevant columns for calibration:
    - Stock price at time 0: S0
    - interest rate: r
    - dividend yield: q
    - strike price: K
    - time to maturity: TTM
    - market price (midquote): MQ
    
    """
    df = dfs[exp_date]
    df_out = df.copy()
    print(df_out)
    
    # 1. Calculate time to maturity TTM
    df_out['TTM'] = (df['ExpirationDate'] - df['PriceDate']).dt.days
    df_out['TTM'] = df_out['TTM']/252
    
    # 2. Calculate Miquote
    df_out['MQ'] = (df_out['Ask'] + df_out['Bid'])/2
    
    
    # 3. Filter for unique combinations of strike and maturity 
    df_out = df_out.drop_duplicates(subset=['Strike','TTM','Type'], keep='last')
    
    print(df_out.shape)
    
    return df_out


data = pd.read_csv('data_XNG/data_options.csv')
exp_dates = ['16.01.2010', '17.04.2010']
exp_date = '16.01.2010'
price_dates = ['04.01.2010']
ticker = 'XNG'

dfs = get_dfs_per_expiration_date(data,exp_dates,price_dates,ticker)
df_calc = calc_params(dfs,exp_date,1,1,1)
print(df_calc)
print('output: ',df_calc.shape)



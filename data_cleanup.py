import pandas as pd

def filter_data(data,expiration_date,price_date,ticker,type):
    """
    Returns a dataframe with filtered data

    Output columns:
        - 'PriceDate': date of the price
        - 'ExpirationDate'
        - 'Type' : C=call; P=Put
        - 'Strike'
        - 'Bid'
        - 'Ask'
        - 'ImpliedVol'
        - 'Ticker'
        - 'Company'
        - 'EorA'

    Input: dataframe containing the full dataset
 
    """

    df = data.copy()

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
    

    # 2. Filter for the date of pricing, expiration date, ticker and type
    df = df[df['PriceDate'] == price_date]
    df = df[df['ExpirationDate'] == expiration_date]
    df = df[df['Ticker'] == ticker]

    if type == 0:
        Type = 'C'
    elif type == 1:
        Type = 'P'
    
    df = df[df['Type']==Type]
    df = df.reset_index(drop=True)

    # 3. Convert dates to datetime objects
    df['PriceDate'] = pd.to_datetime(df['PriceDate'], format='%d.%m.%Y')
    df['ExpirationDate'] = pd.to_datetime(df['ExpirationDate'], format='%d.%m.%Y')
    
    # 4. Set Ask to zero if equal to 5
    # df.loc[df['Ask'] == 5, 'Ask'] = 0

    return df

def calc_params(df,S0,r,q):
    """
    Takes dataframe containing option data for one price date and expiration date 
    - calculate the Midquote and store in seperate column
    - calculate time to maturity [years - 250 days]
    - remove duplicates wrt strike, TTM and type

    Returns dataframe with only relevant columns for calibration:
    - Stock price at time 0: 'S0'
    - interest rate: 'R'
    - dividend yield: 'Q'
    - strike price: 'Strike'
    - time-to-maturity: 'TTM' (expressed in years assuming 252 trading days per year)
    - market price (midquote): 'MQ'
    
    """
    df_out = df.copy()
    
    # 1. Calculate time to maturity TTM
    df_out['TTM'] = (df['ExpirationDate'] - df['PriceDate']).dt.days
    df_out['TTM'] = df_out['TTM']/252
    
    # 2. Calculate Miquote
    df_out['MQ'] = (df_out['Ask'] + df_out['Bid'])/2
    
    # 3. Filter for unique combinations of strike and maturity 
    df_out = df_out.drop_duplicates(subset=['Strike','TTM','Type'], keep='last')

    # 4. Add interest rate, dividend yield and initial index price to the dataframe
    df_out['R'] = r
    df_out['Q'] = q
    df_out['S0'] = S0
    df_out = df_out.reset_index(drop=True)
    
    return df_out

# data = pd.read_csv('data_XNG/data_options.csv')
# exp_date = '16.01.2010' # or  '17.04.2010'
# price_date = '04.01.2010'
# ticker = 'XNG' #or BTK, XBD and MSH (use XNG, others have less quotes per day)
# type = 1
# df = filter_data(data,exp_date,price_date,ticker,type)

# S0 = 100
# r=0.05
# q=0.02
# df_calc = calc_params(df,S0,r,q)
# print(df_calc)
# print('output shape: ',df_calc.shape)



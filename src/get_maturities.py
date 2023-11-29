from datetime import datetime

def get_maturities(expiry_dates, date_of_retrieval):
    exp_dates = [datetime.strptime(date, '%Y-%m-%d') for date in expiry_dates]
    retrieval_date = datetime.strptime(date_of_retrieval, '%Y-%m-%d')
    T_datetime = [(date - retrieval_date).days for date in exp_dates]
    T = [days / 365 for days in T_datetime]
    return T
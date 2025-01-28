import pandas
import csv

def median_of_dataset(data):
    if len(data) % 2 != 0:
        return int(len(data)/2)
    else:
        return len(data)/2
    
def low_inter_quart_range(quarter_one, quarter_three, min_value):
    iqr = quarter_three - quarter_one
    low_bound = quarter_one - 1.5*iqr
    if low_bound < min_value:
        low_bound = min_value
    return low_bound

def high_inter_quart_range(quarter_one, quarter_three, max_value):
    iqr = quarter_three - quarter_one
    high_bound = quarter_three + 1.5*iqr
    if high_bound > max_value:
        high_bound = max_value()
    return high_bound

def in_ascending_order(data):
    for index in range(len(data)-1):
        for index_two in range(len(data)-1-index):
            if data[index_two] > data[index_two +1]:
                _ = data[index_two]
                data[index_two] = data[index_two + 1]
                data[index_two + 1] = _
    return data

def avg_daily_range(filepath):
    data = pandas.read_csv(filepath)
    record_num = len(data.values)
    sum = 0.0
    for price in data['Price Range']:
        sum += float(price)
    return sum/record_num

def standard_dev(avg_range, all_range):
    """
    σ = ( Σ(x - μ)² / N )^(1/2)
    """
    sum = 0
    total = len(all_range)
    for a_range in all_range:
        sum += (a_range - avg_range)**2
    std = (sum/total)**(1/2)
    return std

def wide_threshold(stand_dev, avg_range):
    threshold = avg_range + (stand_dev*1.5)
    return threshold

def narrow_threshold(stand_dev, avg_range):
    threshold = avg_range - (stand_dev*1.5)
    return threshold

def cal_vol_price_trend(close, vol_col):
    """
    VPT Today = VPT Yesterday + (Volume Today * ((Today's Close - Yesterday's Close)/Yesterday's Close)) 
    Traits:
        VPT Today = calcualted volume-price trend value for the current day
        VPT Yesterday = trading volume for the current day
        Today's Close = closing price of the sercurity for the current day
        Yesterday's Close = closing price of teh security from the previous day
        Volume Today = volume column for each row
    """
    vpt_array = []
    vpt_yesterday = 0 # use is similar to a temp variable
    vol_counter = 0
    # loop here
    for index in range(1, len(close)):
        today_close = close[index]
        yesterday_close = close[index-1]
        stock_change = today_close - yesterday_close
        percent_stock = stock_change/yesterday_close
        vpt_today = vpt_yesterday + (vol_col[vol_counter] * percent_stock) # per record
        vpt_array.append(vpt_today)
        vpt_yesterday = vpt_today
        vol_counter += 1

    # data frame
    vpt_dataframe = pandas.DataFrame({"VPT": vpt_array})
    # csv
    vpt_dataframe.to_csv("Programmed/VPT_DATA.csv", index = False)


def cal_on_balence_vol(close, vol_col):
    """
    Three conditions:
        Today's Close > Yesterday's Close:  Today = Yesterday + Volume
        Today's Close < Yesterday's Close:  Today = Yesterday - Volume
        Today's Close = Yesterday's Close:  Today = Yesterday
        Traits:
            Today's OBV = calculated On-Balance Volume value for the current day
            Yesterday's OBV = calculated On-Balence Volume value from the previous day
            Today's Volume = total trading volume for the current day
            Today's Close = closing price of the sercurity for the current day
            Yesterday's Close = closing price of the security from the previous day
    """
    obv_array = []
    obv_yesterday = 0

    # loop here
    for index in range(1, len(close)):
        today = close[index]
        yesterday = close[index -1]
        if today > yesterday:
            obv_today = obv_yesterday + vol_col[index] # per reocrd
            
        elif today < yesterday:
            obv_today = obv_yesterday - vol_col[index] # per  record
            
        else:
            obv_today = obv_yesterday # per record
        obv_array.append(obv_today)
        obv_yesterday = obv_today
    # put itno a dataframe
    obv_dataframe = pandas.DataFrame({"OBV": obv_array})
    # put into a csv
    obv_dataframe.to_csv("Programmed/OBV_DATA.csv", index = False)

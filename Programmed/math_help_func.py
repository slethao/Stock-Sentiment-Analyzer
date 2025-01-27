import pandas

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

def wide_threshold(stand_dev):
    threshold = stand_dev*1.5
    return threshold

def narrow_threshold(stand_dev):
    threshold = stand_dev*1.5*-1
    return threshold
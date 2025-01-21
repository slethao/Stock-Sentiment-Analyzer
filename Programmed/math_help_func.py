def median_of_dataset(data):
    if len(data) % 2 != 0:
        return int(len(data)/2)
    else:
        return len(data)/2
    
def low_inter_quart_range(quarter_one, quarter_three):
    iqr = quarter_three - quarter_one
    low_bound = quarter_one - 1.5*iqr
    return low_bound

def high_inter_quart_range(quarter_one, quarter_three):
    iqr = quarter_three - quarter_one
    high_bound = quarter_three + 1.5*iqr
    return high_bound

def in_ascending_order(data):
    for index in range(len(data)-1):
        for index_two in range(len(data)-1-index):
            if data[index_two] > data[index_two +1]:
                temp = data[index_two]
                data[index_two] = data[index_two + 1]
                data[index_two + 1] = temp
    return data
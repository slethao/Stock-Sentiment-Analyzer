import matplotlib.pyplot
import pandas
import tensorflow
import plot_transfer as img

"""
This method finds the median of a sorted dataset based on its index
Args:
    data = the dataset given from a csv.
Returns:
    returns the index of the element that represents the median
Raises:
    Nothing
Implemented:
    If written in another file:
        import math_help_func.py as math

        math.median_of_dataset()

    If written in the same file the method was invokded:
        median_of_dataset()
"""
def median_of_dataset(data):
    if len(data) % 2 != 0:
        return int(len(data)/2)
    else:
        return len(data)/2


"""
This method calculates the lower interquartile range of the dataset.
Args:
    quarter_one = The median of the lower end of the dataset
    quarter_three = The median of the higher end of the dataset
    min_value = the minimum value
Returns:
    returns the lower interquartile range
Raises:
    Nothing
Implemented:
    If written in another file:
        import math_help_func.py as math

        math.low_inter_quart_range(quarter_one, quarter_three, min_value)

    If written in the same file the method was invokded:
        low_inter_quart_range(quarter_one, quarter_three, min_value)
"""
def low_inter_quart_range(quarter_one, quarter_three, min_value):
    iqr = quarter_three - quarter_one
    low_bound = quarter_one - 1.5*iqr
    if low_bound < min_value:
        low_bound = min_value
    return low_bound


"""
This method is used to calculate the higher interquartile range.
Args:
    quarter_one = The median of the lower end of the dataset
    quarter_three = The median of the higher end of the dataset
    min_value = the minimum value
Returns:
    returns the higher interquartile range
Raises:
    Nothing
Implemented:
    If written in another file:
        import math_help_func.py as math

        math.high_inter_quart_range(quarter_one, quarter_three, max_value)

    If written in the same file the method was invokded:
        high_inter_quart_range(quarter_one, quarter_three, max_value)
"""
def high_inter_quart_range(quarter_one, quarter_three, max_value):
    iqr = quarter_three - quarter_one
    high_bound = quarter_three + 1.5*iqr
    if high_bound > max_value:
        high_bound = max_value()
    return high_bound


"""
This method display the dataset in ascending order (going from low to high).
Args:
    data = the given dataset
Returns:
    returns a sorted dataset in ascending order
Raises:
    Nothing
Implemented:
    If written in another file:
        import math_help_func.py as math

        math.in_ascending_order(data)

    If written in the same file the method was invokded:
        in_ascending_order(data)
"""
def in_ascending_order(data):
    for index in range(len(data)-1):
        for index_two in range(len(data)-1-index):
            if data[index_two] > data[index_two +1]:
                _ = data[index_two]
                data[index_two] = data[index_two + 1]
                data[index_two + 1] = _
    return data


"""
The method is used to calculate teh average daily range.
Args:
    filepath = the location where the csv file holding the dataset
Returns:
    returns the value that represent the avergae daily range
Raises:
    Nothing
Implemented:
    If written in another file:
        import math_help_func.py as math

        math.avg_daily_range(filepath)

    If written in the same file the method was invokded:
        avg_daily_range(filepath)
"""
def avg_daily_range(filepath):
    data = pandas.read_csv(filepath)
    record_num = len(data.values)
    sum = 0.0
    for price in data['Price Range']:
        sum += float(price)
    return sum/record_num


"""
The method calculates the standard deviation.
Args:
    avg_range = the average value of all the ranges in the dataset
    all_range = conatains all the ranges that were found in all the datasets
Returns:
    returns the value that represent the standard deviation 
Raises:
    Nothing
Implemented:
    If written in another file:
        import math_help_func.py as math

        math.standard_dev(avg_range, all_range)

    If written in the same file the method was invokded:
        standard_dev(avg_range, all_range)
"""
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


"""
The method calcualtes the wide threshold.
Args:
    stand_dev = represents the standard deviation of a dataset
    avg_range = represents the avearge range of a dataset
Returns:
    returns the value that represent the wide threshold
Raises:
    Nothing
Implemented:
    If written in another file:
        import math_help_func.py as math

        math.wide_threshold(stand_dev, avg_range)

    If written in the same file the method was invokded:
        wide_threshold(stand_dev, avg_range)
"""
def wide_threshold(stand_dev, avg_range):
    threshold = avg_range + (stand_dev*1.5)
    return threshold


"""
This mehtod calculates the narrow threshold.
Args:
    stand_dev = represents the standard deviation of a dataset
    avg_range = represents the avearge range of a dataset
Returns:
    returns the value that represent the narrow threshold
Raises:
    Nothing
Implemented:
    If written in another file:
        import math_help_func.py as math

        math.narrow_threshold(stand_dev, avg_range)

    If written in the same file the method was invokded:
        narrow_threshold(stand_dev, avg_range)
"""
def narrow_threshold(stand_dev, avg_range):
    threshold = avg_range - (stand_dev*1.5)
    return threshold


"""
This method calculates the Volume Price Trend (VPT)
Args:
    close = the value of Today's Close Price
    vol_col = the vaule of Today's Volume
Returns:
    Nothing
Raises:
    Nothing
Implemented:
    If written in another file:
        import math_help_func.py as math

        math.cal_vol_price_trend(close, vol_col)

    If written in the same file the method was invokded:
        cal_vol_price_trend(close, vol_col)
"""
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
    day_array = []
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
        day_array.append(vol_counter)

    # data frame
    vpt_dataframe = pandas.DataFrame({"VPT": vpt_array})
    #@TODO: create candstick here
    matplotlib.pyplot.plot(day_array,vpt_array)
    fig = matplotlib.pyplot.gcf()
    writer = tensorflow.summary.create_file_writer("Programmed/logs") # Create a writer 
    with writer.as_default():
        tensorflow.summary.image("matplotlib_plot_02", img.plot_to_image(fig), step=0)
    # csv
    vpt_dataframe.to_csv("Stock-Sentiment-Analyzer/Programmed/Calculations/VPT_DATA.csv", index = False)


"""
This method calculates the On-Balence Volume (OBV).
Args:
    close = Today's Close Price
    vol_col = Today's Volume
Returns:
    Nothing
Raises:
    Nothing
Implemented:
    If written in another file:
        import math_help_func.py as math

        math.cal_on_balence_vol(close, vol_col)

    If written in the same file the method was invokded:
        cal_on_balence_vol(close, vol_col)
"""
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
    day_arry = []
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
        day_arry.append(index+1)
    # put itno a dataframe
    obv_dataframe = pandas.DataFrame({"OBV": obv_array})
    
    matplotlib.pyplot.plot(day_arry, obv_array)
    fig = matplotlib.pyplot.gcf() # Get the current figure

    writer = tensorflow.summary.create_file_writer("Programmed/logs") # Create a writer 
    with writer.as_default():
        tensorflow.summary.image("matplotlib_plot_03", img.plot_to_image(fig), step=0)
    # put into a csv
    obv_dataframe.to_csv("Stock-Sentiment-Analyzer/Programmed/Calculations/OBV_DATA.csv", index = False)


"""
This method is used the calculate the mean.
Args:
    data = the entrie dataset
    group = the given group that the user wants to see
Returns:
    return the value of the mean
Raises:
    Nothing
Implemented:
    If written in another file:
        import math_help_func.py as math

        math.calculate_mean(data, group)

    If written in the same file the method was invokded:
        calculate_mean(data, group)
"""
def calculate_mean(data, group):
    records = len(data[group])
    sum = 0
    for parts in data[group]:
        sum += float(parts)
    return sum/records


"""
This calculates the sum of squared errors (SSE).
Args:
    data = contains the entire dataset
    index = the number that refernces where a particular record is in dataset
    group = the given group the user wants to see
Returns:
    return the value of the sum of squared errors
Raises:
    Nothing
Implemented:
    If written in another file:
        import math_help_func.py as math

        math.calculate_sse(data, index, group)

    If written in the same file the method was invokded:
        calculate_sse(data, index, group)
"""
def calculate_sse(data, index, group):
    sse = (float(data[group][index]))**2
    return sse


"""
This method calculates the total sum of squares (TSS)
Args:
    data = the entire datset
    index = the location of a record within the datset
    mean = the mean value of the entrie dataset
    group = the given group the user wants to see
Returns:
    return the value of the total sum of squares
Raises:
    Nothing
Implemented:
    If written in another file:
        import math_help_func.py as math

        math.calculate_tss(data, index, mean, group)

    If written in the same file the method was invokded:
        calculate_tss(data, index, mean, group)
"""
def calculate_tss(data, index, mean, group):
    tss = (data[group][index]- mean)**2
    return tss
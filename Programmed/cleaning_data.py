import pandas
from sklearn.impute import KNNImputer
import math_help_func as math
import csv

"""
This method rearranges the groups in order to make sure they are in propper alignment.
Args:
    Nothing

Returns:
    Nothing

Raises:
    Nothing

Implemented:
    If written in another file:
        import cleaning_data.py as clean

        clean.alter_groups()

    If written in the same file the method was invokded:
        alter_groups()
"""
def alter_groups():
    with open('Stock-Sentiment-Analyzer/Programmed/Standard filter/Bronze/NVIDIA_STOCK.csv') as content:
        groups = content.readline().rstrip("\n")
        extra_line = content.readline().rstrip("\n")
        date_line = content.readline().rstrip("\n")
        date_group = date_line.split(",")[0]
        new_groups = groups.replace("Price", date_group)

        with open('Stock-Sentiment-Analyzer/Programmed/Standard Filter/Silver/NVIDIA_STOCK_02.csv', 'w') as new_file:
            new_file.write(f"{new_groups}\n")
            for line in content:
                new_file.write(line)


"""
This method removes the outliers that are within the dataset.
Args:
    group = the given group the user wants to see
    file_path = the location in which the dataset can be accessed
Returns:
    returns a list that contains all the values that are greater than the lower bound
    calculated and less than the higher bound calculated
        NOTE Remove Outliers Interquartile Range Formulas
            lower bound: Q1 -1.5*IQR
            upper bound: Q3 + 1.5*IQR
            IQR: (Q3 - Q1)
Raises:
    Nothing
Implemented:
    If written in another file:
        import cleaning_data.py as clean

        clean.outliers_removed(group, file_path)

    If written in the same file the method was invokded:
        outliers_removed(group, file_path)
"""
def outliers_removed(group, file_path):
    low_bound = 0
    high_bound = 0
    all_data = []
    with open(file_path, 'r') as alter_data:
        data_list = alter_data.readlines()
        for index in range(1, len(data_list)):
            group_data = data_list[index].rstrip("\n").split(",")[group]
            all_data.append(float(group_data))
        ordered_data = math.in_ascending_order(all_data)
        if len(ordered_data) % 2 == 0: # even
            index_median = math.median_of_dataset(ordered_data)
            quart_one = index_median//2
            quarter_one = ordered_data[quart_one]
            quarter_three = ordered_data[index_median+quart_one]
            low_bound = math.low_inter_quart_range(quarter_one, quarter_three,  ordered_data[0])
            high_bound = math.high_inter_quart_range(quarter_one, quarter_three, ordered_data[len(ordered_data)-1])
        else: # odd
            index_median = math.median_of_dataset(ordered_data)
            quart_one = index_median//2
            quarter_one = (ordered_data[quart_one] + ordered_data[quart_one+1])/2
            print(quarter_one)
            quarter_three = (ordered_data[index_median+quart_one] + ordered_data[index_median+quart_one+1])/2
            low_bound = math.low_inter_quart_range(quarter_one, quarter_three,  ordered_data[0])
            high_bound = math.high_inter_quart_range(quarter_one, quarter_three, ordered_data[len(ordered_data)-1])
        # print("low bound", low_bound)
        # print("high bound", high_bound)
        return [x for x in all_data if low_bound <= x <= high_bound]


"""
This method gets rid of missing values found in the dataset.
Args:
    file_path = the location in which the dataset can be accessed
Returns:
    Nothing
Raises:
    Nothing
Implemented:
    If written in another file:
        import cleaning_data.py as clean

        clean.rid_missing_values(filepath)

    If written in the same file the method was invokded:
        rid_missing_values(filepath)
"""
def rid_missing_values(filepath):
    overall_data = ""
    blank_filler_algo = KNNImputer(n_neighbors=10, weights="uniform")
    with open(filepath, 'r') as new_content:
        for line in new_content:
            end_of_date = line.index(",")
            overall_data += line[end_of_date+1:]    
        with open('Stock-Sentiment-Analyzer/Programmed/Standard Filter/Gold/NVIDIA_STOCK_04.csv', 'w') as new_file:
            new_file.write(overall_data)
    
    data = pandas.read_csv('Stock-Sentiment-Analyzer/Programmed/Standard Filter/Gold/NVIDIA_STOCK_04.csv')
    result = blank_filler_algo.fit_transform(data)
    new_dataframe = pandas.DataFrame(result, columns=data.columns)
    new_dataframe.to_csv("Stock-Sentiment-Analyzer/Programmed/Standard Filter/Gold/NVIDIA_STOCK_04.csv", index = False)


"""
This is the method filters all the outliers and missing values. In addtion to making sure
the raw dataset's attributes are propperly aligned.
Args:
    Nothing
Returns:
    returns a filter dataset that does not have any outliers or missing values
Raises:
    Nothing
Implemented:
    If written in another file:
        import cleaning_data.py as clean

        clean.main()

    If written in the same file the method was invokded:
        main()
"""
def main():
    alter_groups()
    with open('Stock-Sentiment-Analyzer/Programmed/Standard Filter/Silver/NVIDIA_STOCK_02.csv', 'r') as data:
        gathered = {"Date": [],"Adj Close": [],"Close": [],"High": [],"Low": [],"Open": [],"Volume": []}
        group_list = data.readline().rstrip("\n").split(",")
        data_list = data.readlines()
        for index in range(len(group_list)):
            if index > 0:
                collected = outliers_removed(index, 'Stock-Sentiment-Analyzer/Programmed/Standard Filter/Silver/NVIDIA_STOCK_02.CSV')
                gathered[group_list[index]] = collected
            else:
                for line in data_list:
                    data = line.split(",")
                    gathered["Date"].append(data[0])
        
        with open("Stock-Sentiment-Analyzer/Programmed/Standard Filter/Gold/NVIDIA_STOCK_03.csv", "w") as final_data:
            writer = csv.DictWriter(final_data, fieldnames=gathered.keys())
            writer.writeheader()
            rows = [dict(zip(gathered, t)) for t in zip(*gathered.values())]
            writer.writerows(rows)
    rid_missing_values("Stock-Sentiment-Analyzer/Programmed/Standard Filter/Gold/NVIDIA_STOCK_03.csv")
    return gathered

    #NOTE make into a class.. 

main()
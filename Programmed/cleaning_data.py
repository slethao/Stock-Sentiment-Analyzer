import pandas
from sklearn.ensemble import IsolationForest
from sklearn.impute import KNNImputer
import math_help_func as math

"""
(summary of filtering)
"""

"""
@TODO: fixed the group name to be the date
"""
def alter_groups():
    with open('Programmed/NVIDIA_STOCK.csv') as content:
        groups = content.readline().rstrip("\n")
        extra_line = content.readline().rstrip("\n")
        date_line = content.readline().rstrip("\n")
        date_group = date_line.split(",")[0]
        new_groups = groups.replace("Price", date_group)

        with open('Programmed/NVIDIA_STOCK_02.csv', 'w') as new_file:
            new_file.write(f"Hash Value,{new_groups}\n")
            for line in content:
                new_file(line)

    return f"{new_groups.replace("Date,", "")}\n"

"""
@TODO: missing values
"""
def ridding_missing_values():
    overall_data = ""
    blank_filler_algo = KNNImputer(n_neighbors=10, weights="uniform")
    with open('Programmed/NVIDIA_STOCK_02.csv', 'r') as new_content:
        for line in new_content:
            end_of_hash = line.index(",")
            alter_line = line[end_of_hash+1:]
            end_of_date = alter_line.index(",")
            overall_data += alter_line[end_of_date+1:]
        
        with open('Programmed/NVIDIA_STOCK_03.csv', 'w') as new_file:
            new_file.write(overall_data)

    data = pandas.read_csv('Programmed/NVIDIA_STOCK_03.csv')
    result = blank_filler_algo.fit_transform(data)
    new_dataframe = pandas.DataFrame(result, columns=data.columns)
    new_dataframe.to_csv("Programmed/NVIDIA_STOCK_04.csv", index = False)


"""
@TODO: remove outliers Interquartile Range
lower bound: Q1 -1.5*IQR
upper bound: Q3 + 1.5*IQR
IQR: (Q3 - Q1)
"""
def outliers_removed(group, file_path):
    low_bound = 0
    high_bound = 0
    all_data = []
    with open(file_path, 'r') as alter_data:
        data_list = alter_data.readlines()
        for index in range(1, len(data_list)):
            group_data = data_list[index].split(",")[group]
            all_data.append(float(group_data))
        ordered_data = math.in_ascending_order(all_data)
        if len(ordered_data) % 2 == 0: # even
            index_median = math.median_of_dataset(ordered_data)
            quart_one = int(index_median/2)
            quarter_one = ordered_data[quart_one]
            quarter_three = ordered_data[index_median+quart_one]
            low_bound = math.low_inter_quart_range(quarter_one, quarter_three)
            high_bound = math.high_inter_quart_range(quarter_one, quarter_three)
        else: # odd
            index_median = math.median_of_dataset(ordered_data)
            quart_one = int(index_median/2)
            quarter_one = (ordered_data[quart_one] + ordered_data[quart_one+1])/2
            quarter_three = (ordered_data[index_median+quart_one] + ordered_data[index_median+quart_one+1])/2
            low_bound = math.low_inter_quart_range(quarter_one, quarter_three)
            high_bound = math.high_inter_quart_range(quarter_one, quarter_three)
        return [x for x in ordered_data if x >= low_bound and high_bound >= x]


def main():
    new_groups = alter_groups()
    ridding_missing_values()
    #TODO after the method works create a loop to call each group here
    #NOTE get started on the isolation tree
    #NOTE put hashing here
    """
    # index = 0 
            # hash_algo = 0
            for line in content:
                #hash_algo += 2**index
                new_file(line)
                #new_file.write(f"{hash_algo},{line}")
                #index += 1
    """
    with open('Programmed/NVIDIA_STOCK_05.csv', 'w') as final_data:
        print(new_groups.split())
        for index in range(len(new_groups.split())):
            adj_price = outliers_removed(index, 'Programmed/NVIDIA_STOCK_04.CSV')
        
        # print(adj_price)
        # print(close_price)
        # print(high_price)
        # print(lose_price)
        # print(open_price)
        # print(volume_price)
main()
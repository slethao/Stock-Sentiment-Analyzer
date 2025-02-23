import pandas

"""
This method combines all the predicted files into one csv file.
Args:
    Nonthing
Returns:
    Nonthing
Raises:
    Nonthing
Implementated:
    If written in another file:
        import files_help_func.py as file

        file.combining_files()

    If written in the same file the method was invokded:
        combining_files()

"""
def combining_files(): 
    # all the predicted data for ecah attribute in the orginal csv. (NVIDIA_STOCK.csv)
    adj_close_csv = pandas.read_csv("Programmed/Predicted Data/NVIDIA_STOCK_PREDICT_AdjClose.csv")
    close_csv = pandas.read_csv("Programmed/Predicted Data/NVIDIA_STOCK_PREDICT_Close.csv")
    high_csv = pandas.read_csv("Programmed/Predicted Data/NVIDIA_STOCK_PREDICT_High.csv")
    low_csv = pandas.read_csv("Programmed/Predicted Data/NVIDIA_STOCK_PREDICT_Low.csv")
    open_csv = pandas.read_csv("Programmed/Predicted Data/NVIDIA_STOCK_PREDICT_Open.csv")
    volume_csv = pandas.read_csv("Programmed/Predicted Data/NVIDIA_STOCK_PREDICT_Volume.csv")
    
    # propper group aliegnment to ensure that each group is initilized propperly
    adj_close_csv.columns = ["Date","Adj Close", "Guess Adj Close"]
    close_csv.columns = ["Date", "Close", "Guess Close"]
    high_csv.columns = ["Date", "High", "Guess High"]
    low_csv.columns = ["Date", "Low", "Guess Low"]
    open_csv.columns = ["Date", "Open", "Guess Open"]
    volume_csv.columns = ["Date", "Volume", "Guess Volume"]

    # this will hold the starting values of the first merged set
    final_file = pandas.merge(adj_close_csv, close_csv, on="Date", how="left")
    # the remaing attributes that need to be merged with the previous merged set above in line 39
    column_array = [high_csv, low_csv, open_csv, volume_csv]

    # iterate through array of attributes to then merge to the variable holding the value of all the merge sets per iteration
    for group in column_array:
        final_file = pandas.merge(final_file, group, on="Date", how="left")

    # store merged values into a csv file called "OVERALL_PREDICTION.csv"
    final_file.to_csv("Programmed/Predicted Data/OVERALL_PREDICTION.csv", index = False)


"""
The method displays the attributes of the given file.
Args:
    filepath = the path of where the file is stored
Returns:
    all the attributes that are in the csv
Raises:
    Nothing
Implemented:
    If written in another file:
        import files_help_func.py as file

        file.file_attributes()

    If written in the same file the method was invokded:
        file_attributes()
"""
def file_attributes(filepath):
    return pandas.read_csv(filepath).columns


"""
This method displays all the records in the csv.
Args:
    filepath = the path of where the file is stored
Returns:
    all the records that are in the csv
Raises:
    Nothing
Implemented:
    If written in another file:
        import files_help_func.py as file

        file.file_records()

    If written in the same file the method was invokded:
        file_records()
"""
def file_records(filepath):
    return pandas.read_csv(filepath).values
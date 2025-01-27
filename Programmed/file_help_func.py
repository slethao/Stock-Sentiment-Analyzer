import pandas

def combining_files(): #NOTE need files
    date_csv = pandas.read_csv("Programmed/NVIDIA_STOCK_03.csv", usecols=["Date"])
    adj_close_csv = pandas.read_csv("Programmed/NVIDIA_STOCK_PREDICT_AdjClose.csv")
    close_csv = pandas.read_csv("Programmed/NVIDIA_STOCK_PREDICT_Close.csv")
    high_csv = pandas.read_csv("Programmed/NVIDIA_STOCK_PREDICT_High.csv")
    low_csv = pandas.read_csv("Programmed/NVIDIA_STOCK_PREDICT_Low.csv")
    open_csv = pandas.read_csv("Programmed/NVIDIA_STOCK_PREDICT_Open.csv")
    volume_csv = pandas.read_csv("Programmed/NVIDIA_STOCK_PREDICT_Volume.csv")
    final_file = pandas.concat([date_csv, adj_close_csv, close_csv, high_csv, low_csv, open_csv, volume_csv], ignore_index=True)
    final_file.to_csv("Programmed/OVERALL_PREDICTION.csv", index = False)

def file_attributes(filepath):
    return pandas.read_csv(filepath).columns

def file_records(filepath):
    return pandas.read_csv(filepath).values
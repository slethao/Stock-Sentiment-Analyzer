import os
import sqlite3
import pandas

def put_into_db(csv_folder):
    data_base_file = "data_stock.db"
    connection = sqlite3.connect(data_base_file)
    for file in csv_folder:
        if file.endswith(".csv"):
            file_content = os.path.join(csv_folder, file)
            content_df = pandas.read_csv(file_content)
            table_name = file[:-4]
            content_df.to_sql(table_name, connection, if_exists = 'replace', index = False)
    connection.close()

def main():
    csv_dir_cal = "Stock-Sentiment-Analyzer/Programmed/Calculations"
    csv_dir_predict_data = "Stock-Sentiment-Analyzer/Programmed/Predicted Data"
    csv_dir_rss_feed = "Stock-Sentiment-Analyzer/Programmed/RSS Feeder"
    csv_dir_stand_filter = "Stock-Sentiment-Analyzer/Programmed/Standard Filter"
    put_into_db(csv_dir_cal)
    put_into_db(csv_dir_predict_data)
    put_into_db(csv_dir_rss_feed)
    put_into_db(csv_dir_stand_filter)

main()
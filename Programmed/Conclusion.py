import math_help_func as math
import pandas

class Conclusion:
    def __init__(self, filepath):
        self._csv_file = pandas.read_csv(filepath)
        self._today_price = self._csv_file["Close"].values
        self._yesterday_price = self._csv_file["Adj Close"].values
        self._high_stock = self._csv_file["High"].values
        self._low_stock = self._csv_file["Low"].values

    def calculate_daily_return(self): # daily return
        """
        Daily Return = (Today's Closing Price - Yesterday's Closing Price) / Yesterday's Closing 
        Today's Price = Close (regular trading session)
        Yesterday's Price = Adj Close (stock splits and dividends and help with long term analysis (price change behaviro))
            Example:
                yesterday price  = 140
                today price = 200

                (200 - 140)/140 = 0.428 (for the ratio) 
                                = 42.8 % (for the percentage *multipy by 100*)
        """
        daily_return = (self._today_price - self._yesterday_price) / self._yesterday_price
        return daily_return

    def price_change(self): # for stock change
        """
        High vol = strong buying or selling pressure
        Low vol = thin liquidity and increased risk
        1) put all ranges into a .csv and calculate the average daily range (done)
        2) calculate standard deviation (done)
        3) set the threshold (wide or narrow range) (done)
        4) current day's range for th (pending maybe put into a pie chart or something to show on tensor board)
        """
        low_stock_counter = 0
        all_ranges = []
        file_path = "Programmed/ALL_PRICE_RANGES.csv"
        for stock in self._high_stock:
            all_ranges.append(stock - self._low_stock[low_stock_counter])
            low_stock_counter += 1
        price_range_data_frame = pandas.DataFrame({"Price Range": all_ranges})
        price_range_data_frame.to_csv(file_path, index = False)
        avg = math.avg_daily_range(file_path)
        std = math.standard_dev(avg, all_ranges)
        #print(f"std {std}")
        # print(f"Avg price range: {avg}")
        for a_range in all_ranges:
            if math.wide_threshold(std, avg) < a_range: # wide compared to the average price range
                print("large than usual flucation")
                # print(f"wide: threshold {math.wide_threshold(std, avg)} and the samle ({a_range})")
            if math.narrow_threshold(std, avg) > a_range: # narrow compared to the average price range
                # print(f"narrow threshold {math.narrow_threshold(std,avg)} and the samle ({a_range})")
                print("smaller than usual flucation")

    def price_move(self): # for stock volume
        """
        price movement used in volume
        (plot (line graph) on the same chart VPT and OBV)
        # if uptrend = both are increase
        # if downtrend = both decrease
        # bullish divergence = price makes lower lows with VPT/OBV makes higher low
        # bearish divergence = price makes higher highs wiith VPT/OBV makes lower highs
        Graph:
            -> Close Price
            -> VPT
            -> OBV
        """
        vol_stock = self._csv_file["Volume"].values
        close_stock = self._csv_file["Close"].values
        # print(close_stock) # array
        #print(vol_stock) # array
        math.cal_vol_price_trend(close_stock, vol_stock)
        math.cal_on_balence_vol(close_stock, vol_stock)
    
    def price_cal_with_predict(self, all_info):
        """
        price change:
            price change = (Today's Predicted Close - Yesterday's Predicted Close)
        relationship between the price movements and predicted volume    
        """
        today_predict = pandas.read_csv(all_info)["Guess Close"]
        yesterday_predict = pandas.read_csv(all_info)["Adj Close"]
        all_price_change = []
        for index in range(len(today_predict)):
            price_change = today_predict[index] - yesterday_predict[index]
            all_price_change.append(price_change)
        price_change_dataframe = pandas.DataFrame({"predict price change": all_price_change})
        price_change_dataframe.to_csv("Programmed/PREDICTED_PRICE_CHANGE.csv")


    def range_cal_with_predict(self, all_info):
        """
        predicted price range:
            predicted range = predicted high - predicted low
        """
        predict_high = pandas.read_csv(all_info)["Guess High"].values
        predict_low = pandas.read_csv(all_info)["Guess Low"]
        all_predict_range = []
        for index in range(len(predict_high)):
            predicted_range = predict_high[index] - predict_low[index]
            all_predict_range.append(predicted_range)
        predict_range_dataframe = pandas.DataFrame({"predicted range": all_predict_range})
        predict_range_dataframe.to_csv("Programmed/PREDICTED_PRICE_RANGE.csv")


    def daily_return_with_predict(self, all_info):
        """
        daily return:
            Daily return = (Today's predicted close - yesterday's predicted close) / Yesterday's predicted close
        """
        today_predict = pandas.read_csv(all_info)["Guess Close"]
        yesterday_predict = pandas.read_csv(all_info)["Adj Close"]
        all_predict_returns = []
        for index in range(len(today_predict)):
            daily_return = (today_predict - yesterday_predict) / yesterday_predict
            all_predict_returns.append(daily_return)
        predict_return_dataframe = pandas.DataFrame({"Predicted daily return": all_predict_returns})
        predict_return_dataframe.to_csv("Programmed/PREDICTED_DAILY_RETURN.csv")


    def daily_mov_with_predict(self, all_info):
        """
        daily movement direction:
            if price change > 0 (price mov up)
            if pirce change < 0 (price move down)
            price change = 0 (no change)
        show on a graph
        """
        # read csv with the price change
        # call the prediction method store it in predict_change
        predict_changes = []
        all_moves = [price_up, price_down, no_change]
        for change in predict_changes:
            if change > 0:
                price_up += 1
            elif change < 0:
                price_down += 1
            else:
                no_change += 1


    def cal_mse(self, all_info):
        """
        evaluate model:
            mean squared error(MSE)
            = (1/n) * summation(predicted close - actual close)^2
        """
        predict_close = pandas.read_csv(all_info)["Guess Close"]
        actual_close = pandas.read_csv(all_info)["Close"]
        total_num = len(actual_close.values)
        all_mse = []
        # loop it
        for index in range(len(predict_close)):
            mse = (1/total_num) * sum(predict_close[index] - actual_close[index])**2
            all_mse.append(mse)
        mse_data_frame = pandas.DataFrame({"Mean Squared Error": all_mse})
        mse_data_frame.to_csv("Programmed/MEAN_SQUARE_ERROR.csv")


    def cal_rmse(self, all_info):
        """
        evaluate model:
            root mean squared error (rmse)
            = (MSE)**(1/2)
        """
        all_data = pandas.read_csv(all_info)["Mean Squared Error"].values
        all_rmse = []
        for data in all_data:
            rmse = data**(1/2)
            all_rmse.append(rmse)
        rmse_data_frame = pandas.DataFrame({"Root Mean Squared Error": all_rmse})
        rmse_data_frame.to_csv("Programmed/ROOT_MEAN_SQUARED_ERROR.csv")


    def cal_r_square(self, all_info, group):
        """
        evaluate model:
            R-squared to find accuracy of model's prediction for attributes
            R^2 = 1 - (Sum of Squared Errors(SSE)/(Total Sum of Squares(SST)))
        Note: R^2 is going to fall betwen 0 or 1
        """
        all_data = pandas.read_csv(all_info)
        all_group = ["No Activity", "Activity"]
        all_value = [no_activity, activity]
        actual_avg = math.calculate_mean(all_data, group)
        total_sse = 0
        total_tss = 0
        for index in range(len(all_data[group])):
            sse = math.calculate_sse(all_group, index, group)
            tss = math.calculate_tss(all_data, index, actual_avg)
            total_sse += sse
            total_tss += tss
        r_squt = 1 - (total_sse/total_tss)
        if r_squt == 0: # horizontal line that does not show upward or downward trend
            no_activity += 1
        if r_squt == 1: # can predict upward or downward trend
            activity += 1
        # graph it here!!

import matplotlib.pyplot
import math_help_func as math
import pandas
import plot_transfer as image
import matplotlib
import tensorflow

class Conclusion:
    """
    This class is a blueprint of the object resposible for finding forestcasting results
    and the calculate the daily return, price movement and price direction.

    Attributes:
        csv_file = the content in the given csv file
        today_price = the values in the attribute "Close" (Today's Close Price)
        yesterday_price = the values in the attribute "Adj Close" (Yesterday's Close Price)
        high_stock = the values in the attribute "High" (high stocks during that day)
        low_stock = the values in the attribute "Low" (low stocks during that day) 

    """
    def __init__(self, filepath:str):
        self._csv_file = pandas.read_csv(filepath)
        self._today_price = self._csv_file["Close"].values
        self._yesterday_price = self._csv_file["Adj Close"].values
        self._high_stock = self._csv_file["High"].values
        self._low_stock = self._csv_file["Low"].values


    """
    This method is used to calculate the daily return per record.
    Args:
        Nothing
    Returns:
        returns the value that represents the daily return for given record
    Raises:
        Nothing
    Implemented:
        If written in another file:
            import Conclusion.py as result

            result.Conclusion(filepath).calculate_daily_return()

        If written in the same file the method was invokded:
            calculate_daily_return()
    """
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


    """
    This method is used for calculating price change per record
    Args:
        Nothing
    Returns:
        returns the value that represents the price change for given record
    Raises:
        Nothing
    Implemented:
        If written in another file:
            import Conclusion.py as result

            result.Conclusion(filepath).price_change()

        If written in the same file the method was invokded:
            price_change()
    """
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
        file_path = "Stock-Sentiment-Analyzer/Programmed/Calculations/ALL_PRICE_RANGES.csv"
        for stock in self._high_stock:
            all_ranges.append(stock - self._low_stock[low_stock_counter])
            low_stock_counter += 1
        price_range_data_frame = pandas.DataFrame({"Price Range": all_ranges})
        price_range_data_frame.to_csv(file_path, index = False)
        avg = math.avg_daily_range(file_path)
        std = math.standard_dev(avg, all_ranges)
        #print(f"std {std}")
        # print(f"Avg price range: {avg}")
        # for a_range in all_ranges:
        #     if math.wide_threshold(std, avg) < a_range: # wide compared to the average price range
        #         print("large than usual flucation")
        #         # print(f"wide: threshold {math.wide_threshold(std, avg)} and the samle ({a_range})")
        #     if math.narrow_threshold(std, avg) > a_range: # narrow compared to the average price range
        #         # print(f"narrow threshold {math.narrow_threshold(std,avg)} and the samle ({a_range})")
        #         print("smaller than usual flucation")
        #@TODO: create a matplotlib graph here
        fig = matplotlib.pyplot.figure(figsize=(12,6))
        matplotlib.pyplot.vlines(self._csv_file["Date"].values, self._low_stock, self._high_stock, label="Price Range")
        matplotlib.pyplot.axhline(math.wide_threshold(std, avg))
        matplotlib.pyplot.axhline(math.narrow_threshold(std, avg))
        matplotlib.pyplot.axhline(avg)
        matplotlib.pyplot.xlabel("Date")
        matplotlib.pyplot.ylabel("Price Range")
        matplotlib.pyplot.grid(True)
        matplotlib.pyplot.xticks(rotation = 45)
        
        writer = tensorflow.summary.create_file_writer("Programmed/logs/mode_price_change") # Create a writer 
        with writer.as_default():
            tensorflow.summary.image("matplotlib_plot", image.plot_to_image(fig), step=0)

    
    """
    The method is used to calculate price movement per record
    Args:
        Nothing
    Returns:
        Nothing
    Raises:
        Nothing
    Implemented:
        If written in another file:
            import Conclusion.py as result

            result.Conclusion(filepath).price_move()

        If written in the same file the method was invokded:
            price_move()
    """
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
        #@TODO: create a matplotlib graph here

    """
    The method is used to calculate the predicted price change
    Args:
        all_info = contains teh entire dataset that is given
    Returns:
        Nothing
    Raises:
        Nothing
    Implemented:
        If written in another file:
            import Conclusion.py as result

            result.Conclusion(filepath).price_cal_with_predict(all_info)

        If written in the same file the method was invokded:
            price_cal_with_predict(all_info)
    """
    def price_cal_with_predict(self, all_info:str):
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
        #@TODO: create a matplotlib graph here
        matplotlib.pyplot.plot(pandas.read_csv(all_info)["Date"].values,all_price_change)
        fig = matplotlib.pyplot.gcf()
        writer = tensorflow.summary.create_file_writer("Programmed/logs") # Create a writer 
        with writer.as_default():
            tensorflow.summary.image("matplotlib_plot_04", image.plot_to_image(fig), step=0)

        price_change_dataframe.to_csv("Stock-Sentiment-Analyzer/Programmed/Predicted Data/PREDICTED_PRICE_CHANGE.csv", index = False)
        
    """
    This method is used to calcualte the predicted price range.
    Args:
        all_info = contains teh entire dataset that is given
    Returns:
        Nothing
    Raises:
        Nothing
    Implemented:
        If written in another file:
            import Conclusion.py as result

            result.Conclusion(filepath).range_cal_with_predict(all_info)

        If written in the same file the method was invokded:
            range_cal_with_predict(all_info)
    """
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
        predict_range_dataframe.to_csv("Stock-Sentiment-Analyzer/Programmed/Predicted Data/PREDICTED_PRICE_RANGE.csv")
        matplotlib.pyplot.plot(pandas.read_csv(all_info)["Date"].values,all_predict_range)
        fig = matplotlib.pyplot.gcf()
        writer = tensorflow.summary.create_file_writer("Programmed/logs") # Create a writer 
        with writer.as_default():
            tensorflow.summary.image("matplotlib_plot_05", image.plot_to_image(fig), step=0)

    """
    This method is used to calculate the predicted daily return.
    Args:
        all_info = contains teh entire dataset that is given
    Returns:
        Nothing
    Raises:
        Nothing
    Implemented:
        If written in another file:
            import Conclusion.py as result

            result.Conclusion(filepath).daily_return_with_predict(all_info)

        If written in the same file the method was invokded:
            daily_return_with_predict(all_info)
    """
    def daily_return_with_predict(self, all_info):
        """
        daily return:
            Daily return = (Today's predicted close - yesterday's predicted close) / Yesterday's predicted close
        """
        today_predict = pandas.read_csv(all_info)["Guess Close"].values
        yesterday_predict = pandas.read_csv(all_info)["Adj Close"].values
        all_predict_returns = []
        for index in range(len(today_predict)):
            daily_return = (today_predict[index] - yesterday_predict[index]) / yesterday_predict[index]
            all_predict_returns.append(daily_return)
        predict_return_dataframe = pandas.DataFrame({"Predicted daily return": all_predict_returns})
        matplotlib.pyplot.plot(pandas.read_csv(all_info)["Date"].values,all_predict_returns)
        fig = matplotlib.pyplot.gcf()
        writer = tensorflow.summary.create_file_writer("Programmed/logs") # Create a writer 
        with writer.as_default():
            tensorflow.summary.image("matplotlib_plot_06", image.plot_to_image(fig), step=0)
        predict_return_dataframe.to_csv("Stock-Sentiment-Analyzer/Programmed/Predicted Data/PREDICTED_DAILY_RETURN.csv", index = False)


    """
    This method is used to calculate the predicted daily price movement.
    Args:
        all_info = contains teh entire dataset that is given
    Returns:
        Nothing
    Raises:
        Nothing
    Implemented:
        If written in another file:
            import Conclusion.py as result

            result.Conclusion(filepath).daily_mov_with_predict(all_info)

        If written in the same file the method was invokded:
            daily_mov_with_predict(all_info)
    """
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
        predict_changes = pandas.read_csv(all_info)["predict price change"].values
        move_val = []
        for change in predict_changes.astype(float):
            if change > 0:
                move_val.append(1)
            elif change < 0:
                move_val.append(-1)
            else:
                move_val.append(0)
        matplotlib.pyplot.plot(range(len(predict_changes)), move_val)
        matplotlib.pyplot.yticks([-1, 0, 1], ["Price Down", "No Change", "Price Up"])
        fig = matplotlib.pyplot.gcf() # Get the current figure
        writer = tensorflow.summary.create_file_writer("Programmed/logs") # Create a writer 
        with writer.as_default():
            tensorflow.summary.image("matplotlib_plot_07", image.plot_to_image(fig), step=0)

    """
    This method is used to calculate the mean squarred error (MSE)
    Args:
        all_info = contains teh entire dataset that is given
    Returns:
        Nothing
    Raises:
        Nothing
    Implemented:
        If written in another file:
            import Conclusion.py as result

            result.Conclusion(filepath).cal_mse(all_info)

        If written in the same file the method was invokded:
            cal_mse(all_info)
    """
    def cal_mse(self, all_info):
        """
        evaluate model:
            mean squared error(MSE)
            = (1/n) * summation(predicted close - actual close)^2
        """
        predict_close = pandas.read_csv(all_info)["Guess Close"]
        actual_close = pandas.read_csv(all_info)["Close"]
        total_num = len(predict_close)
        summation = 0
        # loop it
        for index in range(total_num):
            summation += (predict_close[index] - actual_close[index])**2
        mse = (1/total_num) *summation
        mse_data_frame = pandas.DataFrame({"Mean Squared Error": [mse]})
        matplotlib.pyplot.plot(range(total_num), actual_close, label="Actual Close", color="blue")
        matplotlib.pyplot.plot(range(total_num), predict_close, label="Predicted Close", color="red", linestyle="--") # dashed line

        matplotlib.pyplot.xlabel("Time/Index")  # X-axis label
        matplotlib.pyplot.ylabel("Closing Price")  # Y-axis label
        matplotlib.pyplot.title("Actual vs. Predicted Closing Prices")  # Plot title

        fig = matplotlib.pyplot.gcf() # Get the current figure

        writer = tensorflow.summary.create_file_writer("Programmed/logs") # Create a writer 
        with writer.as_default():
            tensorflow.summary.image("matplotlib_plot_08", image.plot_to_image(fig), step=0)
        mse_data_frame.to_csv("Stock-Sentiment-Analyzer/Programmed/Calculations/MEAN_SQUARE_ERROR.csv")


    """
    This method is used to calculate the root mean squared error.
    Args:
        all_info = contains teh entire dataset that is given
    Returns:
        Nothing
    Raises:
        Nothing
    Implemented:
        If written in another file:
            import Conclusion.py as result

            result.Conclusion(filepath).cal_rmse(all_info)

        If written in the same file the method was invokded:
            cal_rmse(all_info)
    """
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
        rmse_data_frame.to_csv("Stock-Sentiment-Analyzer/Programmed/Calculations/ROOT_MEAN_SQUARED_ERROR.csv")


    """
    This method is use to calculate the r squared value.
    Args:
        all_info = contains teh entire dataset that is given
        group = contains the group the user would like to see
    Returns:
        Nothing
    Raises:
        Nothing
    Implemented:
        If written in another file:
            import Conclusion.py as result

            result.Conclusion(filepath).cal_r_square(self, all_info, group)

        If written in the same file the method was invokded:
            cal_r_square(self, all_info, group)
    """
    def cal_r_square(self, all_info, group):
        """
        evaluate model:
            R-squared to find accuracy of model's prediction for attributes
            R^2 = 1 - (Sum of Squared Errors(SSE)/(Total Sum of Squares(SST)))
        Note: R^2 is going to fall betwen 0 or 1
        """
        all_data = pandas.read_csv(all_info)
        all_group = ["No Activity", "Activity"]
        no_activity = 0
        activity = 0
        all_value = [no_activity, activity]
        actual_avg = math.calculate_mean(all_data, group)
        total_sse = 0
        total_tss = 0
        for index in range(len(all_data[group])):
            sse = math.calculate_sse(all_data, index, group)
            tss = math.calculate_tss(all_data, index, actual_avg, group)
            total_sse += sse
            total_tss += tss
        r_squt = 1 - (total_sse/total_tss)
        if r_squt == 0: # horizontal line that does not show upward or downward trend
            no_activity += 1
        if r_squt == 1: # can predict upward or downward trend
            activity += 1
        # graph it here!!

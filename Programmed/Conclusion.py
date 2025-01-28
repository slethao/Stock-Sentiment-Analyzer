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

        (200 - 140)/100 = 0.6 (for the ratio) 
                        = 60 % (for the percentage *multipy by 100*)
        """
        daily_return = (self._today_price - self._yesterday_price) / self._yesterday_price
        return daily_return

    def price_change(self): # for stock volume
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
                print(f"wide: threshold {math.wide_threshold(std, avg)} and the samle ({a_range})")
            if math.narrow_threshold(std, avg) > a_range: # narrow compared to the average price range
                print(f"narrow threshold {math.narrow_threshold(std,avg)} and the samle ({a_range})")
                print("smaller than usual flucation")

    def price_move(self): # for price change
        """
        price movement used in volume
        """
        pass
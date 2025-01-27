import math_help_func as math
import pandas

class Conclusion:
    """
    Goal of this class: Find the daily returns and the price change
    Daily Return = (Today's Closing Price - Yesterday's Closing Price) / Yesterday's Closing 
    Today's Price = Close (regular trading session)
    Yesterday's Price = Adj Close (stock splits and dividends and help with long term analysis (price change behaviro))
    Example:
        yesterday price  = 140
        today price = 200

        (200 - 140)/100 = 0.6 (for the ratio) = 60 % (for the percentage *multipy by 100*)
    
    Price Change:    
    High and Low is the price at whcih the stokc is traded
    price movement used in volume
    """

    def __init__(self, filepath):
        self._csv_file = pandas.read_csv(filepath)
        self._today_price = self._csv_file["Close"].values
        self._yesterday_price = self._csv_file["Adj Close"].values
        self._high_stock = self._csv_file["High"].values
        self._low_stock = self._csv_file["Low"].values

    def calculate_daily_return(self): # daily return
        """
        Example:
        yesterday price  = 140
        today price = 200

        (200 - 140)/100 = 0.6 (for the ratio) 
                        = 60 % (for the percentage *multipy by 100*)
        """
        daily_return = (self._today_price - self._yesterday_price) / self._yesterday_price
        return daily_return

    def price_move(self): # for stock volume
        """
        High vol = strong buying or selling pressure
        Low vol = thin liquidity and increased risk
        1) put all ranges into a .csv and calculate the average daily range (done)
        2) calculate standard deviation (done)
        3) set the threshold (wide or narrow range) (pending)
        4) current day's range for th
        """
        low_stock_counter = 0
        all_ranges = []
        for stock in self._high_stock:
            all_ranges.append(stock - self._low_stock[low_stock_counter])
            low_stock_counter += 1
        price_range_data_frame = pandas.DataFrame({"Price Range": all_ranges})
        price_range_data_frame.to_csv("Programmed/ALL_PRICE_RANGES.csv", index = False)
        avg = math.avg_daily_range("Programmed/ALL_PRICE_RANGES.csv")
        std = math.standard_dev(avg, all_ranges)
        print(f"std {std}")
        # print(f"Avg price range: {avg}")

        # if : # wide compared to the average price range
        #     pass
        # else: # narrow compared to the average price range
        #     pass

    def price_behave(self): # for price change
        pass
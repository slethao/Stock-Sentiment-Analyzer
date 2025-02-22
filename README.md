# <span style="color:#FFB4A2"> **Stock Sentiment Analyzer**
### <span style="color:#E5989B"> *Predicting NVIDIA Stock Trends with Machine Learning*
##### This program uses machine learning to analyze NVIDIA stock data and predict daily returns and price changes. It incorporates sentiment analysis based on news articles to provide a comprehensive outlook for the current and future year. The inforamtion that is loaded into the program is a csv file that contains <span style="color:orange"><u>seven attributes</u></span>:
| Attributes | Description |
| --- | --- |
| <span style="color:orange">*Date*</span> | The date of the stock data |
| <span style="color:orange">*Open*</span> | The opening price of NVIDIA's stock on the given date |
| <span style="color:orange">*High*</span> | The highest price NVIDIA's stock reached during the trading day |
|<span style="color:orange">*Low*</span>| The lowest price NVIDIA's stock reached during the trading day. |
|<span style="color:orange">*Close*</span>| The closing price of NVIDIA's stock on the given date |
|<span style="color:orange">*Adj Close*</span>| The adjusted closing price, reflecting dividends splits, and other adjustments. |
|<span style="color:orange">*Volume*</span>|The number of shares traded during the trading day.|
## <span style="color:#FFB4A2"> Data Preproccessing
The data is first filtered by getting rid of outliers and filling up missing values with 
KNN Imputer. Afterward, the data is put into an Isolation Forest in which the data is split 
into groups based on its attributes and then idenify anomaly results and outliers within each 
group made, and afterward each results made from each group in put into a csv. file.</br>
<br><span style="color:orange">*Profile Who Provided The Dataset:*
[Link Here](https://www.kaggle.com/muhammaddawood42)</span></br>
## <span style="color:#FFB4A2"> Data Modeling
The filtered data is then loaded into a TensorFlow model which is then built to fit standard scaler and the data is compiled to be trained to learn how to evualte data. Then it creates a prediction on the learned data to comput up with postive or negative results based on how well it was trained.
| Type of Results | Description |
| --- | --- |
| <span style="color:orange"> Negative value | Prediction made is lower value than the baseline value|
| <span style="color:orange"> Positive Value | Prediction made is higher value than the baseline value |
The predicted data is then loaded and stored into the csv file along with the previous data that was loaded into the TensorFlow model.
## <span style="color:#FFB4A2"> Sentiment Analysis
The predicted results from the TensorFlow model and the 
results from the Isolation Forest Model. The data is used 
to calculate the current daily returns, price change, price 
movement. In addition, the data is also used to calcualted 
the predicted price change, pprice range, daily return, price 
movement. Finally, the mean squared error (MSE) to take the predicted close and the actual close price to find how close the predicted close price is to the actual close price. The root mean squared error (RMSE) 
to measure the average difference between the predicted closing and actual price. The R-Squared value to find the predicted today's close 
price and yesterday's price. The data will ran through a 
polarity class which will conclude the results: negative, 
postive or netural.
## <span style="color:#FFB4A2"> Display Results
The results that are found will be stored into a database and displayed on TensorBoard to show graphical data and how the data will be compared with today's current events.
<blockquote style="border-left: 5px solid #FFB4A2; padding-left: 10px;">
    <h2><span style="color:#FFB4A2">Installation</span></h2>
    <p><h5><span style="color:#B5838D">To install the Stock Sentiment Analyzer, follow these steps:</span></h5></p>
    <ol>
        <li><h4><span style="color:#E5989B">Clone the repository</span><br><code>git clone https://github.com/slethao/Stock-Sentiment-Analyzer.git</h4></code></li>
        <li><h4><span style="color:#E5989B">Create and activate a virtual environment</span><br><code>python -m venv venv</h4></code></li>
        <li><h4><span style="color:#E5989B">Install the required dependencies</span><br>For MacOS:
        <ol><code>pip3 install tensorflow</code><br>
        <code>pip3 isntall pandas</code><br>
        <code>pip3 install numpy</code><br>
        <code>pip3 install csv</code><br>
        <code>pip3 install request</code><br>
        <code>pip3 install scikit-learn</code><br>
        </ol>
        For Windows:<ol><code>pip install tensorflow</code><br>
        <code>pip install pandas<br></code>
        <code>pip install numpy</code><br>
        <code>pip install csv<br></code>
        <code>pip install request<br></code>
        <code>pip install scikit-learn</code>
        Kaggle API Setup:
        <li><h4><span style="color:#E5989B">Create your own Kaggle API token by going to your Kaggle account settings.</span><br>
        1.  Download the <code>kaggle.json</code> file. <br>
        2.  Place the <code>kaggle.json</code> file in the <code>~/.kaggle/</code> directory (create the directory if it doesn't exist).
        OR<br>
        Set the <code>KAGGLE_USERNAME</code> and <code>KAGGLE_KEY</code> environment variables.
</h4></li>
</blockquote>

## <span style="color:#FFB4A2">Usage
To run the entire:
```python Main_Program.py ```

<blockquote style="border-left: 5px solid #FFB4A2; padding-left: 10px;">
    <h2><span style="color:#FFB4A2">Important Formulas Used</span></h2>
    <ol>
        <span style="color:#E5989B">Daily Return</span><br>The percentage change in the value of an investment over a single day. It's calculated using the daily closing prices of the investment.<br>
        <span style="color:#E5989B">Volume-Price Trend (VPT)</span><br>A formula used to help understand the strength of a price trend by considering the volume associated with those price movements by seeing the relationship between price and volume.<br>
        <span style="color:#E5989B">On-Balance Volume (OBV)
        </span><br>A formula that is used to predict price changes base on volume trends.<br>
        <span style="color:#E5989B">Sum of Squared Errors(SSE)</span><br>Used to create a model to predict today's and yesterday's closing price values.<br>
        <span style="color:#E5989B">Total Sum of Squares(TSS)</span><br>This formula measures the total amount of variability in a dataset that shows the mean of all the data points in the stocks.<br>
    </ol>
</blockquote>

## <span style="color:#FFB4A2">Formulas Used

$$\text{Daily Return} = \frac{\text{Today's Closing Price} - \text{Yesterday's Closing Price}}{\text{Yesterday's Closing Price}}$$

$$\text{Volume-Price Trend (VPT)}=\text{VPT Yesterday} + \text{(Volume Today} \times \frac{\text{(Today's Close - Yesterday's Close)}}{\text{Yesterday's Close}} $$

$$\text{Three Formulas for On-Balance Volume (OBV)}:\\\text{Today's Close > Yesterday's Close}:
\quad{\text{Today = Yesterday + Volume}}
\\\text{Today's Close < Yesterday's Close}:
\quad{\text{Today = Yesterday - Volume}}
\\\text{Today's Close = Yesterday's Close}:
\quad{\text{Today = Yesterday}}$$

$$\text{Mean Squared Error (MSE)}=\frac{\text{1}}{\text{n}}\times\sum{(\text{Actual}-\text{Predicted})^2}$$

$$\text{Root Mean Squared Error (RMSE)}=\sqrt{MSE}$$


## <span style="color:#FFB4A2">License

This project is licensed under the GNU General Public License v3.0 - see the [LICENSE](LICENSE) file for details.

<blockquote style="border-left: 5px solid #FFB4A2; padding-left: 10px;">
<h2><span style="color:#FFB4A2">Resources Used</span></h2>
<p>TensorFlow Documentation: <a href="https://www.tensorflow.org">the link</a></p>
<p>Linked to Dataset: <a href="https://www.kaggle.com/datasets/muhammaddawood42/nvidia-stock-data/data">the link</a></p>
</blockquote>

<blockquote style="border-left: 5px solid #FFB4A2; padding-left: 10px;">
<h2><span style="color:#FFB4A2">Contact Information</span></h2> 
<p>thaosle4@gmail.com</p>
</blockquote>

<blockquote style="border-left: 5px solid #FFB4A2; padding-left: 10px;">
<h2><span style="color:#FFB4A2">⭐️ Special Note For Viewers</span></h2>
<p>If you find this project useful or interesting, please consider starring the repository on GitHub. Your support is greatly appreciated!</p>
</blockquote>


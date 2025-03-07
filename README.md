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
The filtered data is then loaded into a TensorFlow model which is then built to fit standard scaler and the data is compiled to be trained to learn how to evualte data. Then it creates a prediction on the learned data to comput up with postive or negative results based on how well it was trained. The predicted data is then loaded and stored into the csv file along with the previous data that was loaded into the TensorFlow model.
| Type of Results | Description |
| --- | --- |
| <span style="color:orange"> Negative value | Prediction made is lower value than the baseline value|
| <span style="color:orange"> Positive Value | Prediction made is higher value than the baseline value |
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
The results that are found will be stored into a database and displayed on TensorBoard to show graphical data and how the data will be compared with today's current events. The entire program will be executed in the <code>Main_Program.py</code>. After the execution of the program type <code> tensorboard --logdir Programmed/logs/</code> in the terminal of the IDE that is currently being used to execute the TensorBoard to display results.
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
        <code>pip3 install re</code><br>
        <code>pip3 install spacy</code><br>
        <code>pip3 install os</code><br>
        <code>pip3 install feedparser</code><br>
        <code>pip3 install matplotlib</code><br>
        <code>pip3 install sqlite3</code><br>
        </ol>
        For Windows:<ol><code>pip install tensorflow</code><br>
        <code>pip install tensorflow</code><br>
        <code>pip isntall pandas</code><br>
        <code>pip install numpy</code><br>
        <code>pip install csv</code><br>
        <code>pip install request</code><br>
        <code>pip install scikit-learn</code><br>
        <code>pip install re</code><br>
        <code>pip install spacy</code><br>
        <code>pip install os</code><br>
        <code>pip install feedparser</code><br>
        <code>pip install matplotlib</code><br>
        <code>pip install sqlite3</code><br>
        <li><h4><span style="color:#E5989B">Kaggle API Setup: Create a Kaggle API token through your Kaggle Account</span></li></h4>
        <ol>
        1. Download the <code>kaggle.json</code> file.<br>
        2. Place the <code>kaggle.json</code> file in the <code>~/.kaggle/</code> directory (create the directory if it doesn't exist)
        OR set the <code>KAGGLE_USERNAME</code> and <code>KAGGLE_KEY</code> environment variables.<br>
        <p><span style="color:#E5989B">Kaggle API Token Documentation: </span><a href="https://github.com/Kaggle/kaggle-api">the link</a></p>
        </ol>
    </ol>
</h4></li>
</blockquote>

## <span style="color:#FFB4A2">Usage
To run the entire:
```python Programmed/Main_Program.py ```

## <span style="color:#FFB4A2">License

This project is licensed under the GNU General Public License v3.0 - see the [LICENSE](LICENSE) file for details.

<blockquote style="border-left: 5px solid #FFB4A2; padding-left: 10px;">
<h2><span style="color:#FFB4A2">Resources Used</span></h2>
<p>TensorFlow Documentation: <a href="https://www.tensorflow.org">the link</a></p>
<p>Linked to Dataset: <a href="https://www.kaggle.com/datasets/muhammaddawood42/nvidia-stock-data/data">the link</a></p>
</blockquote>

<blockquote style="border-left: 5px solid #FFB4A2; padding-left: 10px;">
<h2><span style="color:#FFB4A2">Contact Information</span></h2> 
<p>Email: <a href="mailto:thaosle4@gmail.com">the link</a></p>
<p>LinkIn (Preferred): <a href="http://linkedin.com/in/sommer-le-474100347">the link</a></p>

</blockquote>

<blockquote style="border-left: 5px solid #FFB4A2; padding-left: 10px;">
<h2><span style="color:#FFB4A2">⭐️ Special Note For Viewers</span></h2>
<p>If you find this project useful or interesting, please consider starring the repository on GitHub. Your support is greatly appreciated!</p>
</blockquote>


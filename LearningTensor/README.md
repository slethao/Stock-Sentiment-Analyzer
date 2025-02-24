# <span style="color:#F3D2AC">Getting Started with TensorFlow</span>
*Crash course on how to learn TensorFlow.*
## <span style="color:#F3B8B1"> My Thoughts on TensorFlow
*My initial approach for the preparation to create the Stock Analyzer with TensorFlow was to follow the official documentation.  Unfortunately, I encountered numerous execution problems due to outdated code examples.  This necessitated extensive research on GitHub and within the TensorFlow community forums to find solutions* 

<blockquote style="border-left: 5px solid #CECBD6; padding-left: 10px;">
    <h2><span style="color:#CECBD6">Example Issuses That Arose During My Experience</span></h2>
    <ol>
        <span style="color:#CECBD6"><u>One Example of This</u></span><br>
        <code>
        mnist = tf.keras.datasets.mnist
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        x_train, x_test = x_train / 255.0, x_test / 255.0
        <>
        model = tf.keras.models.Sequential([
                tf.keras.layers.Flatten(input_shape=(28, 28)),
                tf.keras.layers.Dense(128, activation='relu'),
                tf.keras.layers.Dropout(0.2),
                tf.keras.layers.Dense(10)
                ])<br></code>
        Error Made:
        <b><br>UserWarning: Do not pass an `input_shape`/`input_dim` argument to a layer. When using Sequential models, prefer using an `Input(shape)` object as the first layer in the model instead.
  super().__init__(**kwargs)</b><br>Linked to Solution Found: [Link here](www.google.com)</br></br>
        Solution found:
        <br><code>data = tensorflow.keras.datasets.mnist
        (x_train, y_train), (x_test, y_test) = data.load_data()
        x_train, x_test = x_train / 255.0, x_test / 255.0</code>
        <code>model = tf_keras.src.engine.sequential.Sequential(
                                    [tensorflow.keras.layers.Flatten(input_shape = (28,28)),
                                     tensorflow.keras.layers.Dense(128, activation = 'relu'),
                                     tensorflow.keras.layers.Dropout(0.2),
                                     tensorflow.keras.layers.Dense(10)] # this is the final layer)</code>
    </ol>
</blockquote>

## <span style="color:#7DC8CA">Section's Pupose </span> 
This part of the reposity is my testing and experimenting with the documation examples and finding out what works, so you don't have to!
| Lessons | Description|
| --- | --- |
| <span style="color:#7DC8CA">lesson_one.py</span> | An overview of how TensorFlow models work and the purpose behind it. |
| <span style="color:#7DC8CA">lesson_two.py</span> | A quick lesson to know how to load and prcoess data from flat files such as .txt, .csv, and more |
| <span style="color:#7DC8CA">lesson_three.py</span> | A lesson on how to do basic image and text classifcation in TensorFlow. |
| <span style="color:#7DC8CA">lesson_four.py</span> | This will cover the concept of regression that is a reoccuring theme in tensor flow that is use for prediction of values to evalute relationships between them. In addition to different sorts of regressions may come accross. |
| <span style="color:#7DC8CA">lesson_five.py</span> | This lesson will take baout the concept of overfit and underfit and how these conepts are seen in the TensorFlow model. In addition, to learning how to use hpyerparameters within your TensorFlow model to find a happy-median between overfit and underfit. |
| <span style="color:#7DC8CA">lesson_six.py</span> | This lesson will cover layers taht can be used in the neurtal netowrks that can be used in the TensroFlow model to reduce loss of data and validation of data.|
| <span style="color:#7DC8CA">lesson_seven.py</span> | A breif explaination on how to create a customer traingin for your model to customize the efficeny of for your TensorFlow model. |
| <span style="color:#7DC8CA">lesson_eight.py</span> | A breif lesson to learn how to use optimizers to  |
| <span style="color:#7DC8CA">lesson_nine.py</span> | A crashcourse lesson on how to use Tensorboard to display results from created models. |
<blockquote style="border-left: 5px solid #7DC8CA; padding-left: 10px;">

<h2><span style="color:#CAE9E0">★ Special Note For Viewers</span></h2>
<p>If you find this project useful or interesting, please consider starring the repository on GitHub. In additions, if you have any suggestions on new updates that were created in tensorflow, feel free to make a contribute to this project. Your support is greatly appreciated! 💚</p>
</blockquote>


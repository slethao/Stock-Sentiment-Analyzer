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
| <span style="color:#7DC8CA">lesson_one.py</span> | This file will allow the programmer to see the steps of how the machine learning model will be built and executed in a non-object oriented way. |
| <span style="color:#7DC8CA">lesson_two.py</span> | This file shows is an extension of <code>lesson_one.py</code> that will show the programmer how to use an optimizer in their machine learning model along with using the decorator <code>@tensorflow.function</code> |
| <span style="color:#7DC8CA">lesson_three.py</span> | This file will show you how to build and execute and machine learning model through a class and includes how to create a personalized training method for the model. |
| <span style="color:#7DC8CA">lesson_four.py</span> | something |
| <span style="color:#7DC8CA">lesson_five.py</span> | something |
| <span style="color:#7DC8CA">lesson_six.py</span> | something |
| <span style="color:#7DC8CA">lesson_seven.py</span> | something |
| <span style="color:#7DC8CA">lesson_eight.py</span> | something |
| <span style="color:#7DC8CA">lesson_nine.py</span> | something |
<blockquote style="border-left: 5px solid #7DC8CA; padding-left: 10px;">

<h2><span style="color:#CAE9E0">★ Special Note For Viewers</span></h2>
<p>If you find this project useful or interesting, please consider starring the repository on GitHub. In additions, if you have any suggestions on new updates that were created in tensorflow, feel free to make a contribute to this project. Your support is greatly appreciated! 💚</p>
</blockquote>


# Keras

[Category:AI and Machine Learning](Category:AI_and_Machine_Learning.md)
"Keras is a high-level neural networks API, written in Python and capable of running on top of TensorFlow, CNTK, or Theano."<ref>https://keras.io/</ref>

If you are porting a Keras program to one of our clusters, you should follow [our tutorial on the subject](Tutoriel_Apprentissage_machine.md).

## Installing

#Install [TensorFlow](TensorFlow.md), CNTK, or Theano in a Python [virtual environment](Python#Creating_and_using_a_virtual_environment.md).
#Activate the Python virtual environment (named <tt>$HOME/tensorflow</tt> in our example).
#:{{Command2|source $HOME/tensorflow/bin/activate}}
#Install Keras in your virtual environment.
#:{{Command2
|prompt=(tensorflow)_[name@server ~]$
|pip install keras}}

### R package

This section details how to install Keras for R and use TensorFlow as the backend.

#Install TensorFlow for R by following [these instructions](Tensorflow#R_package.md).
#Follow the instructions from the parent section.
#Load the required modules. 
#:{{Command2|module load gcc/7.3.0 r/3.5.2}}
1. Launch R.
#:{{Command2|R}}
#In R, install the Keras package with <code>devtools</code>. 
#:<syntaxhighlight lang='r'>
devtools::install_github('rstudio/keras')
</syntaxhighlight>


You are then good to go. Do not call <code>install_keras()</code> in R, as Keras and TensorFlow have already been installed in your virtual environment with <code>pip</code>. To use the Keras package installed in your virtual environment, enter the following commands in R after the environment has been activated.
<syntaxhighlight lang='r'>
library(keras)
use_virtualenv(Sys.getenv('VIRTUAL_ENV'))
</syntaxhighlight>

## References
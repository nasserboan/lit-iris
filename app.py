from tensorflow_core.python.keras.api._v2.keras import Sequential

from tensorflow_core.python.keras.api._v2.keras.layers import Dense, Dropout
from tensorflow_core.python.keras.api._v2.keras.utils import to_categorical, plot_model

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix

import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt


st.title('Iris MLP Visualizer')
st.write('Use the sidebar to change hyperparameters and train the MLP.')

def _data(test_size):

    ## getting the data
    data = load_iris()

    ## defining x and y
    x = data.data
    y = data.target

    ## train test split
    x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=test_size,random_state=42)

    ## transforming y to categorical data
    y_train_cat = to_categorical(y_train,num_classes=3)
    y_test_cat = to_categorical(y_test,num_classes=3)

    return x_train,x_test,y_train,y_test,y_train_cat,y_test_cat


def _define_model(input_size,act_input,dense1,dense2,act_dense,drop_out):

    ## generating a model
    model = Sequential()

    ## input layer

    model.add(Dense(input_size,input_dim=4,activation=act_input))
    model.add(Dense(dense1,activation=act_dense))
    model.add(Dense(dense2,activation=act_dense))
    model.add(Dropout(drop_out))

    ## output layer
    model.add(Dense(3,activation='softmax'))

    ## compiling the model
    model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
    
    return model


def train(model,x_train,y_train_cat,NUM_EPOCHS,x_test,y_test_cat):

    history = model.fit(x_train,y_train_cat,epochs = NUM_EPOCHS, validation_data=(x_test,y_test_cat))

    return history,model


def model_plot(NUM_EPOCHS,test_size,drop_out,input_size,dense1,dense2,act_input,act_dense,model,x_test,y_test,history):

    fig = plt.figure(figsize=(17,9))
    fig.suptitle(f'Summary\nEpochs : {NUM_EPOCHS}   Test Size : {test_size}   Dropout : {drop_out}\n Layers : {input_size,dense1,dense2,3}\n Activation : {act_input,act_dense,act_dense,"softmax"}',fontsize=10)
    grid = plt.GridSpec(3,2,hspace=0.2)

    ax0 = plt.subplot(grid[:,0]) ## confusion matrix
    ax1 = plt.subplot(grid[0,1]) ## text
    ax2 = plt.subplot(grid[1,1]) ## loss
    ax3 = plt.subplot(grid[2,1]) ## metric

    ## confusion matrix
    sns.heatmap(confusion_matrix(y_test,model.predict_classes(x_test)),annot=True,cmap=plt.cm.Purples,cbar=False,ax=ax0)
    ax0.set_xlabel('Predicted Label',size=10,labelpad=15)
    ax0.set_ylabel('True Label',size=10,labelpad=15)
    ax0.set_xticklabels(['setosa', 'versicolor', 'virginica'])
    ax0.set_yticklabels(['setosa', 'versicolor', 'virginica'],fontdict = {'verticalalignment':'center'})

    ## loss
    sns.lineplot(x = range(1,NUM_EPOCHS+1),y = history.history['val_loss'], color = 'tab:orange', ax = ax2, label='Test LOSS')
    sns.lineplot(x = range(1,NUM_EPOCHS+1),y = history.history['loss'], color = 'tab:blue', ax = ax2, label='Train LOSS')
    ax2.set_xticklabels([])
    ax2.spines['right'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax2.legend(fontsize='small')
    ax2.spines['left'].set_position(('outward', 10))
    ax2.spines['bottom'].set_position(('outward', 10))


    ## ACC
    sns.lineplot(x = range(1,NUM_EPOCHS+1),y = history.history['val_accuracy'], color = 'tab:orange', ax = ax3, label='Test ACCURACY')
    sns.lineplot(x = range(1,NUM_EPOCHS+1),y = history.history['accuracy'], color = 'tab:blue', ax = ax3, label='Train ACCURACY')
    ax3.spines['right'].set_visible(False)
    ax3.spines['top'].set_visible(False)
    ax3.legend(fontsize='small')
    ax3.spines['left'].set_position(('outward', 10))
    ax3.spines['bottom'].set_position(('outward', 10))


    ## text
    report = classification_report(y_test,model.predict_classes(x_test),output_dict = True)
    ax1.text(x=0.0,y=0.8,s='Classification Report')
    ax1.axhline(y=0.77,lw=0.6,xmax=0.25)
    ax1.text(x=0.0,y=0.65,s='Setosa')
    ax1.text(x=0.0,y=0.5,s=f'Precision   --> {round(report["0"]["precision"],2)}')
    ax1.text(x=0.0,y=0.4,s=f'Recall        --> {round(report["0"]["recall"],2)}')
    ax1.text(x=0.0,y=0.3,s=f'F1              --> {round(report["0"]["f1-score"],2)}')
    ax1.text(x=0.3,y=0.65,s='Versicolor')
    ax1.text(x=0.3,y=0.5,s=f'Precision   --> {round(report["1"]["precision"],2)}')
    ax1.text(x=0.3,y=0.4,s=f'Recall        --> {round(report["1"]["recall"],2)}')
    ax1.text(x=0.3,y=0.3,s=f'F1              --> {round(report["1"]["f1-score"],2)}')
    ax1.text(x=0.6,y=0.65,s='Virginica')
    ax1.text(x=0.6,y=0.5,s=f'Precision   --> {round(report["2"]["precision"],2)}')
    ax1.text(x=0.6,y=0.4,s=f'Recall        --> {round(report["2"]["recall"],2)}')
    ax1.text(x=0.6,y=0.3,s=f'F1              --> {round(report["2"]["f1-score"],2)}')
    ax1.text(x=0.0,y=0.1,s=f'Model Accuracy : {round(report["accuracy"],2)}')
    ax1.spines['right'].set_visible(False)
    ax1.spines['top'].set_visible(False)
    ax1.spines['left'].set_position(('outward', 10))
    ax1.spines['bottom'].set_position(('outward', 10))
    ax1.set_xticklabels([])
    ax1.set_yticklabels([])
    ax1.tick_params(axis='both', which='both',bottom=False,left=False)

    return fig

## creating the sidebar

st.sidebar.title('Teste1')
st.sidebar.success('You\'re about to train a [MLPClassifier](https://en.wikipedia.org/wiki/Multilayer_perceptron) and test it using the [Iris Dataset](https://en.wikipedia.org/wiki/Iris_flower_data_set), but first you have to choose the hyperparameters below. After you\'re done click the \'Train Model!\' button.')


## parameters

test_size = st.sidebar.slider('Test size',min_value=0.01,max_value=0.99,value=0.25)
NUM_EPOCHS = st.sidebar.slider('Number of epochs',min_value=10,max_value=500,step=10,value=100)

input_size = st.sidebar.slider('# neurons of dense_1',min_value=1,max_value=64,value=4,step=1)
dense1 = st.sidebar.slider('# neurons of dense_2',min_value=1,max_value=64,value=4,step=1)
dense2 = st.sidebar.slider('# neurons of dense_3',min_value=1,max_value=64,value=4,step=1)
drop_out = st.sidebar.slider('Dropout',min_value=0.0,max_value=0.8,value=0.2)

act_input = st.sidebar.radio('dense_1 activation function',['relu','tanh','sigmoid','elu','selu','linear'])
act_dense = st.sidebar.radio('dense_2 and dense_3 activation function',['relu','tanh','sigmoid','elu','selu','linear'])


def main():
    x_train,x_test,y_train,y_test,y_train_cat,y_test_cat = _data(test_size)
    model = _define_model(input_size,act_input,dense1,dense2,act_dense,drop_out)
    history,model = train(model,x_train,y_train_cat,NUM_EPOCHS,x_test,y_test_cat)
    fig = model_plot(NUM_EPOCHS,test_size,drop_out,input_size,dense1,dense2,act_input,act_dense,model,x_test,y_test,history)
    return fig

st.sidebar.selectbox('Choose you model',['vgg','alexnet','resnet32'])

if st.sidebar.button('Train model!'):
    st.write(main())
    st.balloons()
else:
    pass

st.sidebar.info('Hey! Nice of you to come here :smile: Let me know if you liked this little experiment!')
st.sidebar.info("Author: Nasser Boan \n\r[> linkedin 👨🏼‍💻](https://www.linkedin.com/in/nasser-boan-2b264a85/)\n\r[> portfolio 💡](nasserboan.github.io)\n\r[> code 🔦](nasserboan.github.io)")
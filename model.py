import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import factor_generator
import label_generator

raw_data = pd.read_csv("raw_data.csv", index_col=[0], header=[0, 1], parse_dates=True)
raw_data = raw_data.dropna(axis=1,inplace=False)
raw_data = raw_data.drop(index=[pd.to_datetime("2019-10-29")])

x_data = factor_generator.get_x_data(raw_data)
y_data = label_generator.get_stock_performance(raw_data.loc[:,"Adj Close"],label_generator.rolling_sr,5,period=int(250/4))

x_data = x_data.dropna(axis=0)
y_data = y_data.dropna(axis=0)
y_data = y_data.loc[x_data.index[0]:,:]
x_data = x_data.loc[:y_data.index[-1],:]


uni_x_data = x_data.loc[:,(slice(None),"ABT")]
uni_y_data = y_data.loc[:,"ABT"]
uni_x_data = (uni_x_data-uni_x_data.mean())/uni_x_data.std()
###########################################################################
def multivariate_data(dataset, target, start_index, end_index, history_size,
                      target_size, step, single_step=False):
  data = []
  labels = []
  start_index = start_index + history_size
  if end_index is None:
    end_index = len(dataset) - target_size
  for i in range(start_index, end_index):
    indices = range(i-history_size, i, step)
    data.append(np.asarray(dataset.iloc[indices,:]))
    if single_step:
      labels.append(target.iloc[i+target_size])
    else:
      labels.append(target.iloc[i:i+target_size])
  return np.asarray(data), np.asarray(labels)

past_history = 100
future_target = 0
STEP = 1
TRAIN_SPLIT =1400
BATCH_SIZE = 100
BUFFER_SIZE = 200
x_train_single, y_train_single = multivariate_data(uni_x_data , uni_y_data, 0,
                                                   TRAIN_SPLIT, past_history,
                                                   future_target, STEP,
                                                   single_step=True)
x_val_single, y_val_single = multivariate_data(uni_x_data , uni_y_data,
                                               TRAIN_SPLIT, None, past_history,
                                               future_target, STEP,
                                               single_step=True)

train_data_single = tf.data.Dataset.from_tensor_slices((x_train_single, y_train_single))
train_data_single = train_data_single.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()

val_data_single = tf.data.Dataset.from_tensor_slices((x_val_single, y_val_single))
val_data_single = val_data_single.batch(BATCH_SIZE).repeat()

del raw_data,x_data,y_data,uni_x_data,uni_y_data

single_step_model = tf.keras.models.Sequential()
single_step_model.add(tf.keras.layers.LSTM(20,input_shape=x_train_single.shape[-2:],return_sequences=True,dropout=0.2))
single_step_model.add(tf.keras.layers.LSTM(30,input_shape=x_train_single.shape[-2:],dropout=0.2))
single_step_model.add(tf.keras.layers.Dense(5,activation='softmax'))
single_step_model.compile(optimizer=tf.keras.optimizers.RMSprop(), loss='sparse_categorical_crossentropy',metrics=['accuracy'])
single_step_model.summary()

single_step_history = single_step_model.fit(train_data_single, epochs=30,steps_per_epoch=200,
                                            validation_data=val_data_single,
                                            validation_steps=50)

plt.plot(single_step_history.history['accuracy'], label='train')
plt.plot(single_step_history.history['val_accuracy'], label='test')
plt.title("accuracy of test & train set")
plt.legend()
plt.show()

def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    #classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           #xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    plt.show()
    return ax,cm

y_pre = single_step_model.predict(x_val_single)
y_pre = np.argmax(y_pre,axis=1)
plot_confusion_matrix(y_val_single, y_pre, classes=5,
                      normalize=False,
                      title=None,
                      cmap=plt.cm.Blues)

np.cov(y_val_single,y_pre)
np.cov(np.diff(y_val_single),np.diff(y_pre))

plt.plot(y_val_single, label="real")
plt.plot(y_pre, label="prediction")
plt.legend()
plt.title("Label time series")
plt.show()

single_step_model.save('my_model2.h5')





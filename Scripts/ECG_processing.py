import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from scipy import optimize
from sklearn.utils.class_weight import compute_class_weight


def make_class_with_unscored_labels(labels, unscored_labels_df):
    df_labels = pd.DataFrame(labels)
    for i in range(len(unscored_labels_df.iloc[0:,1])):
        df_labels.replace(to_replace=str(unscored_labels_df.iloc[i,1]), inplace=True ,value="unscored class", regex=True)
    return df_labels


def plot_classes(classes,y,scored_labels_df, plt_name="my_plot"):
  for j in range(len(classes)):
    for i in range(len(scored_labels_df.iloc[:,1])):
      if (str(scored_labels_df.iloc[:,1][i]) == classes[j]):
        classes[j] = scored_labels_df.iloc[:,0][i]
  plt.figure(figsize=(30,20))
  plt.bar(x=classes,height=y.sum(axis=0))
  plt.title("Distribution of Diagnosis", color = "black")
  plt.tick_params(axis="both", colors = "black")
  plt.xlabel("Diagnosis", color = "black", fontsize=30)
  plt.ylabel("Count", color = "black",fontsize=30)
  plt.xticks(rotation=90, fontsize=20)
  plt.yticks(fontsize = 20)
  plt.savefig("Results/ECG_results/" + plt_name + ".png",dpi=100)
  plt.show()

def train_test_split_unbalanced_data(X,y,samples_pr_class):
    test_index = []
    for i in range(len(y.T)):
        test_index.append(np.random.choice(np.where(y.T[i] == 1)[0],size = samples_pr_class, replace=False))
    test_index = np.unique(np.array(test_index).ravel())

    X_train = X.drop(X.iloc[test_index].index)
    y_train = np.delete(y,test_index,axis=0)
    X_test = X.iloc[test_index]
    y_test = y[test_index]

    return X_train,y_train,X_test,y_test



def NN_ECG(input_shape,output_shape,n_units = 100):

    input_layer = tf.keras.layers.Input(shape=(input_shape)) 
    mod1 = tf.keras.layers.Dense(units=n_units, activation=tf.keras.layers.PReLU() , kernel_initializer='normal')(input_layer)
    mod1 = tf.keras.layers.Dense(units=n_units, activation=tf.keras.layers.LeakyReLU(),  kernel_initializer='normal')(mod1)
    mod1 = tf.keras.layers.Dense(units=n_units, activation="relu" ,  kernel_initializer='normal')(mod1)
    output_layer = tf.keras.layers.Dense(output_shape, activation="sigmoid" , kernel_initializer='normal')(mod1)
    model = tf.keras.models.Model(inputs=input_layer, outputs=output_layer)


    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

    model.compile(loss=tf.losses.BinaryCrossentropy(), optimizer=optimizer, metrics=[tf.metrics.CategoricalAccuracy()])

    return model


def compute_challenge_metric_for_opt(labels, outputs):
    classes=['10370003','111975006','164889003','164890007','164909002','164917005','164934002','164947007','17338001',
 '251146004','270492004','284470004','39732003','426177001','426627000','426783006','427084000','427172004','427393009','445118002','47665007','59118001',
 '59931005','63593006','698252002','713426002','713427006']

    '''
    24 classes
    ['10370003', '111975006', '164889003', '164890007', '164909002', '164917005',
      '164934002', '164947007', '17338001', '251146004', '270492004', '39732003',
      '426177001', '426627000', '426783006' ,'427084000' ,'427393009', '445118002',
      '47665007' ,'59118001', '59931005', '63593006', '698252002', '713426002']

    
      
    '''

    normal_class = '426783006'
    weights = np.array([[1.    , 0.425 , 0.375 , 0.375 , 0.4   , 0.275 , 0.375 , 0.425 ,
        0.5   , 0.45  , 0.425 , 0.4625, 0.475 , 0.425 , 0.425 , 0.375 ,
        0.5   , 0.5   , 0.425 , 0.475 , 0.475 , 0.475 , 0.375 , 0.4625,
        0.475 , 0.425 , 0.475 ],
       [0.425 , 1.    , 0.45  , 0.45  , 0.475 , 0.35  , 0.45  , 0.35  ,
        0.425 , 0.475 , 0.35  , 0.3875, 0.4   , 0.35  , 0.35  , 0.3   ,
        0.425 , 0.425 , 0.35  , 0.4   , 0.4   , 0.45  , 0.45  , 0.3875,
        0.4   , 0.35  , 0.45  ],
       [0.375 , 0.45  , 1.    , 0.5   , 0.475 , 0.4   , 0.5   , 0.3   ,
        0.375 , 0.425 , 0.3   , 0.3375, 0.35  , 0.3   , 0.3   , 0.25  ,
        0.375 , 0.375 , 0.3   , 0.35  , 0.35  , 0.4   , 0.5   , 0.3375,
        0.35  , 0.3   , 0.4   ],
       [0.375 , 0.45  , 0.5   , 1.    , 0.475 , 0.4   , 0.5   , 0.3   ,
        0.375 , 0.425 , 0.3   , 0.3375, 0.35  , 0.3   , 0.3   , 0.25  ,
        0.375 , 0.375 , 0.3   , 0.35  , 0.35  , 0.4   , 0.5   , 0.3375,
        0.35  , 0.3   , 0.4   ],
       [0.4   , 0.475 , 0.475 , 0.475 , 1.    , 0.375 , 0.475 , 0.325 ,
        0.4   , 0.45  , 0.325 , 0.3625, 0.375 , 0.325 , 0.325 , 0.275 ,
        0.4   , 0.4   , 0.325 , 0.375 , 0.375 , 0.425 , 0.475 , 0.3625,
        0.375 , 0.325 , 0.425 ],
       [0.275 , 0.35  , 0.4   , 0.4   , 0.375 , 1.    , 0.4   , 0.2   ,
        0.275 , 0.325 , 0.2   , 0.2375, 0.25  , 0.2   , 0.2   , 0.15  ,
        0.275 , 0.275 , 0.2   , 0.25  , 0.25  , 0.3   , 0.4   , 0.2375,
        0.25  , 0.2   , 0.3   ],
       [0.375 , 0.45  , 0.5   , 0.5   , 0.475 , 0.4   , 1.    , 0.3   ,
        0.375 , 0.425 , 0.3   , 0.3375, 0.35  , 0.3   , 0.3   , 0.25  ,
        0.375 , 0.375 , 0.3   , 0.35  , 0.35  , 0.4   , 0.5   , 0.3375,
        0.35  , 0.3   , 0.4   ],
       [0.425 , 0.35  , 0.3   , 0.3   , 0.325 , 0.2   , 0.3   , 1.    ,
        0.425 , 0.375 , 0.5   , 0.4625, 0.45  , 0.5   , 0.5   , 0.45  ,
        0.425 , 0.425 , 0.5   , 0.45  , 0.45  , 0.4   , 0.3   , 0.4625,
        0.45  , 0.5   , 0.4   ],
       [0.5   , 0.425 , 0.375 , 0.375 , 0.4   , 0.275 , 0.375 , 0.425 ,
        1.    , 0.45  , 0.425 , 0.4625, 0.475 , 0.425 , 0.425 , 0.375 ,
        0.5   , 1.    , 0.425 , 0.475 , 0.475 , 0.475 , 0.375 , 0.4625,
        0.475 , 0.425 , 0.475 ],
       [0.45  , 0.475 , 0.425 , 0.425 , 0.45  , 0.325 , 0.425 , 0.375 ,
        0.45  , 1.    , 0.375 , 0.4125, 0.425 , 0.375 , 0.375 , 0.325 ,
        0.45  , 0.45  , 0.375 , 0.425 , 0.425 , 0.475 , 0.425 , 0.4125,
        0.425 , 0.375 , 0.475 ],
       [0.425 , 0.35  , 0.3   , 0.3   , 0.325 , 0.2   , 0.3   , 0.5   ,
        0.425 , 0.375 , 1.    , 0.4625, 0.45  , 0.5   , 0.5   , 0.45  ,
        0.425 , 0.425 , 0.5   , 0.45  , 0.45  , 0.4   , 0.3   , 0.4625,
        0.45  , 0.5   , 0.4   ],
       [0.4625, 0.3875, 0.3375, 0.3375, 0.3625, 0.2375, 0.3375, 0.4625,
        0.4625, 0.4125, 0.4625, 1.    , 0.4875, 0.4625, 0.4625, 0.4125,
        0.4625, 0.4625, 0.4625, 0.4875, 0.4875, 0.4375, 0.3375, 1.    ,
        0.4875, 0.4625, 0.4375],
       [0.475 , 0.4   , 0.35  , 0.35  , 0.375 , 0.25  , 0.35  , 0.45  ,
        0.475 , 0.425 , 0.45  , 0.4875, 1.    , 0.45  , 0.45  , 0.4   ,
        0.475 , 0.475 , 0.45  , 0.5   , 0.5   , 0.45  , 0.35  , 0.4875,
        0.5   , 0.45  , 0.45  ],
       [0.425 , 0.35  , 0.3   , 0.3   , 0.325 , 0.2   , 0.3   , 0.5   ,
        0.425 , 0.375 , 0.5   , 0.4625, 0.45  , 1.    , 0.5   , 0.45  ,
        0.425 , 0.425 , 0.5   , 0.45  , 0.45  , 0.4   , 0.3   , 0.4625,
        0.45  , 0.5   , 0.4   ],
       [0.425 , 0.35  , 0.3   , 0.3   , 0.325 , 0.2   , 0.3   , 0.5   ,
        0.425 , 0.375 , 0.5   , 0.4625, 0.45  , 0.5   , 1.    , 0.45  ,
        0.425 , 0.425 , 0.5   , 0.45  , 0.45  , 0.4   , 0.3   , 0.4625,
        0.45  , 0.5   , 0.4   ],
       [0.375 , 0.3   , 0.25  , 0.25  , 0.275 , 0.15  , 0.25  , 0.45  ,
        0.375 , 0.325 , 0.45  , 0.4125, 0.4   , 0.45  , 0.45  , 1.    ,
        0.375 , 0.375 , 0.45  , 0.4   , 0.4   , 0.35  , 0.25  , 0.4125,
        0.4   , 0.45  , 0.35  ],
       [0.5   , 0.425 , 0.375 , 0.375 , 0.4   , 0.275 , 0.375 , 0.425 ,
        0.5   , 0.45  , 0.425 , 0.4625, 0.475 , 0.425 , 0.425 , 0.375 ,
        1.    , 0.5   , 0.425 , 0.475 , 0.475 , 0.475 , 0.375 , 0.4625,
        0.475 , 0.425 , 0.475 ],
       [0.5   , 0.425 , 0.375 , 0.375 , 0.4   , 0.275 , 0.375 , 0.425 ,
        1.    , 0.45  , 0.425 , 0.4625, 0.475 , 0.425 , 0.425 , 0.375 ,
        0.5   , 1.    , 0.425 , 0.475 , 0.475 , 0.475 , 0.375 , 0.4625,
        0.475 , 0.425 , 0.475 ],
       [0.425 , 0.35  , 0.3   , 0.3   , 0.325 , 0.2   , 0.3   , 0.5   ,
        0.425 , 0.375 , 0.5   , 0.4625, 0.45  , 0.5   , 0.5   , 0.45  ,
        0.425 , 0.425 , 1.    , 0.45  , 0.45  , 0.4   , 0.3   , 0.4625,
        0.45  , 0.5   , 0.4   ],
       [0.475 , 0.4   , 0.35  , 0.35  , 0.375 , 0.25  , 0.35  , 0.45  ,
        0.475 , 0.425 , 0.45  , 0.4875, 0.5   , 0.45  , 0.45  , 0.4   ,
        0.475 , 0.475 , 0.45  , 1.    , 0.5   , 0.45  , 0.35  , 0.4875,
        0.5   , 0.45  , 0.45  ],
       [0.475 , 0.4   , 0.35  , 0.35  , 0.375 , 0.25  , 0.35  , 0.45  ,
        0.475 , 0.425 , 0.45  , 0.4875, 0.5   , 0.45  , 0.45  , 0.4   ,
        0.475 , 0.475 , 0.45  , 0.5   , 1.    , 0.45  , 0.35  , 0.4875,
        0.5   , 0.45  , 0.45  ],
       [0.475 , 0.45  , 0.4   , 0.4   , 0.425 , 0.3   , 0.4   , 0.4   ,
        0.475 , 0.475 , 0.4   , 0.4375, 0.45  , 0.4   , 0.4   , 0.35  ,
        0.475 , 0.475 , 0.4   , 0.45  , 0.45  , 1.    , 0.4   , 0.4375,
        0.45  , 0.4   , 1.    ],
       [0.375 , 0.45  , 0.5   , 0.5   , 0.475 , 0.4   , 0.5   , 0.3   ,
        0.375 , 0.425 , 0.3   , 0.3375, 0.35  , 0.3   , 0.3   , 0.25  ,
        0.375 , 0.375 , 0.3   , 0.35  , 0.35  , 0.4   , 1.    , 0.3375,
        0.35  , 0.3   , 0.4   ],
       [0.4625, 0.3875, 0.3375, 0.3375, 0.3625, 0.2375, 0.3375, 0.4625,
        0.4625, 0.4125, 0.4625, 1.    , 0.4875, 0.4625, 0.4625, 0.4125,
        0.4625, 0.4625, 0.4625, 0.4875, 0.4875, 0.4375, 0.3375, 1.    ,
        0.4875, 0.4625, 0.4375],
       [0.475 , 0.4   , 0.35  , 0.35  , 0.375 , 0.25  , 0.35  , 0.45  ,
        0.475 , 0.425 , 0.45  , 0.4875, 0.5   , 0.45  , 0.45  , 0.4   ,
        0.475 , 0.475 , 0.45  , 0.5   , 0.5   , 0.45  , 0.35  , 0.4875,
        1.    , 0.45  , 0.45  ],
       [0.425 , 0.35  , 0.3   , 0.3   , 0.325 , 0.2   , 0.3   , 0.5   ,
        0.425 , 0.375 , 0.5   , 0.4625, 0.45  , 0.5   , 0.5   , 0.45  ,
        0.425 , 0.425 , 0.5   , 0.45  , 0.45  , 0.4   , 0.3   , 0.4625,
        0.45  , 1.    , 0.4   ],
       [0.475 , 0.45  , 0.4   , 0.4   , 0.425 , 0.3   , 0.4   , 0.4   ,
        0.475 , 0.475 , 0.4   , 0.4375, 0.45  , 0.4   , 0.4   , 0.35  ,
        0.475 , 0.475 , 0.4   , 0.45  , 0.45  , 1.    , 0.4   , 0.4375,
        0.45  , 0.4   , 1.    ]])
    
    num_recordings, num_classes = np.shape(labels)
    normal_index = classes.index(normal_class)

    # Compute the observed score.
    A = compute_modified_confusion_matrix(labels, outputs)
    observed_score = np.nansum(weights * A)

    # Compute the score for the model that always chooses the correct label(s).
    correct_outputs = labels
    A = compute_modified_confusion_matrix(labels, correct_outputs)
    correct_score = np.nansum(weights * A)

    # Compute the score for the model that always chooses the normal class.
    inactive_outputs = np.zeros((num_recordings, num_classes), dtype=np.bool)
    inactive_outputs[:, normal_index] = 1
    A = compute_modified_confusion_matrix(labels, inactive_outputs)
    inactive_score = np.nansum(weights * A)

    if correct_score != inactive_score:
        normalized_score = float(observed_score - inactive_score) / float(correct_score - inactive_score)
    else:
        normalized_score = float('nan')

    return normalized_score

def compute_modified_confusion_matrix(labels, outputs):
    # Compute a binary multi-class, multi-label confusion matrix, where the rows
    # are the labels and the columns are the outputs.
    num_recordings, num_classes = np.shape(labels)
    A = np.zeros((num_classes, num_classes))

    # Iterate over all of the recordings.
    for i in range(num_recordings):
        # Calculate the number of positive labels and/or outputs.
        normalization = float(max(np.sum(np.any((labels[i, :], outputs[i, :]), axis=0)), 1))
        # Iterate over all of the classes.
        for j in range(num_classes):
            # Assign full and/or partial credit for each positive class.
            if labels[i, j]:
                for k in range(num_classes):
                    if outputs[i, k]:
                        A[j, k] += 1.0/normalization

    return A


def thr_chall_metrics(thr, label, output_prob):
    return -compute_challenge_metric_for_opt(label, np.array(output_prob>thr))


def compute_modified_confusion_matrix_nonorm(labels, outputs):
    # Compute a binary multi-class, multi-label confusion matrix, where the rows
    # are the labels and the columns are the outputs.
    num_recordings, num_classes = np.shape(labels)
    A = np.zeros((num_classes, num_classes))

    # Iterate over all of the recordings.
    for i in range(num_recordings):
        # Calculate the number of positive labels and/or outputs.
        #####normalization = float(max(np.sum(np.any((labels[i, :], outputs[i, :]), axis=0)), 1))
        # Iterate over all of the classes.
        for j in range(num_classes):
            # Assign full and/or partial credit for each positive class.
            if labels[i, j]:
                for k in range(num_classes):
                    if outputs[i, k]:
                        A[j, k] += 1.0#/normalization

    return A


def calculating_class_weights(y_true):
    number_dim = np.shape(y_true)[1]
    weights = np.empty([number_dim, 2])
    for i in range(number_dim):
        weights[i] = compute_class_weight('balanced', [0.,1.], y_true[:, i])
    return weights

def compute_beta_measures(labels, outputs, beta):
    num_recordings, num_classes = np.shape(labels)

    A = compute_confusion_matrices(labels, outputs, normalize=True)

    f_beta_measure = np.zeros(num_classes)
    g_beta_measure = np.zeros(num_classes)
    for k in range(num_classes):
        tp, fp, fn, tn = A[k, 1, 1], A[k, 1, 0], A[k, 0, 1], A[k, 0, 0]
        if (1+beta**2)*tp + fp + beta**2*fn:
            f_beta_measure[k] = float((1+beta**2)*tp) / float((1+beta**2)*tp + fp + beta**2*fn)
        else:
            f_beta_measure[k] = float('nan')
        if tp + fp + beta*fn:
            g_beta_measure[k] = float(tp) / float(tp + fp + beta*fn)
        else:
            g_beta_measure[k] = float('nan')

    macro_f_beta_measure = np.nanmean(f_beta_measure)
    macro_g_beta_measure = np.nanmean(g_beta_measure)

    return macro_f_beta_measure, macro_g_beta_measure

def compute_modified_confusion_matrix(labels, outputs):
    # Compute a binary multi-class, multi-label confusion matrix, where the rows
    # are the labels and the columns are the outputs.
    num_recordings, num_classes = np.shape(labels)
    A = np.zeros((num_classes, num_classes))

    # Iterate over all of the recordings.
    for i in range(num_recordings):
        # Calculate the number of positive labels and/or outputs.
        normalization = float(max(np.sum(np.any((labels[i, :], outputs[i, :]), axis=0)), 1))
        # Iterate over all of the classes.
        for j in range(num_classes):
            # Assign full and/or partial credit for each positive class.
            if labels[i, j]:
                for k in range(num_classes):
                    if outputs[i, k]:
                        A[j, k] += 1.0/normalization

    return A

def compute_confusion_matrices(labels, outputs, normalize=False):
    # Compute a binary confusion matrix for each class k:
    #
    #     [TN_k FN_k]
    #     [FP_k TP_k]
    #
    # If the normalize variable is set to true, then normalize the contributions
    # to the confusion matrix by the number of labels per recording.
    num_recordings, num_classes = np.shape(labels)

    if not normalize:
        A = np.zeros((num_classes, 2, 2))
        for i in range(num_recordings):
            for j in range(num_classes):
                if labels[i, j]==1 and outputs[i, j]==1: # TP
                    A[j, 1, 1] += 1
                elif labels[i, j]==0 and outputs[i, j]==1: # FP
                    A[j, 1, 0] += 1
                elif labels[i, j]==1 and outputs[i, j]==0: # FN
                    A[j, 0, 1] += 1
                elif labels[i, j]==0 and outputs[i, j]==0: # TN
                    A[j, 0, 0] += 1
                else: # This condition should not happen.
                    raise ValueError('Error in computing the confusion matrix.')
    else:
        A = np.zeros((num_classes, 2, 2))
        for i in range(num_recordings):
            normalization = float(max(np.sum(labels[i, :]), 1))
            for j in range(num_classes):
                if labels[i, j]==1 and outputs[i, j]==1: # TP
                    A[j, 1, 1] += 1.0/normalization
                elif labels[i, j]==0 and outputs[i, j]==1: # FP
                    A[j, 1, 0] += 1.0/normalization
                elif labels[i, j]==1 and outputs[i, j]==0: # FN
                    A[j, 0, 1] += 1.0/normalization
                elif labels[i, j]==0 and outputs[i, j]==0: # TN
                    A[j, 0, 0] += 1.0/normalization
                else: # This condition should not happen.
                    raise ValueError('Error in computing the confusion matrix.')

    return A
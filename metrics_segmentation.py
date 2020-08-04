from tensorflow.keras.metrics import CategoricalAccuracy
import tensorflow.keras.backend as K
import tensorflow as tf

        
def jaccard(y_true, y_pred):
    y_true_arg = K.flatten(tf.math.argmax(y_true, axis=-1))
    y_pred_arg = K.flatten(tf.math.argmax(y_pred, axis=-1))
     
    conf_mtx = tf.cast(tf.math.confusion_matrix(y_true_arg,y_pred_arg, num_classes=5), tf.float32)
     
    FN = tf.math.reduce_sum(conf_mtx[2:,0:2])
    FP = tf.math.reduce_sum(conf_mtx[0:2,2:])
    TP = tf.math.reduce_sum(conf_mtx[2:,2:])
           
    return TP/(TP+FP+FN+K.epsilon())


def dice(y_true, y_pred): #any shape can go - can't be a loss function
    
    y_true_arg = K.flatten(tf.math.argmax(y_true, axis=-1))
    y_pred_arg = K.flatten(tf.math.argmax(y_pred, axis=-1))
             
    conf_mtx = tf.cast(tf.math.confusion_matrix(y_true_arg,y_pred_arg, num_classes=5), tf.float32)
     
    FN = tf.math.reduce_sum(conf_mtx[2:,0:2])
    FP = tf.math.reduce_sum(conf_mtx[0:2,2:])
    TP = tf.math.reduce_sum(conf_mtx[2:,2:])
    
    return 2*TP/(2*TP+FP+FN+K.epsilon())

def init_metrics(paramDict):
    
    def jaccard_metric():
        def fn(y_true, y_pred):
            return jaccard(y_true, y_pred)
        
        fn.__name__ = 'jaccard'
        return fn
      
    def dice_metric():
        def fn(y_true, y_pred):
            return dice(y_true, y_pred)
        
        fn.__name__ = 'dice'
        return fn
    
            
    metrics = ['mse', CategoricalAccuracy(), jaccard_metric(), dice_metric()]

    return metrics

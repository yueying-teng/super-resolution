# -*- coding: utf-8 -*-
import tensorflow as tf
from tensorflow.python.keras import backend as K


# logger = logging.getLogger(__name__)


class CustomLearningRateScheduler(tf.keras.callbacks.Callback):
    """
    Learning rate scheduler which sets the learning rate according to schedule.
    Arguments:
        schedule: a function that takes an epoch index
            (integer, indexed from 0) and current learning rate
            as inputs and returns a new learning rate as output (float).
    """

    def __init__(self, schedule, epoch_size, verbose=0):
        super(CustomLearningRateScheduler, self).__init__()
        self.schedule = schedule
        self.verbose = verbose
        self.epoch_size = epoch_size
        self.cnt = 0

    def on_batch_begin(self, batch, logs=None):
        if not hasattr(self.model.optimizer, "lr"):
            raise ValueError('Optimizer must have a "lr" attribute.')
        # Get the current learning rate from model's optimizer.
        lr = float(tf.keras.backend.get_value(self.model.optimizer.learning_rate))
        # Call schedule function to get the scheduled learning rate.
        cum_batch = batch + self.cnt *(self.epoch_size -1)
        if batch == self.epoch_size-1:
            self.cnt +=1

        scheduled_lr = self.schedule(cum_batch)
       
        # Set the value back to the optimizer before this epoch starts
        tf.keras.backend.set_value(self.model.optimizer.lr, scheduled_lr)
        
        if self.verbose > 0:
            print('cum batch and epoch counter', cum_batch, self.cnt)
            print("\nbatch %04d: Learning rate is %6.8f." % (batch + 1, scheduled_lr))
    
    def on_epoch_end(self, epoch, logs={}):
        lr = float(tf.keras.backend.get_value(self.model.optimizer.learning_rate))
        logs['learning_rate'] =lr
        if self.verbose > 0:
            print('epoch end learning rate', lr)
      
 

from keras.initializers import glorot_uniform
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import SGD, RMSprop
from keras.regularizers import l2
import keras.backend as K

#Loss Function
def negative_log_likelihood(E):
    def loss(y_true, y_pred):
        hazard_ratio = K.exp(y_pred)
        log_risk = K.log(K.cumsum(hazard_ratio, axis=1))
        uncensored_likelihood = y_pred - log_risk
        censored_likelihood = uncensored_likelihood * E
        neg_likelihood = -K.sum(censored_likelihood)
        return neg_likelihood
    return loss


#Keras model
def build_model():
    model = Sequential()
    model.add(Dense(32, input_shape=(7,),
                    kernel_initializer=glorot_uniform())) # shape= length, dimension
    model.add(Activation('relu'))
    model.add(Dense(32, kernel_initializer=glorot_uniform()))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation="linear",
                    kernel_initializer=glorot_uniform(),
                    kernel_regularizer=l2(0.01),
                    activity_regularizer=l2(0.01)))
    return model

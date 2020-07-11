from keras.models import Sequential
from keras.initializers import RandomNormal
from keras.layers import Dense, BatchNormalization, Input, Dropout, Activation
from keras import optimizers

def define_network(inp):
    def dn():
        random_norm_init = RandomNormal(seed=1)

        model = Sequential()
        model.add(Dense(128, input_shape=(inp.shape[1],), kernel_initializer=random_norm_init, bias_initializer=random_norm_init, activation='relu'))

        model.add(Dense(256, kernel_initializer=random_norm_init))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        #model.add(Dropout(0.11))
        model.add(Dense(256, kernel_initializer=random_norm_init,activation='relu'))
        model.add(Dropout(0.09))

        model.add(Dense(256, kernel_initializer=random_norm_init,activation='relu'))
        model.add(Dropout(0.11))

        model.add(Dense(1, activation='relu'))

        adam = optimizers.Adam(learning_rate=0.00008, beta_1=0.9, beta_2=0.999, amsgrad=False)
        #sgd = optimizers.SGD(learning_rate=0.001, momentum=0.1, nesterov=True)
        #adadelta = optimizers.Adadelta(learning_rate=0.009)

        model.compile(loss='mse', optimizer=adam, metrics=['mse','mae'])
        return model
    
    return dn
    

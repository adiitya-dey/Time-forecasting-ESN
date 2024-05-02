import tensorflow as tf
from einops import rearrange

tf.random.set_seed(123)

class ESN(tf.keras.Model):
    def __init__(self, reservoir_size=100, input_size=1, output_size=1,  spectral_radius=1.0, connectivity_rate=1.0, learning_rate = 0.1, epochs=1, washout=1, activation="tanh"):
        
        super(ESN, self).__init__()
        self.reservoir_size = reservoir_size
        self.input_size = input_size
        self.epochs = epochs
        self.connectivity_rate = connectivity_rate
        self.spectral_radius = spectral_radius
        self.washout = washout
        self.output_size = output_size
        self.lr = learning_rate
        self.activation = self.activation_fn(activation)

        self.state = tf.zeros((self.reservoir_size, 1))
        self.W_in = tf.random.uniform((self.reservoir_size, self.input_size), minval=-1, maxval=1)

        self.W_out = None

        ## Initializing Reservoir Weights according to "Re-visiting the echo state property"(2012)
        ##
        ## Initialize a random matrix and induce sparsity.
        self.W_res = tf.random.normal((reservoir_size, reservoir_size))
        self.W_res = tf.where(tf.random.normal(self.W_res.shape) > self.connectivity_rate, tf.zeros_like(self.W_res), self.W_res)


        ## Scale the matrix based on user defined spectral radius.
        current_spectral_radius = tf.reduce_max(tf.abs(tf.linalg.eigvals(self.W_res)))
        self.W_res = self.W_res * (self.spectral_radius / current_spectral_radius)        

        self.all_states = [self.state]

    
    @staticmethod
    def activation_fn(x):
         
        activation_keys = ["sigmoid", "relu", "tanh"]

        if x in activation_keys:
              if x == "tanh":
                   return tf.keras.activations.tanh
              elif x == "relu":
                   return tf.keras.activations.relu
              elif x == "sigmoid":
                   return tf.keras.activations.sigmoid
            
        else:
            raise ValueError(f"Activation {x} does not exists")
        

    def fit(self, X_train, y_train):
        

        state_collection_matrix = tf.zeros((self.input_size + self.reservoir_size, 1))
        # self.state = np.zeros((self.reservoir_size, 1))

        ## Calculate state of reservoirs per time step
        for i in range(X_train.shape[0]-1):

            

            input = rearrange(X_train[i], 'c -> c 1')
            
            # print(input.dtype, self.W_in.dtype)
            input_product = self.W_in@input
            state_product = self.W_res@self.state
            self.state = self.activation(input_product + state_product)
            state_collection_matrix= tf.concat([state_collection_matrix, tf.concat([self.state, input], axis=0)], axis=1)

            self.all_states.append(self.state)

        ## Update W_out
        mat1 = tf.transpose(state_collection_matrix)[self.washout:,:]
        self.W_out = tf.transpose(tf.linalg.lstsq(matrix=mat1, rhs=y_train[self.washout:,:], l2_regularizer=0.01))

        
        
    def predict(self, X_test):
            prediction = tf.zeros((self.output_size,1))
            for i in range(X_test.shape[0]- 1):
                input = rearrange(X_test[i], 'c -> c 1')
                
                input_product = self.W_in@input
                state_product = self.W_res@self.state
                self.state = self.activation(input_product + state_product)
                concat_matrix= tf.concat([self.state, input], axis=0)
                # print(self.W_out.shape, concat_matrix.shape)
                pred =  self.W_out@concat_matrix
                prediction = tf.concat([prediction, pred], axis=1)

                self.all_states.append(self.state)
            
            prediction = rearrange(prediction, 'c r -> r c')
            if self.output_size == self.input_size:
                return prediction[1:,:]
            else:
                return prediction
import numpy as np

class SimpleCNN:
    def __init__(self):
        ## Initialize weights and bias
        self.weights = {"conv1": np.random.randn(3,3,1,32)*0.01, ## 3X3 filters, 1 input channel, 32 kernal size
                        "conv2":np.random.randn(3,3,32,64)*0.01, ## 3X3 filters, 32 input channel, 64 kernal size
                        "flt1": np.random.randn(7*7*64, 128), ## Flatten layer size
                        "flt2": np.random.randn(128, 10)*0.01} ## Output layer
        
        self.bias = {"conv1": np.zeros((1,1,32)),
                     "conv2": np.zeros((1,1,64)),
                     "flt1": np.zeros((1, 128)),
                     "flt2": np.zeros((1,10))}
        

    def relu(self, x):
        return np.maximum(0, x)

    def softmax(self, x):
        exp_x = np.exp(x-np.max(x, axis = 1, keepdims=True))
        res = exp_x/np.sum(exp_x, axis = 1, keepdims=True)
        return res

    def conv_forward(self, A_prev, W, b, stride = 1, padding = 0):
        # Add padding to the input
        A_prev_padded = np.pad(A_prev, 
                               ((0,), (padding, ), (padding, ), (0, )), 
                                mode = "constant",
                                constant_values=0)
        n_H, n_W, n_C = A_prev.shape[1], A_prev.shape[2], A_prev.shape[3]
        f, f, n_C_prev, n_C = W.shape
        n_H_out = int((n_H-f+2*padding)/stride)+1
        n_W_out = int((n_W-f+2*padding)/stride)+1
        Z = np.zeros((A_prev.shape[0], n_H_out, n_W_out, n_C))

        for h in range(n_H_out):
            for w in range(n_W_out):
                for c in range(n_C):
                    Z[:, h, w, c] = np.sum(A_prev_padded[:, h*stride: h*stride+f,
                                                         w*stride: w*stride+f]*W[:, :, :, c],
                                                         axis = (1,2,3))
        return Z + b
    
    def pooling_forward(self, A_prev, f = 2, stride = 2):
        n_H, n_W, n_C = A_prev.shape[1], A_prev.shape[2], A_prev.shape[3]
        n_H_out = int((n_H - f)/stride)+1
        n_W_out = int((n_W - f)/stride)+1
        Z = np.zeros((A_prev.shape[0], n_H_out, n_W_out, n_C))
        for h in range(n_H_out):
            for w in range(n_W_out):
                Z[:, h, w, :] = np.max(A_prev[:, h*stride: h*stride+f,
                                             w*stride:w*stride+f, :],
                                             axis = (1,2))
                
        return Z
    

    def flatten(self, A):
        return A.reshape(A.shape[0], -1)
    
    def fc_forward(self, A_prev, W, b):
        return A_prev.dot(W)+b
    
    def conv_backward(self, dZ, A_prev, W, b, stride = 1, padding = 0):
        # Get the dimenssion
        m, n_H_prev, n_W_prev, n_C_prev = A_prev.shape
        f,f, n_C_prev, n_C = W.shape
        n_H, n_W, n_C = dZ.shape

        # Initialize the gradients
        dA_prev = np.zeros_like(A_prev)
        dW = np.zeros_like(W)
        db = np.zeros_like(b)

        ## Add padding to A_prev
        A_prev_padded = np.pad(A_prev, ((0,), (padding, ), (padding, ), (0,)), mode="constant", constant_values=0)

        # Backepropagation through the convolution
        for i in range(m): # Loop over the batch size
            for h in range(n_H): # Loop over the Heights
                for w in range(n_W): # Loop over the Weights
                    for c in range(n_C): # Loop over the channels
                        # Find the corners of the current slice
                        vert_start = h * stride
                        vert_end = vert_start+f
                        horiz_start = w*stride
                        horiz_end = horiz_start+f
                        # Backpropagation to dA_prev
                        dA_prev[i, vert_start:vert_end, horiz_start:horiz_end, :] += dZ[i, h, w, c]* W[:,:,:,c]
                        # Backpropagation to dW
                        dW[:,:,:,c] += dZ[i, h, w, c]*A_prev_padded[i, vert_start:vert_end, horiz_start:horiz_end, :]
                        # Backpropagation for db
                        db[:,:,c]+=dZ[i,h,w,c]

        ## Remove padding from dA_prev
        if padding > 0:
            dA_prev = dA_prev[:, padding:-padding, padding:-padding, :]

        return dA_prev, dW, db
    
    ## Backpropagation for Pooling Layer

    def pooling_backward(self, dA, A_prev, f = 2, stride = 2):
        m, n_H_prev, n_W_prev, n_C_prev = A_prev.shape
        n_H = int((n_H_prev - f)/stride)+1
        n_W = int((n_W_prev - f)/stride)+1
        # Initialize the gradients
        dA_prev = np.zeros_like(A_prev)
        # Backpropagation through the pooling layer
        for i in range(m): # Loop over the bacth size
            for h in range(n_H): # loop over the height
                for w in range(n_W): # Loop over the Width
                    ## Find corners of the current slice
                    vert_start = h*stride
                    vert_end = vert_start+f
                    horiz_start =w*stride
                    horiz_end = horiz_start+f
                    # Find the index of the maximum value in the pooling window
                    a_prev_slice = A_prev[i, vert_start:vert_end, horiz_start:horiz_end, :]
                    mask = (a_prev_slice == np.max(a_prev_slice, axis = (0,1), keepdims=True))
                    # Backpropagate the gradient to the input
                    dA_prev[i,vert_start:vert_end, horiz_start:horiz_end, :] += mask*dA[i,h,w,:]

        return dA_prev
    
    ## backward propagation for Fully Connected Layer
    def fc_backward(self, dA, A_prev, W):
        # Compute the gradients
        dW = A_prev.T.dot(dA)
        db = np.sum(dA, axis = 0, keepdims=True)
        dA_prev = dA.dot(W.T)
        return dA_prev, dW, db
    
    ## Train the model
    def train(self, x_train, y_train, learning_rate = 0.01, epochs = 10, batch_size = 64):
        for epoch in range(epochs):
            for i in range(0, len(x_train), batch_size):
                # Get the batch
                X_batch = x_train[i:i+batch_size]
                Y_batch = y_train[i:i+batch_size]
                # Forward pass
                Z1 = self.conv_forward(X_batch, self.weights['conv1'], self.bias['conv1'])
                A1 = self.relu(Z1)
                P1 = self.pooling_forward(A1)
                Z2 = self.conv_forward(P1, self.weights['conv2'], self.bias['conv2'])
                A2 = self.relu(Z2)
                P2 = self.pooling_forward(A2)
                F1 = self.flatten(P2)
                Z3 = self.fc_forward(F1, self.weights['flt1'], self.bias["flt1"])
                A3 = self.relu(Z3)
                Z4 = self.fc_forward(A3, self.weights['flt2'], self.bias['flt2'])
                A4 = self.softmax(Z4)
                # Compute the loss(cross-entropy)
                # Add small value to avoid log(0)
                loss = -np.mean(np.sum(Y_batch*np.log(A4+ 1e-8), axis = 1))
                # Backward pass
                dA4 = A4 - Y_batch # Derivative of softmax loss
                dA3, dW2, db2 = self.fc_backward(dA4, A3, self.weights['flt2'])
                dA3 = dA3*(A3>0) # Apply ReLU derivative
                dA2_flat, dW1, db1 = self.fc_backward(dA3, F1, self.weights['flt1'])
                # Reshape the flatten gradient back to the pooling output shape
                dA2 = dA2_flat.reshape(P2.shape)
                dP2 = self.pooling_backward(dA2, A2)
                dA1 = self.relu(dP2) # Apply ReLU derivation
                dZ2, dW_conv2, db_conv2 = self.conv_backward(dA1, P1, self.weights['conv2'], self.bias["conv2"])
                dP1 = self.pooling_backward(dZ2, A1)
                dZ1, dW_conv1, db_conv1 = self.conv_backward(dP1, X_batch, self.weights['conv1'], self.bias["conv1"])
                # Update weights and biases
                self.weights['flt2'] -= learning_rate*dW2
                self.bias['flt2'] -= learning_rate*db2
                self.weights['flt1'] -= learning_rate*dW1
                self.bias['flt1'] -= learning_rate*db1
                self.weights['conv2'] -= learning_rate*dW_conv2
                self.bias['conv2'] -= learning_rate*db_conv2
                self.weights['conv1'] -= learning_rate*dW_conv1
                self.bias['conv1'] -= learning_rate*db_conv1
        return f"Epoch {epoch+1}/{epochs}, Loss: {loss: .4f}"
    

    def evalute(self, x_test, y_test):
        # Initialize variables to track correct prediction
        corr_pred = 0
        total_samples = len(x_test)

        # Iterate over the test samples
        for i in range(total_samples):
            # Perform a forward pass through the network
            Z1 = self.conv_forward(x_test[i:i+1], self.weights['conv1'], self.bias['conv1'])
            A1 = self.relu(Z1)
            P1 = self.pooling_forward(A1)
            Z2 = self.conv_forward(P1, self.weights["conv2"], self.bias['conv2'])
            A2 = self.relu(Z2)
            P2 = self.pooling_forward(A2)
            F1 = self.flatten(P2)
            Z3 = self.fc_forward(F1, self.weights['flt1'], self.bias['flt1'])
            A3 = self.relu(Z3)
            Z4 = self.fc_forward(A3, self.weights['flt2'], self.bias['flt2'])
            A4 = self.softmax(Z4)
            # Deteremine the predicted class
            pred_class = np.argmax(A4, axis = 1)
            true_class = np.argmax(y_test[i])
            # Check if the prediction is correct
            if pred_class == true_class:
                corr_pred += 1
        acc = corr_pred/total_samples
        return f"Model accuracy: {acc*100:.2f}%"
    



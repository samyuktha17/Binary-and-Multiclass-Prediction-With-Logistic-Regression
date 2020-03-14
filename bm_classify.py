import numpy as np


def binary_train(X, y, loss="perceptron", w0=None, b0=None, step_size=0.5, max_iterations=1000):
    """
    Inputs:
    - X: training features, a N-by-D numpy array, where N is the 
    number of training points and D is the dimensionality of features
    - y: binary training labels, a N dimensional numpy array where 
    N is the number of training points, indicating the labels of 
    training data
    - loss: loss type, either perceptron or logistic
    - step_size: step size (learning rate)
	- max_iterations: number of iterations to perform gradient descent
    w <-- w + step_size * (y_n sign) * x_n * 1/N, where I've made y_n sign -1 if y_n = 0 and 1 if y_n = 1
    Returns:
    - w: D-dimensional vector, a numpy array which is the weight 
    vector of logistic or perceptron regression
    - b: scalar, which is the bias of logistic or perceptron regression
    """
    N, D = X.shape
    assert len(np.unique(y)) == 2

    y=np.where(y==0,-1,1)
    w = np.zeros(D)
    if w0 is not None:
        w = w0
    
    b = 0
    if b0 is not None:
        b = b0

    if loss == "perceptron":
        ############################################
        # TODO 1 : Edit this if part               #
        #          Compute w and b here            #
        #w = np.zeros(D)
        #b = 0
        #wrong implementation:
        for _ in range(max_iterations):
            indices=np.where(np.multiply(y,np.dot(X,w)+b))
            gradient=(np.dot(y[indices],X[indices]))/N
            w= w + (step_size*(gradient))
            b= b + (step_size*(np.sum(y[indices]))/N)           
            
        ############################################
        

    elif loss == "logistic":
        ############################################
        # TODO 2 : Edit this if part               #
        #          Compute w and b here            #
        #w = np.zeros(D)
        #b = 0
        for _ in range(max_iterations):
            z=np.dot(X,w)+b #Nx1 np array
            loss=sigmoid(z)-y
            w=w- (step_size*(np.dot(loss,X)/N))
            b=b-(step_size*(np.sum(loss)/N))
        ############################################
        

    else:
        raise "Loss Function is undefined."

    assert w.shape == (D,)
    return w, b

def sigmoid(z):
    
    """
    Inputs:
    - z: a numpy array or a float number
    
    Returns:
    - value: a numpy array or a float number after computing sigmoid function value = 1/(1+exp(-z)).
    """

    ############################################
    # TODO 3 : Edit this part to               #
    #          Compute value                   #
    
    value = 1/(1+np.exp(-1*z))
    ############################################
    
    return value

def binary_predict(X, w, b, loss="perceptron"):
    """
    Inputs:
    - X: testing features, a N-by-D numpy array, where N is the 
    number of training points and D is the dimensionality of features
    - w: D-dimensional vector, a numpy array which is the weight 
    vector of your learned model
    - b: scalar, which is the bias of your model
    - loss: loss type, either perceptron or logistic
    
    Returns:
    - preds: N dimensional vector of binary predictions: {0, 1}
    """
    N, D = X.shape
    
    if loss == "perceptron":
        ############################################
        # TODO 4 : Edit this if part               #
        #          Compute preds                   #
        preds = np.zeros(N)
        out = np.dot(X,w)+b
        for i in range(N):
            if out[i]>0:
                preds[i]=1
            else:
                preds[i]=0
        ############################################
        

    elif loss == "logistic":
        ############################################
        # TODO 5 : Edit this if part               #
        #          Compute preds                   #
        preds = np.zeros(N)
        out= np.dot(X,w)+b
        for i in range(N):
            if sigmoid(out[i])>=0.5:
                preds[i]=1
            else:
                preds[i]=0
        ############################################
        

    else:
        raise "Loss Function is undefined."
    

    assert preds.shape == (N,) 
    return preds



def multiclass_train(X, y, C,
                     w0=None, 
                     b0=None,
                     gd_type="sgd",
                     step_size=0.5, 
                     max_iterations=1000):
    """
    Inputs:
    - X: training features, a N-by-D numpy array, where N is the 
    number of training points and D is the dimensionality of features
    - y: multiclass training labels, a N dimensional numpy array where
    N is the number of training points, indicating the labels of 
    training data
    - C: number of classes in the data
    - gd_type: gradient descent type, either GD or SGD
    - step_size: step size (learning rate)
    - max_iterations: number of iterations to perform gradient descent

    Returns:
    - w: C-by-D weight matrix of multinomial logistic regression, where 
    C is the number of classes and D is the dimensionality of features.
    - b: bias vector of length C, where C is the number of classes
    """

    N, D = X.shape

    w = np.zeros((C, D))
    if w0 is not None:
        w = w0
    
    b = np.zeros(C)
    if b0 is not None:
        b = b0

        
    if gd_type == "sgd":
        ############################################
        # TODO 6 : Edit this if part               #
        #          Compute w and b                 #
        
        w = np.zeros((C, D))
        b = np.zeros(C)

        np.random.seed(0)
        for _ in range(max_iterations):
            #choose a random value for entry (x,y) since sgd
            myrandval=np.random.choice(N) #pickn = np.random.choice(N)
            x_new=X[myrandval] # xn = X[pickn]
            y_new=y[myrandval] #yn = y[pickn]
            xi=np.dot(x_new,np.transpose(w))+b #loss = xn.dot(w.T) + b  # 1*C
            z=xi-xi.max() # as instructed for numerical integrity # loss -= loss.max()
            k=np.exp(z) # yp = np.exp(loss)
            y_pred=k/np.sum(k) # yp /= yp.sum()
            y_pred[y_new]=y_pred[y_new]-1 #softmax-1 when yn=y err = yp;  err[yn] -= 1
            gradient=np.dot(np.transpose(y_pred),x_new) # update = np.dot(err.reshape(C, 1), xn.reshape(1, D))
            w=w-((step_size)*(gradient)) #w -= step_size * update
            b=b-((step_size)*(y_pred)) # b -= step_size * err
        ############################################
        
    
    elif gd_type == "gd":
        ############################################
        # TODO 7 : Edit this if part               #
        #          Compute w and b                 #
        w = np.zeros((C, D))
        b = np.zeros(C)
        for it in range(max_iterations):
            result= np.dot(X,np.transpose(w))+b # loss=X.dot(w.T) + b  # N*C 
            y_pred=np.exp(result) #yp = np.exp(loss) yes
            k=y_pred/y_pred.sum(axis=1,keepdims=True) #yp /= yp.sum(axis=1, keepdims=True)
            ohm=np.zeroes([N,C]) #onehot = np.zeros([N, C])
            #onehot[np.arange(N), y.astype(int)] = 1.0
            for i in range(len(ohm)):
                ohm[i][y[i]]=1.0
            result=k-ohm #err = yp - onehot
            
            gradient=(np.dot(np.transpose(result),X))/N #update = np.dot(err.T, X)
            w=w-(step_size)*(gradient) # w -= step_size/N * update
            b=b-(step_size)*((np.sum(result,axis=0))/N) #  b -= step_size/N * err.sum(axis=0)

        ############################################
        

    else:
        raise "Type of Gradient Descent is undefined."
    

    assert w.shape == (C, D)
    assert b.shape == (C,)

    return w, b


def multiclass_predict(X, w, b):
    """
    Inputs:
    - X: testing features, a N-by-D numpy array, where N is the 
    number of training points and D is the dimensionality of features
    - w: weights of the trained multinomial classifier, C-by-D 
    - b: bias terms of the trained multinomial classifier, length of C
    
    Returns:
    - preds: N dimensional vector of multiclass predictions.
    Outputted predictions should be from {0, C - 1}, where
    C is the number of classes
    """
    N, D = X.shape
    ############################################
    # TODO 8 : Edit this part to               #
    #          Compute preds                   #
    z=(w.dot(X.T)).T + b
    classes=list()
    for i in z:
        classes.append(np.argmax(i))
    preds=np.array(classes)
    
    #preds = softmax((w.dot(X.T)).T + b)
    #preds = np.argmax(preds, axis=1)
    ############################################

    assert preds.shape == (N,)
    return preds


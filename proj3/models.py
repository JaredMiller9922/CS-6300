import nn

class PerceptronModel(object):
    def __init__(self, dimensions):
        """
        Initialize a new Perceptron instance.

        A perceptron classifies data points as either belonging to a particular
        class (+1) or not (-1). `dimensions` is the dimensionality of the data.
        For example, dimensions=2 would mean that the perceptron must classify
        2D points.
        """
        self.w = nn.Parameter(1, dimensions)

    def get_weights(self):
        """
        Return a Parameter instance with the current weights of the perceptron.
        """
        return self.w

    def run(self, x):
        """
        Calculates the score assigned by the perceptron to a data point x.

        Inputs:
            x: a node with shape (1 x dimensions)
        Returns: a node containing a single number (the score)
        """
        "*** YOUR CODE HERE ***"
        return nn.DotProduct(x, self.get_weights())

    def get_prediction(self, x):
        """
        Calculates the predicted class for a single data point `x`.

        Returns: 1 or -1
        """
        "*** YOUR CODE HERE ***"
        dP = self.run(x)
        dPS = nn.as_scalar(dP)
        return 1 if dPS >= 0 else -1

    def train(self, dataset):
        """
        Train the perceptron until convergence.
        """
        "*** YOUR CODE HERE ***"
        batch_size = 1
        accurate = False
        # Loop until we are 100 percent accurate
        while (accurate == False) : 
            accurate = True
            # Update any missclassified values from the dataset
            for x,y in dataset.iterate_once(batch_size) : 
                predictedLabel = self.get_prediction(x)
                actualLabel = nn.as_scalar(y)
                if predictedLabel != actualLabel :
                    # We predicted incorrectly so we need to update weights
                    accurate = False
                    # Preform the perceptron update rule w' = w + yx. Scale x by the true value.
                    self.w.update(x, actualLabel)

class RegressionModel(object):
    """
    A neural network model for approximating a function that maps from real
    numbers to real numbers. The network should be sufficiently large to be able
    to approximate sin(x) on the interval [-2pi, 2pi] to reasonable precision.
    """
    def __init__(self):
        # Initialize your model parameters here
        "*** YOUR CODE HERE ***"
        # Layer 1
        hLNeurons = 250 
        self.hLM = nn.Parameter(1, hLNeurons)
        self.hLB = nn.Parameter(1, hLNeurons)

        # Layer 2 (After ReLU)
        self.outM = nn.Parameter(hLNeurons, 1)
        self.outB = nn.Parameter(1, 1)

    def run(self, x):
        """
        Runs the model for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
        Returns:
            A node with shape (batch_size x 1) containing predicted y-values
        """
        "*** YOUR CODE HERE ***"
        # Our run method has to go through all the portions of the layer
        # Layer 1
        Layer1 = nn.Linear(x, self.hLM)
        Layer1wBias = nn.AddBias(Layer1, self.hLB)
        # Relu Layer
        ReluLayer = nn.ReLU(Layer1wBias)
        # Output Layer
        OutputLayer = nn.Linear(ReluLayer, self.outM)
        OutputLayerwBias = nn.AddBias(OutputLayer, self.outB)
        return OutputLayerwBias

    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
            y: a node with shape (batch_size x 1), containing the true y-values
                to be used for training
        Returns: a loss node
        """
        "*** YOUR CODE HERE ***"
        prediction = self.run(x)
        return nn.SquareLoss(prediction, y)

    def train(self, dataset):
        """
        Trains the model.
        """
        "*** YOUR CODE HERE ***"
        # The goal of our training should be to minimize our loss 
        batch_size = 50
        learningRate = 0.1
        maxLoss = float('inf')
        threshold = 0.02
        while (maxLoss > threshold) :
            maxLoss = 0
            # Calculate our loss 
            for x,y in dataset.iterate_once(batch_size) : 
                loss = self.get_loss(x, y) 
                batch_loss = nn.as_scalar(loss)
                maxLoss = max(batch_loss, maxLoss)
                grad_wrt_m, grad_wrt_b = nn.gradients(loss, [self.hLM, self.hLB])
                # Update our parameters make sure and update the negative! We go against the gradient
                self.hLM.update(grad_wrt_m, -learningRate)
                self.hLB.update(grad_wrt_b, -learningRate)

# TODO: As needed we could easily add another hidden layer
class DigitClassificationModel(object):
    """
    A model for handwritten digit classification using the MNIST dataset.

    Each handwritten digit is a 28x28 pixel grayscale image, which is flattened
    into a 784-dimensional vector for the purposes of this model. Each entry in
    the vector is a floating point number between 0 and 1.

    The goal is to sort each digit into one of 10 classes (number 0 through 9).

    (See RegressionModel for more information about the APIs of different
    methods here. We recommend that you implement the RegressionModel before
    working on this part of the project.)
    """
    def __init__(self):
        # Initialize your model parameters here
        "*** YOUR CODE HERE ***"
        # Layer 1
        hLNeurons = 350
        inputNeurons = 784
        self.hLM = nn.Parameter(inputNeurons, hLNeurons)
        # self.hLB = nn.Parameter(inputNeurons, hLNeurons)
        self.hLB = nn.Parameter(1, hLNeurons)

        # Layer 2 (After ReLU)
        outLNeurons = 10
        self.outM = nn.Parameter(hLNeurons, outLNeurons)
        self.outB = nn.Parameter(1, 10)


    def run(self, x):
        """
        Runs the model for a batch of examples.

        Your model should predict a node with shape (batch_size x 10),
        containing scores. Higher scores correspond to greater probability of
        the image belonging to a particular class.

        Inputs:
            x: a node with shape (batch_size x 784)
        Output:
            A node with shape (batch_size x 10) containing predicted scores
                (also called logits)
        """
        "*** YOUR CODE HERE ***"
        # Our run method has to go through all the portions of the layer
        # Layer 1
        Layer1 = nn.Linear(x, self.hLM)
        Layer1wBias = nn.AddBias(Layer1, self.hLB)
        # Relu Layer
        ReluLayer = nn.ReLU(Layer1wBias)
        # Output Layer
        OutputLayer = nn.Linear(ReluLayer, self.outM)
        OutputLayerwBias = nn.AddBias(OutputLayer, self.outB)
        return OutputLayerwBias

    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        The correct labels `y` are represented as a node with shape
        (batch_size x 10). Each row is a one-hot vector encoding the correct
        digit class (0-9).

        Inputs:
            x: a node with shape (batch_size x 784)
            y: a node with shape (batch_size x 10)
        Returns: a loss node
        """
        "*** YOUR CODE HERE ***"
        predictions = self.run(x)
        return nn.SoftmaxLoss(predictions, y)

    def train(self, dataset):
        """
        Trains the model.
        """
        "*** YOUR CODE HERE ***"
        batch_size = 100 
        learningRate = 0.8
        while (dataset.get_validation_accuracy() < 0.9805) :
            # Calculate our loss 
            for x,y in dataset.iterate_once(batch_size) : 
                loss = self.get_loss(x, y) 
                grad_wrt_m, grad_wrt_b = nn.gradients(loss, [self.hLM, self.hLB])
                # Update our parameters make sure and update the negative! We go against the gradient
                self.hLM.update(grad_wrt_m, -learningRate)
                self.hLB.update(grad_wrt_b, -learningRate)

class LanguageIDModel(object):
    """
    A model for language identification at a single-word granularity.

    (See RegressionModel for more information about the APIs of different
    methods here. We recommend that you implement the RegressionModel before
    working on this part of the project.)
    """
    def __init__(self):
        # Our dataset contains words from five different languages, and the
        # combined alphabets of the five languages contain a total of 47 unique
        # characters.
        # You can refer to self.num_chars or len(self.languages) in your code
        self.num_chars = 47
        self.languages = ["English", "Spanish", "Finnish", "Dutch", "Polish"]

        # Initialize your model parameters here
        "*** YOUR CODE HERE ***"
        # HL1
        input_features = 47 # This is because xs is batch_size x 47. The input has 47 features because of the 1 hot encoding 
        hidden_layer_neurons = 300 
        self.batch_size = 100 
        self.learningRate = 0.1 
        self.w1 = nn.Parameter(input_features, hidden_layer_neurons) 
        self.b1 = nn.Parameter(1, hidden_layer_neurons)

        # HL2 (After ReLU)
        output_layer_neurons = 5
        self.w2 = nn.Parameter(hidden_layer_neurons, output_layer_neurons)
        self.b2 = nn.Parameter(1, output_layer_neurons)

        # For RNN 
        self.w_hidden = nn.Parameter(hidden_layer_neurons, hidden_layer_neurons)

    def run(self, xs):
        """
        Runs the model for a batch of examples.

        Although words have different lengths, our data processing guarantees
        that within a single batch, all words will be of the same length (L).

        Here `xs` will be a list of length L. Each element of `xs` will be a
        node with shape (batch_size x self.num_chars), where every row in the
        array is a one-hot vector encoding of a character. For example, if we
        have a batch of 8 three-letter words where the last word is "cat", then
        xs[1] will be a node that contains a 1 at position (7, 0). Here the
        index 7 reflects the fact that "cat" is the last word in the batch, and
        the index 0 reflects the fact that the letter "a" is the inital (0th)
        letter of our combined alphabet for this task.

        Your model should use a Recurrent Neural Network to summarize the list
        `xs` into a single node of shape (batch_size x hidden_size), for your
        choice of hidden_size. It should then calculate a node of shape
        (batch_size x 5) containing scores, where higher scores correspond to
        greater probability of the word originating from a particular language.

        Inputs:
            xs: a list with L elements (one per character), where each element
                is a node with shape (batch_size x self.num_chars)
        Returns:
            A node with shape (batch_size x 5) containing predicted scores
                (also called logits)
        """
        "*** YOUR CODE HERE ***"
        # run f_init
        hidden_i = self.run_init(xs[0])

        # run f for all inputs len(xs) - 1
        for x in xs[1:] :
            #Layer1 = nn.Linear(x_init, self.w1)
            # Layer_i = nn.Add(nn.Linear(x, self.w1), nn.Linear(hidden_i, self.w_hidden)) 
            lin1 = nn.Linear(x, self.w1)
            lin2 = nn.Linear(hidden_i, self.w_hidden)
            Layer_i = nn.Add(lin1, lin2)

            Layer1wBias = nn.AddBias(Layer_i, self.b1)
            # Relu Layer
            hidden_i = nn.ReLU(Layer1wBias)

        # Output Layer
        OutputLayer = nn.Linear(hidden_i, self.w2)
        OutputLayerwBias = nn.AddBias(OutputLayer, self.b2)
        return OutputLayerwBias 

    def run_init(self, x_init) :
        # Run a regular network
        # Our run method has to go through all the portions of the layer
        # Layer 1
        Layer1 = nn.Linear(x_init, self.w1)
        Layer1wBias = nn.AddBias(Layer1, self.b1)
        # Relu Layer
        ReluLayer = nn.ReLU(Layer1wBias)
        # We return the Relu layer so it can be used in later calculations
        return ReluLayer 

        # print("size of ReluLayer")
        # print(ReluLayer)
        # # Output Layer
        # OutputLayer = nn.Linear(ReluLayer, self.w2)
        # OutputLayerwBias = nn.AddBias(OutputLayer, self.b2)
        # return OutputLayerwBias

    def get_loss(self, xs, y):
        """
        Computes the loss for a batch of examples.

        The correct labels `y` are represented as a node with shape
        (batch_size x 5). Each row is a one-hot vector encoding the correct
        language.

        Inputs:
            xs: a list with L elements (one per character), where each element
                is a node with shape (batch_size x self.num_chars)
            y: a node with shape (batch_size x 5)
        Returns: a loss node
        """
        "*** YOUR CODE HERE ***"
        prediction = self.run(xs)
        return nn.SoftmaxLoss(prediction, y)

    def train(self, dataset):
        """
        Trains the model.
        """
        "*** YOUR CODE HERE ***"
        batch_size = self.batch_size 
        learningRate = self.learningRate 
        while (dataset.get_validation_accuracy() < 0.87) :
            # Calculate our loss 
            for x,y in dataset.iterate_once(batch_size) : 
                loss = self.get_loss(x, y) 
                # grad_wrt_m, grad_wrt_b = nn.gradients(loss, [self.hLM, self.hLB])
                grad_wrt_w1, grad_wrt_b1, grad_wrt_w2, grad_wrt_b2, grad_wrt_w_hidden = nn.gradients(loss, [self.w1, self.b1, self.w2, self.b2, self.w_hidden])
                # Update our parameters make sure and update the negative! We go against the gradient
                self.w1.update(grad_wrt_w1, -learningRate)
                self.b1.update(grad_wrt_b1, -learningRate)
                self.w2.update(grad_wrt_w2, -learningRate)
                self.b2.update(grad_wrt_b2, -learningRate)
                self.w_hidden.update(grad_wrt_w_hidden, -learningRate)

# Image_Recognition_For_Object_Detection

This code performs the following tasks:

1. Import necessary libraries and modules:
   - It imports TensorFlow and the required submodules for creating the neural network.
   - The `kerastuner` library is imported to perform hyperparameter tuning using random search.
   - The `matplotlib.pyplot` library is imported for plotting.

2. Load and preprocess the MNIST dataset:
   - The MNIST dataset is loaded using `mnist.load_data()` and divided into training and testing sets.
   - The pixel values of the images are normalized to the range [0, 1] by dividing by 255.0.
   - The data is reshaped to have a single channel (grayscale) for use with the CNN.

3. Define the CNN model function for hyperparameter tuning:
   - The `build_model` function defines the architecture of the CNN with tunable hyperparameters.
   - The function takes an instance of `kerastuner.HyperParameters` (`hp`) as input to define the tunable hyperparameters.
   - The model architecture includes two convolutional layers with varying numbers of filters, kernel size (3x3), activation function (ReLU), and batch normalization.
   - Two max-pooling layers are used to downsample the feature maps.
   - Three dropout layers with varying dropout rates are used for regularization.
   - The output layer consists of 10 units (corresponding to the digits 0 to 9) with softmax activation for classification.

4. Define and compile the model:
   - The model is defined using the `Sequential` API from Keras.
   - The model is compiled with the Adam optimizer, categorical cross-entropy loss, and accuracy metric.

5. Define callbacks for training:
   - Three callbacks are defined: `ModelCheckpoint`, `EarlyStopping`, and `ReduceLROnPlateau`.
   - The `ModelCheckpoint` saves the best model based on validation accuracy during training.
   - The `EarlyStopping` stops training if the validation loss does not improve for a certain number of epochs to prevent overfitting.
   - The `ReduceLROnPlateau` reduces the learning rate if the validation loss plateaus to aid convergence.

6. Hyperparameter optimization using Random Search:
   - The code sets up a `RandomSearch` tuner from `kerastuner` to perform hyperparameter optimization.
   - The `RandomSearch` tuner uses the `build_model` function and aims to maximize validation accuracy as the objective.
   - It tries different combinations of hyperparameters for a specified number of trials (`max_trials`).

7. Search for the best hyperparameters:
   - The tuner searches for the best hyperparameters using the training data and validation split.
   - It runs for 20 epochs (`epochs=20`) for each trial.

8. Get the best model and summary of search results:
   - The best model from the tuner's search is obtained using `tuner.get_best_models(num_models=1)`.
   - The summary of the tuner's search results is displayed using `tuner.results_summary()`.

9. Train the best model:
   - The best model is trained with the hyperparameters found by the tuner.
   - It uses the early stopping and learning rate reduction callbacks to prevent overfitting.

10. Evaluate the best model on the test set:
   - The best model's performance is evaluated on the test set to calculate the test accuracy.

11. Visualize filters learned by the first convolutional layer:
   - The code visualizes the filters learned by the first convolutional layer to gain insights into what features the model is detecting.

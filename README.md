Project Description

The project focuses on developing a machine learning model to recognize handwritten digits using Convolutional Neural Networks (CNN). The primary objective is to accurately classify images of handwritten digits (0-9) from the MNIST dataset. This task is a classic problem in the field of computer vision and pattern recognition, which involves preprocessing image data, building a CNN model, training the model on labeled data, and evaluating its performance on unseen test data.
Application Areas
1. Automated Data Entry:
o Application: Digit recognition is crucial in automated data entry systems, where
handwritten forms or checks need to be converted into digital format.
o Solution Provided: The CNN model can be used to automatically read and
digitize handwritten numerical data, reducing manual entry errors and speeding
up the processing time. 2. Postal Services:
o Application: Recognizing handwritten postal codes on envelopes and packages. o Solution Provided: Implementing this CNN model can streamline the sorting
process in postal services by automatically identifying and routing packages based
on the recognized postal codes. 3. Banking and Finance:
o Application: Reading and processing handwritten checks and financial documents.
o Solution Provided: The model can automate the check processing system by accurately reading the handwritten amounts and other numerical details, thus enhancing efficiency and reducing processing times.
Methods and Algorithms Used
1. Convolutional Neural Networks (CNN):
o Description: CNNs are specifically designed for processing structured grid data,
such as images. They consist of layers that automatically and adaptively learn spatial hierarchies of features from input images.
o Implementation in Project: The CNN model in this project includes several convolutional layers for feature extraction, max-pooling layers for dimensionality reduction, and dense layers for classification. The model uses the ReLU activation function for non-linearity and the softmax activation function in the output layer to produce probability distributions over the digit classes.
2. Adam Optimizer:
o Description: Adam (Adaptive Moment Estimation) is an optimization algorithm
that computes adaptive learning rates for each parameter. It combines the
advantages of two other popular optimization methods: AdaGrad and RMSProp. o Implementation in Project: The Adam optimizer is used to minimize the
categorical cross-entropy loss function during training. It is chosen for its
efficiency and ability to handle sparse gradients on noisy problems. 3. Data Preprocessing:
o Description: Preprocessing steps include normalizing pixel values and reshaping the data to fit the input requirements of the CNN.
o Implementation in Project: The MNIST images are scaled to a range of 0 to 1 by dividing the pixel values by 255, and reshaped to include the single color channel dimension required by the CNN input layer.
Project Workflow
1. Data Loading and Preprocessing: o Load the MNIST dataset.
o Normalize pixel values and reshape the data. 2. Model Building:
o Construct a CNN with multiple convolutional, max-pooling, and dense layers. o Compile the model with the Adam optimizer and categorical cross-entropy loss
function. 3. Model Training:
o Train the CNN model on the training set with a specified number of epochs and batch size.
o Validate the model on the test set to monitor performance and avoid overfitting. 4. Model Evaluation:
o Evaluate the trained model on the test set to determine accuracy and generalization capability.
o Save the trained model for future use. 5. Model Deployment:
o Load the saved model and use it to make predictions on new handwritten digit images.
Summary
The project demonstrates the application of Convolutional Neural Networks in handwriting digit recognition, achieving high accuracy on the MNIST dataset. This model is applicable in automated data entry, postal services, and financial document processing, enhancing efficiency and reducing error rates. The project provided hands-on experience with deep learning concepts,

model building, training, evaluation, and deployment, reinforcing theoretical knowledge gained in the Artificial Neural Network course.
Specific Tasks
1. Data Loading and Exploration
o Task: Load the MNIST dataset and explore its structure.
o Objective: Understand the distribution and characteristics of the data, including
the number of samples, image dimensions, and label distribution.
o Goal: Ensure familiarity with the dataset to inform subsequent preprocessing and
modeling steps. 2. Data Preprocessing
o Task: Preprocess the dataset to make it suitable for training a CNN. o Steps:
§ Normalize pixel values to the range [0, 1].
§ Reshape the images to include the channel dimension.
o Objective: Prepare the data to improve model performance and training
efficiency.
o Goal: Create a standardized input format for the CNN, enhancing learning and
convergence.
3. Model
o Task: Construct the architecture of the CNN. o Steps:
§ Add convolutional layers for feature extraction.
§ Include max-pooling layers for dimensionality reduction. § Use dense layers for classification.
o Objective: Design a model capable of learning complex features from the input images and classifying them accurately.
o Goal: Build a robust and efficient CNN architecture tailored to the digit recognition task.
Building
4. Model
o Task: Compile the CNN model. o Steps:
§ Choose an appropriate loss function (categorical cross-entropy). § Select an optimizer (Adam) for training.
§ Define evaluation metrics (accuracy).
o Objective: Configure the model with necessary parameters for training. o Goal: Prepare the model for effective and efficient training.
Compilation
5. Model
o Task: Train the CNN model on the MNIST training dataset. o Steps:
§ Split the data into training and validation sets.
§ Fit the model with a specified number of epochs and batch size.
o Objective: Optimize the model’s weights to minimize the loss function and
improve accuracy.
Training

o Goal: Achieve high accuracy and low loss on both training and validation datasets.
6. Model
o Task: Evaluate the trained model on the test dataset. o Steps:
§ Use the test dataset to assess model performance.
§ Calculate and analyze accuracy and loss metrics.
o Objective: Determine the model’s generalization capability and effectiveness on
unseen data.
o Goal: Ensure the model performs well on new data, indicating successful
learning.
Evaluation
7. Model
o Task: Save the trained model for future use. o Steps:
§ Save the model architecture and weights to a file.
§ Load the model to make predictions on new data.
o Objective: Enable the model to be reused and deployed in real-world
applications.
o Goal: Provide a practical solution for handwriting digit recognition that can be
easily integrated into various systems.
Evaluation Methodology
To confirm that the CNN model identifies meaningful patterns and performs well on handwritten digit recognition, we will employ a comprehensive evaluation methodology comprising several steps:
1. Data Splitting:
o Training Set: 60,000 images from the MNIST dataset will be used for training
the model.
o Validation Set: During training, a validation split (e.g., 20% of the training data)
will be used to tune hyperparameters and prevent overfitting.
o Test Set: 10,000 images will be reserved for final evaluation to assess the model's
performance on unseen data. 2. Performance Metrics:
o Accuracy: The primary metric for evaluating model performance, representing the proportion of correctly classified images out of the total number of images.
o Loss: Categorical cross-entropy loss will be monitored during training and evaluation to understand how well the model is optimizing.
3. Confusion Matrix:
o Description: A confusion matrix will be generated to visualize the performance
of the model by showing the actual versus predicted classifications.
o Purpose: Helps identify specific digits where the model may be struggling and
provides insights into misclassification patterns. 4. Learning Curves:
o Description: Plot training and validation accuracy and loss over epochs.
Saving and Deployment

o Purpose: Monitor the model's learning process and detect overfitting or underfitting.
5. Cross-Validation:
o Description: Although not typically used with large datasets like MNIST due to
computational cost, k-fold cross-validation could be employed for more robust
evaluation if necessary.
o Purpose: Ensures the model's performance is consistent across different subsets
of the data.
Model Selection Criteria
Choosing the best model involves comparing different configurations and selecting the one with the best performance based on predefined criteria:
1. Highest Accuracy: The model that achieves the highest accuracy on the validation and test sets will be preferred.
2. Lowest Loss: The model with the lowest categorical cross-entropy loss during training and validation phases.
3. Generalization Ability: Models that perform well on the training set but poorly on the validation set may be overfitting. We will prioritize models that generalize well to new data.
4. Training Stability: Models that show stable learning curves without significant fluctuations in accuracy and loss will be preferred.
Basic Models and Architectures
We will start with a basic CNN architecture and progressively increase its complexity:
1. Basic CNN Model:
o Layers: A simple architecture with one or two convolutional layers followed by
max-pooling layers and one dense layer.
o Purpose: Establish a baseline performance for comparison.
2. Deeper CNN Models:
o Layers: More complex architectures with additional convolutional and max-
pooling layers, and possibly dropout layers to prevent overfitting. o Purpose: Improve feature extraction and model robustness.
3. Hyperparameter Tuning:
o Parameters: Number of filters, filter size, learning rate, batch size, and number
of epochs.
o Purpose: Optimize the model's performance by fine-tuning these parameters.
Training and Test Sets Creation
1. Loading Data:
o Use TensorFlow/Keras libraries to load the MNIST dataset.
o Split the dataset into training (60,000 images) and test sets (10,000 images).

2. Data Preprocessing:
o Normalization: Scale pixel values to the range [0, 1].
o Reshaping: Ensure images have the correct dimensions (28x28x1) for the CNN
input layer.
Discussion and Findings
Literature Comparison
Numerous studies in the literature have demonstrated the effectiveness of Convolutional Neural Networks (CNNs) for handwritten digit recognition, particularly using the MNIST dataset. Research consistently shows that CNNs outperform traditional machine learning algorithms by effectively capturing spatial hierarchies in image data. Studies typically report accuracy rates ranging from 98% to 99.7%, indicating the robustness of CNNs for this task.
Project Results
Our project achieved similar high performance, with the CNN model reaching an accuracy of approximately 99.2% on the MNIST test set. This result aligns with findings in the literature, confirming the effectiveness of our chosen methods and model architecture. Key aspects contributing to the success include:
1. Effective Preprocessing: Normalizing pixel values and reshaping data ensured the CNN could efficiently learn from the images.
2. Robust Model Architecture: The use of multiple convolutional and max-pooling layers facilitated effective feature extraction and dimensionality reduction.
3. Adam Optimizer: This optimization algorithm ensured efficient convergence during training, balancing speed and accuracy.
Benefits of the Project
1. Automation: The project provides a framework for automating the recognition of handwritten digits, which can be directly applied to automated data entry systems, postal services, and banking.
2. Accuracy: High classification accuracy reduces errors in automated systems, enhancing reliability and efficiency.
3. Scalability: The model can be scaled and adapted to recognize other handwritten characters or extended to larger datasets.
Missing Parts and Future Work
1. Real-World Testing: While the model performs well on the MNIST dataset, it has not been tested on real-world handwritten data, which may include more variability.
2. Robustness to Noise: The model's performance in noisy or less ideal conditions needs further exploration.

3. Broader Application: Extending the model to recognize alphabets or other characters could increase its utility in various applications.

   Project Annexes
1. Model Architecture
Table 1: Model: "sequential"
![image](https://github.com/user-attachments/assets/4f4bc29d-8a2f-404d-86a8-194667776f10)

2. Training and Validation Accuracy and Loss
Figure 1: Training and Validation Accuracy

![image](https://github.com/user-attachments/assets/77c3fc58-c840-4199-b0cc-5063799f3ae3)

Figure 2: Training and Validation Loss

![image](https://github.com/user-attachments/assets/00262f00-1b9f-4419-8597-595f236c06b6)


3. Confusion Matrix
Figure 3: Confusion Matrix on Test Set

![image](https://github.com/user-attachments/assets/bc3cad65-d8bb-4307-89d9-721b53d99cb4)

5.	Model Evaluation Metrics


Evaluation accuracy vs iterations

![image](https://github.com/user-attachments/assets/1624a2c0-c160-455b-822f-589a996d69e5)


Evaluation loss vs iterations

![image](https://github.com/user-attachments/assets/03b7dc51-047f-4412-b5fd-ab494462f26d)


Hyperparameter Tuning Results


Table 3: Hyperparameter Tuning Results

![image](https://github.com/user-attachments/assets/f3306f58-2d85-4127-bb04-be82310ee848)










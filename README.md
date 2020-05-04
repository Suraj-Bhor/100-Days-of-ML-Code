# 100-Days-of-ML-Code
100 Days of ML Code is a commitment to better my understanding of Machine Learning by dedicating at least 1 hour of my time everyday to studying and/or coding machine learning for 100 days.

## Day 1: 9<sup>th</sup> April 2020 : Deep Neural Networks
* Watched all videos of week 4 of [Neural Networks and Deep Learning course](https://www.coursera.org/learn/neural-networks-deep-learning/home/welcome) by Prof. Andrew Ng. 
* Learnt about the forward and backward propagation along with its vectorized implementations for deep neural nets.
* Learnt about the hyperparameter tuning for complex neural nets.

## Day 2: 10<sup>th</sup> April 2020 : Implementation of deep neural net
* Completed the two programming exercises of [Neural Networks and Deep Learning course](https://www.coursera.org/learn/neural-networks-deep-learning/home/welcome).
* Implemented a 2-layer and 4-layer fully connected feed forward neural network for classifying cats vs non-cats images.
* Learnt about the vectorized implementation of entire neural net model.
* Earned a [certificate](https://www.coursera.org/account/accomplishments/records/ZS8W2LCVSHM5) for completing the course.

## Day 3: 11<sup>th</sup> April 2020 : Regularization and hyperparameter tuning.
* Started with the [Improving Deep Neural Networks: Hyperparameter tuning, Regularization and Optimization](https://www.coursera.org/learn/deep-neural-network/home/welcome) course taught by Prof. Andrew Ng.
* Went through the week 1 videos and learnt:
  * Splitting dataset into train, development/cross validation and test sets.
  * Bias/Variance and their tradeoff.
  * L2 regularization/Frobenius Norm and how does regularization help in reducing high variance.
  * Dropout regularization (Inverted dropout technique).
  * Data Augmentation, early stopping and other regularization techniques.
  * Also learnt about why need to normalize training sets.
* Also to better understand maths for deep learning, revised the concepts of linear algebra by watching videos of 3 Blue 1 Brown's [Essence of linear algebra](https://www.youtube.com/watch?v=fNk_zzaMoSs&list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab) playlist.
* The videos give fantastic intuition about the vectors and matrix transformations and are illustrated beautifully.

## Day 4: 12<sup>th</sup> April 2020 : Implementation of different types of regularization
* Implemented different types of weights initialization methods like zero initialization, random initialization and He initialization and its effects on binary classification.
* Implemented L2 regularization and dropout regularization from scratch and understood its uses for reducing overfitting.
* Understood the concept of gradient checking and implemented it to check whether the backpropagation is computing correct gradients.
* Went through the videos of 3 Blue 1 Brown's essence of linear algebra and got to know about inituition of determinants in matrices.

## Day 5: 13<sup>th</sup> April 2020 : Optimization Algorithms
* Watched the week 2 course videos and learnt about different optimization algorithms like:
  * Mini-batch gradient descent
  * Stochastic gradient descent
  * Gradient Descent Momentum 
  * RMSprop
  * Adam Optimization Algorithm
* Learnt about exponentially weighted averages along with bias correction.
* Understood how learning decay works and also learnt how problem of local optima, saddle point and problem of plateaus occur while training deep neural networks.

## Day 6: 14<sup>th</sup> April 2020 : Implementation of Optimization Algorithms
* Implemented Mini-batch Gradient Descent, Gradient Descent Momentum and Adam Optimization Algorithm from scratch.
* Also understood the difference it makes in choosing the right optimization algorithm.

## Day 7: 15<sup>th</sup> April 2020 : Batch Normalization and Softmax Classifier
* Went through the week 3 videos of course 2 of deep learning specialization.
* Learnt about hyperparameter tuning and appropriate sampling of hyperparameters.
* Understood the concept of Batch Normalization for speeding up the learning process and also learnt how to implement batch normalization for deep neural networks.
* Also got to know how to use batch norm at testing time.
* Learnt about softmax regression and understood the loss function for softmax classifier.
* Learnt about few TensorFlow functions for implementing forward and backward pass.

## Day 8: 16<sup>th</sup> April 2020 : Iplementation of hand signs classifier in TensorFlow.
* Learnt the basics of TensorFlowv1.0. e.g placeholders, constants, variables, sessions, operations like tf.add, tf.matmul, tf.nn etc.
* Implemented a neural network using TensorFlow for classifying hand signs with accuracy of 72%.
* Completed with the course 2 of deep learning specialization and earned a [certificate](https://www.coursera.org/account/accomplishments/records/6JGSS9EHMCTV).

## Day 9: 17<sup>th</sup> April 2020 : Structuring Machine Learning Projects
* Started with the [course 3](https://www.coursera.org/learn/machine-learning-projects/home/welcome) of Deep learning specialization.
* Finished with the week 1 videos and assignment and learnt about the following concepts.
  * Orthogonalization concept.
  * Single number evaluation metric and also about satisficing and optimizing metric.
  * Splitting of Train/Dev/Test sets.
  * Comparing human-level performance to the neural network performance.
  * Avodiable bias and measure of variance.
  * Surpassing human-level performance.
  * Improving model performance.
  
## Day 10: 18<sup>th</sup> April 2020 : Transfer Learning and Multi-task learning
* Finished with the Structuring Machine Learning Projects course.
* Went through week 2 videos and learnt about the following concepts.
  * Carrying out error analysis.
  * Cleaning up incorrectly labeled data.
  * Training and testing on different distributions.
  * Bias and variance with mis-matched data distribution.
  * Transfer Learning.
  * Multi-task learning.
  * End-to-end Deep Learning.
* Earned course [certificate](https://www.coursera.org/account/accomplishments/records/PJJ7B368V36R).

## Day 11: 19<sup>th</sup> April 2020 : Convolutional Neural Networks
* Started with the [course 4](https://www.coursera.org/learn/convolutional-neural-networks/home/welcome) (Convolutional Neural Networks) of the Deep Learning Specialization.
* Finished week 1 videos and learnt about the following concepts.
  * Edge detection using various filters on gray-scale as well as RGB images.
  * Concept of padding and why it is used.
  * Strided convolutions.
  * Convolutions over volumes i.e. on RGB images.
  * Deep convolutional neural network examples.
  * Parameter sharing and sparsity connections.
  
## Day 12: 20<sup>th</sup> April 2020 : Implementation of Convolutional Neural Networks
* Started with the implementation of CNN from scratch.
* Implemented the forward pass of the CNN.

## Day 13: 21<sup>st</sup> April 2020 : Implementation of CNN in TensorFlow
* Finished with the implementation of CNN from scratch.
* Implemented the pooling forward, backward propagration and pooling backward of the CNN from scratch.
* Implemented the classification of hand signs using CNN through the TensorFlow framework with accuracy of 80%.

## Day 14: 22<sup>nd</sup> April 2020 : Classic CNN Architectures
* Finished with the week 2 videos and assignment and learnt about the following concepts.
  * LeNet-5 architecture.
  * AlexNet architecture.
  * VGG-16 architecture.
  * Residual Networks and the intuition behind why resnets work.
  * Network in network and 1 x 1 convolutions.
  * Inception network architecture.
  * Transfer learning and Data augmentation.
  * Data vs Hand-engineering.
  * State of the Computer Vision.
  
## Day 15: 23<sup>rd</sup> April 2020 : Implementation of ResNet-50 in Keras with TensorFlow as backend.
* Implemented CNN in Keras with TensorFlow as backend to classify emotions of the person in the image.
* Implemented the ResNet-50 architecture with its basic identity blocks and convolutional blocks in Keras.
* Implemented the classification of hand signs using ResNet-50 through the Keras framework with accuracy of 86%.

## Day 16: 24<sup>th</sup> April 2020 : Object detection
* Finished with the week 3 videos and assignment and learnt about the following concepts.
  * Difference between image classification, image localization and detection.
  * Landmark detection.
  * Object detection and sliding window detection.
  * Convolutional implementation of sliding window.
  * YOLO Algorithm.
  * Concepts such as specifying bounding boxes, intersection over union and non-max suppression.
  * Anchor box algorithm.
  * Region proposals. R-CNN, Fast R-CNN, Faster R-CNN.
  
## Day 17: 25<sup>th</sup> April 2020 : Implementation of YOLOv2
* Implement YOLOv2 CNN for object detection, specifically to detect cars with bounding boxes on the road.
* Finished with the week 4 videos and learnt about the following concepts.
  * Face verfication vs Face Recognition
  * One Shot Learning
  * Siamese network for one shot learning.
  * Triplet loss function.
  * Face verification and binary classification.
  * Neural Style Transfer.
  * Content loss functio and style loss function.
  * 1D and 3D generalization of models.
  
  
## Day 18: 26<sup>th</sup> April 2020 : Face recognition, Neural Style transfer and RNNs
* Implemented CNN for face recognition and neural style transfer with the use of transfer learning in TensorFlow.
* Completed the course 4 of Deep Learning Specialization and earned [certificate](https://coursera.org/share/6b3ea17a7bc112558a35cdf1100c8705).
* Started with the [course 5 (Sequence models)](https://www.coursera.org/learn/nlp-sequence-models/home/welcome) of Deep Learning Specialization.
* Finished with the week 1 videos and learnt about the following concepts.
  * Recurrent Neural Networks (RNNs)
  * Forward and backward propagation in RNNs.
  * Different types of RNNs for different types of NLP applications.
  * Language modelling and sequence generation using RNNs.
  * Sampling novel sequences and character level language models.
  * Vanishing gradients with RNNs.
  * Gated Recurrent Unit (GRU)
  * Long Short Term Memory (LSTM) model
  * Bi-directional RNNs (BRNN)
  * Deep RNNs
* Started with the Google's [Data Engineering with Google Cloud Professional Certificate course](https://www.coursera.org/professional-certificates/gcp-data-engineering#courses) on Coursera.
* Started with course 1 [(Google Cloud Platform Big Data and Machine Learning Fundamentals)](https://www.coursera.org/learn/gcp-big-data-ml-fundamentals/home/welcome) of the specialization.
* Going through week 1 videos and got to know about the basics of creating VM and creating storage buckets in GCP.
* Attended the Google Developers online meetup and learnt what TensorFlow Lite is, what it's capable to do and also how developers can use it to deploy storage and memory efficient machine learning models on Andriod, iOS and embedded devices.

## Day 19: 27<sup>th</sup> April 2020 : Implementation of RNNs
* Implemented the forward pass and the backward pass of RNN from scratch.
* Implemented a character level language model to generate dinosaurs names from scratch using just numpy and functions from the above RNN implementation. (Tedious assignment but fun to watch the model generate some really cool names.)
* Generated really cool Jazz music using LSTMs with the help of Keras. (Here's the [link](https://drive.google.com/open?id=1GgQ30_ZPEWOoGivlwDlDoYz8Ps7jIODG) to the music generated.)
* In general, felt that the assignments were little difficult to implement. Also lot of stuff is still unknown and will probably need re-iteration to understand the nitty-gritty of the implementations.

## Day 20: 28<sup>th</sup> April 2020 : NLP and Word Embeddinngs
* Started with the week 2 of the Sequence models course from deeplearning.ai.
* Finished with the week 2 videos and learnt about the following concepts.
  * Word Embeddings
  * t-SNE algorithm
  * Transfer learning and word embeddings.
  * Properties of word embeddings.
  * Embedding matrix
  * Neural Language Model
  * Word2Vec algorithm
  * Negative sampling algorithm.
  * GloVe word vectors 
  * Sentiment classification using RNN and word embeddings.
  * Debiasing word embeddings.
  
## Day 21: 29<sup>th</sup> April 2020 : Implementation of Sentiment analysis using LSTMs and Word Embeddings
* Implemented a simple neural network to find cosine similarity between two words and also debiased the word embeddings as mentioned in this [paper](https://arxiv.org/abs/1607.06520).
* Implemented a LSTM model using pretrained word embeddings(GloVe word vectors) in Keras to associate emojis with the input text. In short it was a sentiment classifier with labels in the form of different emojis.
* Went throught the week 1 videos of [GCP Big Data and Machine learning course](https://www.coursera.org/learn/gcp-big-data-ml-fundamentals/home/welcome) on Coursera.
* Had a lab assignment where we had to explore a BigQuery Public Dataset and also upload our dataset into the BigQuery and create custom table from it and run queries on the tables.

## Day 22: 30<sup>th</sup> April 2020 : Attention Networks and Machine Translation
* Finished with the Sequence Models course and completed the Deep Learning Specialization. (Feels good!)
* Concepts learnt from the week 3 of sequence models
  * Sequence to sequence models
  * Image captioning using CNNs and encoder decoder networks.
  * Machine translation as building a conditional language model.
  * Encoder and decoder networks.
  * Beam search algorithm for machine translation applications.
  * Refinements to beam search algorithm - Length Normalization.
  * Error analysis for beam search.
  * Bleu score on n-grams.
  * Attention model
  * Speech recognition and CTC cost for speech recognition.
  * Trigger word detection.
* Implement a Attention model RNN where we had to output the date in machine understandable language from human given input date formats using LSTMs.
* Implement trigger word detection application using GRU.
* Earned the specialization [certificate](https://www.coursera.org/account/accomplishments/specialization/XQQJX2639GBG).

## Day 23: 1<sup>st</sup> May 2020 : Big data and Machine learning in GCP
* Went through the week 1 videos of Big data and Machine Learning in GCP.
* Learnt how to choose right approach for problems.
* Explored a real customer solution architectures in machine learning.
* Learnt what are key roles in a data driven organization.

## Day 24: 2<sup>nd</sup> May 2020 : Recommend products using Cloud SQL and Apache SparkML
* Implemented a recommendation system in GCP Qwiklabs using CloudSQL and Apache SparkML model to predict the recommendations for a user based on his activity.
* Got familiar with BigQuery and explored couple of large dataset with SQL. Also learnt how to injest data into the BigQuery.
* Also used BigQuery GIS to analyse geographic data.
* Learnt about BigQuery ML to build a classification system to predict whether the user will purchase an item from the ecommerce website based on his session data. Also used a logistic regression model written in BigQuery ML SQL to train, evaluate and predict the purchases of users.
* Learnt about the modern data pipeline challenges.
* Learnt briefly about Message-oriented architectures with Cloud Pub/Sub and design and implement streaming data pipelines that scale.
* Also visualized streamlined data on the Data Studio.
*  Created a streaming data pipeline with Cloud Dataflow for real-time New York Taxi Service.

## Day 25: 3<sup>rd</sup> May 2020 : Using AutoML to classify cloud types images.
* Finished with the week 2 videos of the Big Data and Machine Learning in GCP.
* Learnt how to use various tools in GCP (BigQueryML, AutoML, Cloud ML APIs) to perform different tasks on the unstructed data.
* Used the BigQueryML to make a multi-class logistic regression model to classify images of different types of clouds. Accuracy achieved here was just 79%.
* To improve accuracy used the GCP's AutoML to figure out on its own the model to the train images on and then test out our images to classify the cloud types.
* The cloud AutoML is really cool and under the hood it trains our images on variety on networks (from Feed forward neural networks to complex CNN architectures). After training it evaluates its accuracy against different models and picks and returns the model which gives the highest accuracy. It takes a while for training but the output is always the best model.
* Trianed the cloud types model on GCP with the validation accuracy of 97% (Pretty Good!).
* Finished with the course and earned the [certificate](https://www.coursera.org/account/accomplishments/certificate/DEMRZV4GEAWV).
* Started with the reading of research papers. Picked the AlexNet paper first (since its easy to read.) and will try to re-implement it.
* Taking notes of the papers in Notion.

## Day 26: 4<sup>th</sup> May 2020 : Data Engineering 
* Started with the [course 2](https://www.coursera.org/learn/data-lakes-data-warehouses-gcp/home/welcome) (Modernizing Data Lakes and Data Warehouses in GCP) of Data Engineering with GCP Professional Certificate.
* Got a high level overview of what data engineering is and how data engineer builds data pipelines to faciliate easy flow of data into the entire system.
* Understood what are data lakes and data warehouses and what are the considerations to be taken into account while building one.
* Got to know what all challenges data engineer have to face and how they can be tackled using some tools available in GCP.
* Learnt what ETL pipelines are and how they are used to injest data into the data warehouses.
* Understood the difference between the transactional databases and the data warehouses.
* Learnt how the data engineers have to deal with demands of ML engineers, Data analysts and other co data engineers.
* Got to know how data governance can be done to hide sensitive information using the cloud data loss prevention API.
* Finally, learnt how to productionize the data process using Apache AirFlow.

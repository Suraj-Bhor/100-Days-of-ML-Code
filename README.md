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

## Day 27: 5<sup>th</sup> May 2020 : Building data lakes.
* Understood what data lakes are and what are the consideration to be taken into account while building one.
* Learnt the differences between data lake and data warehouse.
* Went through different storage options in GCP.
* Understood the high-level architecture of Google Cloud Storage and how it stores data in the form of buckets and objects.
* Also understood how access is controlled in such storage devices with IAM and access lists.
* Went through the difference between Transactional systems and Analytical systems.
* OLTP vs OLAP.
* Got to know what is fully managed services and serverless services.
* Did some hands-on in the GCP lab where we had to injest data from CSV into the Google Cloud SQL where they can be explored using normal SQL commands. This Cloud SQL here acts as a data lake and from here data can be passed to data warehouses such as Google BigQuery for analytics purposes.
* Continued reading the AlexNet paper and updated the notes in Notion.
* Learnt what are key roles in a data driven organization.

## Day 28: 6<sup>th</sup> May 2020 : Building data warehouses
* Went through the week 2 videos of modernizing the data lakes and data warehouses.
* Learnt what a modern data warehouse should offer in terms of features such as capacity, serverless capabilities, security, machine learning, data visualization tools support etc.
* Understood at a high-level of how BigQuery stores its data into datasets containing tables.
* Learnt how the BigQuery Data Transfer services brings in data from different resources with its connectors connecting BigQuery to Amazon S3 storage, YouTube, Google Ads etc.
* Differences in EL, ELT and ETL.
* Performed a lab demo where we had to import data from csv into the BigQuery and then use SQL to find insights into the data.
* Read further the AlexNet paper and understood its architecture and the number of methods used to avoid overfitting of data.
* All progress stored in Notion.

## Day 29: 7<sup>th</sup> May 2020 : Structs and arrays in BigQuery and AlexNet Implementation
* Finished with the week 2 videos of modernizing data lakes and data warehouses in GCP.
* Learnt how denormalizing is done in BigQuery to make queries faster and reduce the overhead of using expensive JOIN operations.
* Went the through the concepts of nested and repeated fields in BigQuery.
* Learnt about structs and arrays in BigQuery and also performed lab assignment to get hands-on with these concepts.
* Learnt how partitioning and clustering is done on the database tables in BigQuery.
* Finished with the course and earned [certificate](https://www.coursera.org/account/accomplishments/certificate/5J8DV3G8HT75).
* Finished reading the AlexNet paper and now trying to implement it using the Flower Recognition dataset provided by Kaggle.


## Day 30: 8<sup>th</sup> May 2020 : Setting up the environment
* Just worked on setting up the environment for implementing the AlexNet paper on Flower Recognition challenge.
* Had issues with various packages in conda took a lot of time to resolve them. Still some visualization packages dependencies is yet to be resolved.

## Day 31: 9<sup>th</sup> May 2020 : Implementation of AlexNet
* Implemented AlexNet Paper using the Flower Recognition dataset in Kaggle.
* Got familiar with Kaggle notebooks, setup environment and the training data.
* Implemented the CNN using the AlexNet architecture in Keras and ran the code for 15 epochs.
* Accuracy wasn't quite good hence tried using the pretrained weights for the network.
* There were improvements in training accuracy but validation accuracy was not increasing.
* Played around with hyperparameter tuning and using various optimizers.
* Finally used the VGG16 pretrained model to do the transfer learning task.

## Day 32: 10<sup>th</sup> May 2020 : Building batch data pipelines.
* Started with the course 3 i.e [Building batch data pipelines on GCP](https://www.coursera.org/learn/batch-data-pipelines-gcp/home/welcome) of the Data Engineer Specialization on GCP.
* Revised the concepts of EL, ETL and ELT. Understood each architecture and the scenarios where they have to be used.
* Quality consideration in the ETL pipelines.
* Understood how ETL pipelines in Dataflow work to land data into the data warehouses.
* Learnt how different tools are used according to the use cases like uses of Cloud Dataflow ,Cloud Dataproc and Cloud Data Fusion in GCP.
* Had a high level overview of the Hadoop Ecosystem and understood the limitation the on-premise Hadoop Clusters have to face.
* Understood how the GCP Dataproc help in processing Big Data workloads through fully managed Hadoop clusters.
* Had a overview of the Cloud Dataproc architecture in GCP.
* Understood how GCS helps in decoupling the storage from compute in case of Hadoop Clusters where the data is saved in storage buckets rather than on the cluster attached HDFS.

## Day 33: 11<sup>th</sup> May 2020 : Apache Spark Jobs on Cloud Dataproc
* Finished with the week 1 of course 3 i.e [Building batch data pipelines on GCP](https://www.coursera.org/learn/batch-data-pipelines-gcp/home/welcome) of the Data Engineer Specialization on GCP.
* Learnt how to optimize Dataproc in terms of Hadoop and Spark performance for all cluster architectures.
* Learnt how to optimize storage on Cloud Dataproc and when do we really need HDFS for storing the data in the cluster.
* Also understood the need to separate cluster compute resources and data storage.
* Had a brief overview of the lifecycle of cluster in Dataproc.
* Understood the concepts of optimizing the Dataproc templates and learnt how the autoscaling of cluster works depending upon the YARN tool.
* Learnt how to optimize the Dataproc monitoring by setting up the log levels and visualizing this data into the Cloud Stackdriver.
* Finally had a hands-on in the GCP to perform the following tasks.
  * To run original Spark Code on Cloud Dataproc where the storage used is HDFS and the dataset used is that of the Knowledge, Discovery and Data competition.
  * Then instead of using the HDFS use the Google Cloud Storage Buckets to store the processed data.
  * Automated the above task so that it runs on job-specific clusters.
  
## Day 34: 12<sup>th</sup> May 2020 : Cloud Data Fusion
* Started with the week 2 of the building batch data pipelines in GCP course.
* Learnt how data fusion on GCP works and how it's intuitive to build data pipelines which handle complex workloads in Hadoop and Apache Spark.
* Had a overview of the Cloud Composer in GCP and how it leverages the open source Apache AirFlow to orchestrate the various GCP services.
* Had a hands-on on GCP where the following tasks were completed.
  * Created the data fusion instance and connected various data sources such as GCS and Google BigQuery to pore in the data.
  * Used the Wrangler tool to transform and clean the data which contained several mislead data and feed this data to the Dataproc instance using just the Directed Acyclic Graphs in the Data Fusion Studio.
  * Created a Data pipeline and joined the two data sources using the Joiner module to make a simple join operation.
  * Later ran this job which internally spun up the dataproc instance along with the Hadoop cluster and ran the SQL query across the cluster and later saved the output in a table in BigQuery.
  
## Day 35: 13<sup>th</sup> May 2020 : Cloud Composer and Cloud Data Fusion
* Went into the details of Cloud Composer in GCP.
* Understood how the workflow scheduling works along with modules such as monitoring and logging.
* Had a hands-on lab in GCP where I had to perform the following tasks:
  * Creating the Cloud Composer Environment in GCP.
  * Defining the workflow/pipeline like the source, the sink etc.
  * Viewing Airflow UI and setting the Airflow variables.
  * Uploading the DAG to GCS.
  * Running simple word count algorithm on a Dataproc Hadoop cluster and saving the output to a GCS bucket.
* Also learnt about the Cloud Dataflow service offered by GCP.
* Understood the difference between Cloud Dataflow and Cloud Dataproc and when to use them.
* Understood how Cloud Dataflow works internally and how it leverages the open source Apache BEAM tool to make the data pipelines easier to build for both batch and streaming data.
* Understood the architecture of Apache BEAM and its PTransform and Pcollections components. Went into the details of each component.
* Later, had a hands-on on in the GCP lab where we needed to create simple data pipeline using the Cloud Dataflow in Python.

## Day 36: 14<sup>th</sup> May 2020 : Cloud Dataflow
* Finished with the course 3 of Data Engineering Specialization in GCP.
* Learnt about agrregating with GroupByKey and Combine in Dataflow and saw how data skewness can affect the performance of grouping operation at scale.
* Understood how to use the Cloud Dataflow templates and also save our Dataflow instance as a template so that it can be easily spun up again.
* Understood how to use the Dataflow SQL to make queries inside the data pipelines.
* Had hands-on lab to do serverless data analysis with Dataflow, here used the mapreduce function to do the analysis over large dataset.
* Had hands-on lab to better understand the usage of side inputs in Dataflow in GCP.
* Finished with the course and earned a [certificate](https://www.coursera.org/account/accomplishments/certificate/YEL3JBESUEGK).
* Started with the course 4 of the specialization. ([Building Resilient Streaming Analytics Systems on GCP](https://www.coursera.org/learn/streaming-analytics-systems-gcp/home/welcome))
* In week 1 learnt about the stream data processing in general and also understood now Cloud Pub/Sub is used to make the streaming process seamless.
* Also learnt about the push vs pull model of Cloud Pub/Sub and how the acknowledgements in each case are passed on to the events.
* Understood how to publish with Cloud Pub/Sub in GCP.

## Day 37: 15<sup>th</sup> May 2020 : Cloud Pub/Sub and stream processing with analytics on streaming data.
* Finished with week 1 of the course 4 of Data Engineering Specialization on GCP.
* Learnt how to make streaming pipelines in Cloud Dataflow and also understood what are the issues involved in processing streaming data.
* Also understood how modifications are done on the data from cloud Pub/Sub to make transformations using the PTransform.
* Had a hands-on lab in GCP which involved creating streaming pipelines using Cloud Pub/Sub and Cloud Dataflow.
* Understood the concept of Dataflow windowing and also learnt various types of Data Windowing techniques such as Fixed, Sliding and Session Windowing.
* Learnt about the Accumulation and its types.
* Had a hands-on lab in GCP where the following tasks were performed:
  * Launching Dataflow and a Dataflow job.
  * Understanding how data elements flow through the transformations of a Dataflow pipeline.
  * Connecting Dataflow to Pub/Sub and BigQuery.
  * Observing and understanding how Dataflow autoscaling adjusts compute resources to process input data optimally.
  * Learning where to find logging information created by Dataflow.
  * Exploring metrics and create alerts and dashboards with Cloud Monitoring.
* Started with the week 2 videos of Streaming into BigQuery.
* Also learnt how to visualize data into the Cloud Data Fusion.
* Had hands-on lab where the following tasks were performed:
  * Connecting to a BigQuery data source.
  * Creating reports and charts to visualize BigQuery data.
  
## Day 38: 16<sup>th</sup> May 2020 : Cloud Bigtable and Advanced BigQuery Functionalities
* Finished with the course 4 i.e Building Resilient Streaming Analytics Systems on GCP of the Data Engineering Specialization on GCP
* Key concepts learnt in the week 2 of the course:
  * How Cloud BigTable works and how it is different from BigQuery.
  * Learnt when to make use of Cloud Bigtable and BigQuery.
  * Had a overview of the architecture of Cloud Bigtable and also how data is stored in it.
  * Also understood concepts of column families and how to optimize Cloud Bigtable performance.
  * Had a hands-on lab for Cloud Bigtable where streaming data from cloud Dataflow was used to make entries in the Cloud Bigtable. Later queried the table to get the processed data.
  * Learnt about some advanced functionalities in BigQuery.
  * Understood how to store GIS data and also process it using the ST_DISTANCE type functions in BigQuery.
  * Later also saw how to use this data on the GeoViz tool for visualizing the GIS data on Google Maps.
  * Learnt about the analytical windows in BigQuery and how they are used to make queries faster rather than using joins to process any queries.
  * Understood how the RANK and the ARRAYs functions are used in BigQuery. These methods are used in conjuction with the Analytical Windows in BigQuery.
  * Learnt about partioning the table via different method which reduces the read time of the data and in turn save a lot of time in processing the queries on very large data sets.
  * Understood how the performance of the SQL queries can be improved by using some specific tips.
  * Lastly had hands-on on all the above topics in BigQuery.
* Finished with the course and earned [certificate](https://www.coursera.org/account/accomplishments/records/FWK59CDH8FHM).


## Day 39: 17<sup>th</sup> May 2020 : ML and AI on GCP 
* Started with the Course 5 ([Smart Analytics, Machine Learning, and AI on GCP](https://www.coursera.org/learn/smart-analytics-machine-learning-ai-gcp/home/welcome)) of the Data Engineering Specialization on GCP. 
* Went through the week 1 and week 2 videos and learnt the following concepts:
  * Basics of ML.
  * Different ML options in GCP.
  * Introduction to Analytics and AI.
  * Different ML APIs in GCP.
  * Learnt about how AI notebooks are used to write custom ML models.
  * Also learnt how BigQuery can be used in Jupyter notebooks.
  * Understood about the phases in the ML projects.
  * Understood what Kubeflow is and leanrt how to deploy machine learning models built in Tensorflow to train and serve using the Kubeflow platform.
  * Also saw how AI Hub has different prebuilt ML models and pre-configured Kubeflow pipelines so that we dont reinvent the wheel again to do a specific task.
  * Learnt how to productionize the ML models.
  * Got to know how to build custom models in BigQuery ML with models used for tasks such as Classification, Regression and Recommendations.
  * Also, saw how unsupervised learning models can be built in BigQuery where K-means clustering can be done effectively on the tables of the database.
  * Lastly learnt about the Cloud AutoML where it can used in applications like Vision, NLP and Tablular data analytics.
* Also, performed 5 labs in GCP to do the following tasks:
  * Using the Natural Language API to classify unstructured text.
  * Runnig BigQuery in Jupyter Labs on AI Platform.
  * Predicting Bike Trip Duration with a Regression Model in BQML.
  * Movie Recommendations in BigQuery ML.
  * Running AI models on Kubeflow.
* Finally, finished with the course and earned the [certificate](https://www.coursera.org/account/accomplishments/records/ZFPGRWCQHN9Q).

## Day 40: 18<sup>th</sup> May 2020 : Revisiting Pandas.
* Went through the basics of Pandas.
* Revised some concepts such as Series, Dataframes, Lambda functions, Indexing and reindexing, Histogram plotting and accessing and manipulating data in Pandas.
* Did some exercises in Google Colab.

## Day 41: 19<sup>th</sup> May 2020 : Modified AlexNet for Flower Recognition
* Solved the issues that were causing the model to perform very poorely.
* With the original AlexNet got a low validation accuracy of 60%.
* So changed the hyperparameter and the learning algorithm and the valdiation accuracy achieved was 71.6%.
* Plotted some graphs to know the how the training loss and validation loss decreased over epochs.
* Learnt the following concepts while implementation:
  * Difference between model.fit_generator, model.fit and model.train_on_batch functions in Keras.
  * Generator concept in Python.
  * The LabelEncoder module in Sklearn and the le.transform and le.inverse_transform in Keras.
  
## Day 42: 20<sup>th</sup> May 2020 : Original AlexNet paper implementation with LR Annealer and Early Stopping
* Made changes to the original AlexNet paper and tried to increase the accuracy.
* Used the LR annealer in Keras to make the learning rate smaller when the the gradient descent is closer to the global optimal.
* Also used early stopping to avoid overtraining of the training data.
* The results seen were quite surprising since the validation accuracy achieved was approx. 75% where first without annealer and early stopping the accuracy achieved was only 50%.
* For further increasing the accuracy used different optimizer and loss function but the accuracy achieved was just 70%. Issue was with this was the hyperparameter not correctly tuned and also the model had high bias. Will need to debug and try to increase the accuracy.
* Uploaded both the versions of the code in the DL-Paper-Implementations repository.

## Day 43: 21<sup>st</sup> May 2020 : Reading VGG16 paper
* Started reading the [VGG16](https://arxiv.org/pdf/1409.1556) paper and making notes in Notion.
* Wasn't familiar with the concept of 1 x 1 convolutions or Network in Network architecture.
* Read about it and also watched some videos about it.
* Signed up for the Jovian.ml course provided in collaboration with freeCodeCamp.

## Day 44: 22<sup>nd</sup> May 2020 : Dataproc and PySpark Jobs on the Hadoop Cluster
* Finished with the final course of the Data Engineering Specialization on GCP.
* Finished with the 2 labs in GCP where we have to create a Dataset and a table in BigQuery and populate data.
* Also, made queried to analyse data in BigQuery.
* Later in the second lab fired up a Dataproc cluster and ran PySpark jobs on them.
* Earned the following course [certificate](
https://www.coursera.org/account/accomplishments/certificate/68RWFYXRUZZ2).
* Earned the following Data Engineering with Google Cloud Specialization [certificate](https://www.coursera.org/account/accomplishments/specialization/certificate/5JG6UBFL66RF
).

## Day 45: 23<sup>rd</sup> May 2020 : Reading VGG16 paper - Part 2 and Basics of PyTorch.
* Finished reading the [VGG16](https://arxiv.org/pdf/1409.1556) paper and making notes in Notion.
* Understood how the scale jittering at test time increases the accuracy of the model.
* Also, leanrt about the concept of multi-scale evaluation and multi-crop evaluation and how do they help in increasing the validation accuracy.
* Also, got to know that the 1 x 1 convolutionals in a CNN to bring in the non-linearity may be a bad idea since a convolutional layer of some filter other than 1 x 1 conv. might help in finding out more spatial representations in the images and thus increasing accuracy of the model.
* Watched the first video of the Zero to GANs course offered free by Jovian.ml in collaboration with freeCodeCamp.
* Learnt the concepts of tensors basics in PyTorch and how do they calcluate the gradients by using functions such as tensor.backward().
* Also saw an example of linear regression in PyTorch where the basic implementation of Gradient Descent Algorithm was demonstrated using PyTorch.

## Day 46: 24<sup>th</sup> May 2020 : Implementation of Linear Regression using PyTorch
* Started with the week 1 assignment of the ZeroToGANs course.
* Firstly, got familiar with the basics of tensors and the number of operations we can do on them.
* Also, coded some examples where the torch.backward function computed differentiation automatically for a given function.
* Then, implemented a basic linear regression model in PyTorch to identify the yeild of apples and oranges based on the climatic conditions.

## Day 47: 25<sup>th</sup> May 2020 : Probability distribution functions
* As a part of the assignment in the ZeroToGANs course trying to come up with any useful 5 PyTorch tensor functions.
* To do this, I have decided to take the probablity distribution functions in PyTorch.
* So, learnt about the theory of probability distribution functions such as discrete uniform distribution, exponential distribution, Cauchy's distribution, Bernoulli's distribution etc.
* Also, side by side tried to use this functions and play a little bit around it.
* Finally, using this functions, plotted graphs to understand the data better.

## Day 48: 26<sup>th</sup> May 2020 : Probability distribution functions - Continued
* Continuing with the previous day task of learning about probability distribution functions, coding it up and visualizing it.

## Day 49: 27<sup>th</sup> May 2020 : Probability distribution functions - Continued
* Wasn't able to do much today due to some other work commitments. Still tried to do some learning about the probability functions. 
* Learnt about the Bernoulli distribution and how it used to define other distributions such as normal distributuon, binomial distribution etc

## Day 50: 28<sup>th</sup> May 2020 : Probability distribution functions - Visualizations
* It was all about random sampling of data from various distribution.
* Spent most of the time in reading about these distributions and trying them out in PyTorch.
* Also, after sampling data from various distributions tried plotting the histograms of the randomly sampled data along the curves of these distributions.

## Day 51: 29<sup>th</sup> May 2020 : PDF and Jovian.ml
* Done with updating the explanation notebook of random sampling of probability distribution functions in PyTorch.
* Following are the probability distribution functions covered:
  * Exponential Distribution
  * Continuous Normal Distribution
  * Continuous Log Normal Distribution
  * Continuous Uniform Distribution
  * Discrete Geometric Distribution
* Will surely write a blog post explaining the above points.
* Updated all the progress using the Jovian ML source code versioning system.
* You can see the notebook [here](https://jovian.ml/suraj-bhor/assignment-1-sampling-from-different-probability-distributions-copy1).

## Day 52: 30<sup>th</sup> May 2020 : Logistic Regression in PyTorch
* Watched the week 2 video of the ZeroToGANs course provided by Jovian.ml and freeCodeCamp.
* Understood how logistic regression model is trained to classify the hand written digits using the functions in PyTorch.

## Day 53: 31<sup>st</sup> May 2020 : VGG-16 paper implementation
* Started with the implementation of the VGG-16 paper using the German Traffic Sign Recognition Benchmark dataset in Kaggle.
* Finished with the data pre-processing step and made the training data usable for the image recognition task.
* Did all this using basic opencv and numpy arrays.

## Day 54: 1<sup>st</sup> June 2020 : VGG-16 paper implementation - Continued
* Continued on the implementation of the VGG paper in Keras.

## Day 55: 2<sup>nd</sup> June 2020 : VGG-16 paper implementation - Continued
* Continued on the implementation of the VGG paper in Keras.

## Day 56: 3<sup>rd</sup> June 2020 :Jovian ML week 2
* Did the assignment of week 2 of the freeCodeCamp and Jovian.ml course ZeroToGANs.
* Watched the videos of week 2 and understood various concepts and implementation of logistic regression in PyTorch.

## Day 57: 4<sup>th</sup> June 2020 :Jovian ML week 2 - Continued
* Learnt about the PyTorch functions to make a image classifier using the logistic regression.
* Refered this [notebook](https://jovian.ml/aakashns/03-logistic-regression).
* Understood the concept of logistic regression for the classification task.

## Day 58: 5<sup>th</sup> June 2020 : Insurance cost prediction using linear classifier in PyTorch
* Coded up a simple linear regression classifier in PyTorch for predicting the insurance cost based on the age, sex, BMI, No. of Children and smoking habit.
* Performed following steps:
  * Downloaded and explored the dataset. Dataset used from a past Kaggle Competition and did some EDA on it.
  * Prepared the dataset for training which included data preprocessing and loading the cleaned data in pandas dataframes.
  * Created a linear regression classifier in PyTorch.
  * Trained the model to fit the data.
  * Finally, made predictions on the trained model.
* Learnt how soft L1 loss does while calculating the loss.
* Here's the [notebook](https://jovian.ml/suraj-bhor/02-insurance-linear-regression) for reference.

## Day 59: 6<sup>th</sup> June 2020 : Insurance cost prediction using linear classifier in PyTorch. - Continued
* Had some issues regarding the accuracy of the model.
* Considered the output label in weights of the model. So was getting very high validation accuracy.
* Made the appropriate changes and commit them in jovian.

## Day 60: 7<sup>th</sup> June 2020 : VGG-16 paper implementation
* Continued the task of implmenting the VGG16 paper.

## Day 61: 8<sup>th</sup> June 2020 : VGG-16 paper implementation
* Continued the task of implmenting the VGG16 paper.

## Day 62: 9<sup>th</sup> June 2020 : German traffic sign recognition challenge
* Using the German traffic sign recognition challenge dataset to classify images.
* Pre-processing the data and trying to figure out ways to convert the images into the numpy array using both openCV and PIL.
* Plotting the images read using matplotlib.

## Day 63: 10<sup>th</sup> June 2020 : German traffic sign recognition challenge - Continued
* Trying the VGG network on this dataset.
* Created the model in Keras with Learning Annealer but something was wrong and the accuracy didn't go up from 6%.
* Need to debug what went wrong, maybe the image which is 30 x 30 is losing all its content due to the 5 max-pooling layers.

## Day 64: 11<sup>th</sup> June 2020 : German traffic sign recognition challenge - Finished
* Got the solution to the low accuracy.
* Issues was with the image size too small for the huge network that the image size became 1 x 1 at the last max pooling layer.
* Shortened the network and got validation accuracy to approx. 99% and test accuracy of 96%, which is pretty good.

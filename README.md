# objectrecHOG
  This is an image classifier using features from MatLab’s Computer Vision System Toolbox and Statistics and Machine Learning Toolbox to classify an image as pedestrian or non-pedestrian.  To use this code, ,these toolboxes will need to be imported. I created an initial object classification system using SVM and AdaBoost to classify images based on their HOG feature vectors into pedestrian or non-pedestrian categories. Since the feature vectors used to classify images can be significantly long, it proves efficient to compress these vectors for practical applications. We explore whether or not boosting algorithms can restore the performance of the image classifier after the feature vectors have been compressed.  
  
  Initially, I created code that reads in sets of testing and training images. These images feature either pedestrian or non-pedestrian objects from both the INIRIA Person database and the MIT Pedestrian Database. Non-pedestrian images included bikes and cars. Since the purpose of this code was to classify the entire image, I cropped the images so the main focus of each was either the pedestrian or non-pedestrian object, if needed. After the images are read in, each one is resized and preprocessed to remove noise artifacts. Separately for each set, HOG feature vectors are calculated and stored in a matrix and assigned either a pedestrian or non-pedestrian label. This matrix is used to train the AdaBoost and SVM classifiers. These classifiers then predict labels for the testing set. The SVM model has an accuracy of 88.42% and the AdaBoost model has an accuracy of 77.08%. 
  
  Next, I added code that uses random projection to compress the HOG feature vectors to see how the performance degrades. I predict the AdaBoost classifier will perform better than the SVM classifier since the classifiers used by SVM will become weaker as the vectors are further compressed.  The HOG features of both the testing and training set are compressed before they are used by the classifiers. Since random projection will produce a different compressed vector each time and, therefore, the classifiers will be trained on different features, the accuracy will vary with each run. For this reason, at each compression stage I used the average number of correct predictions across ten runs for comparison. The HOG feature vectors has an original length of 12,960. I compressed the vector at 1X (the same length but a different vector), 2X (length of 6,480), 4X (length of 3,240), 8X (length of 1,620), 16X (length of 810), 32X (length of 405), 64X (length of 202). 

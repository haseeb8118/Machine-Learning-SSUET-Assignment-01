ASSIGNMENT-01:
Question:

(i) How the ensemble leaning helpful for avoiding overfitting. Discuss with suitable example.

(ii) Implement and compare the performance of following algorithms for classification:
  (i) KNN
  (ii) Naïve Bayes
  (iii) Decision tree
  (iv) Random forest
  (v) Logistic regression

Download public dataset of heart disease from Kaggle, GitHub or UCI Machine Learning repository. Implement
and evaluate performance of each algorithm using accuracy, precision, recall, F1- score and AUC curve metrics.
Bonus: Implement a simple ensemble method (e.g. bagging or boosting) using one of the above algorithms and
evaluate its performance.

Note: You can use Python with scikit-learn and TensorFlow libraries to implement the algorithms and evaluate their performance.

-------------------------------------------------------------------------------------------------------------

ASSIGNMENT-02:
Question: Apply YOLO to the given image containing multiple objects. Use the following settings:
  1. Anchor points: 5
  2. Anchor boxes: 3
  3. NMS threshold: 0.5
  4. Confidence threshold: 0.8

Image: [Insert image with multiple objects, e.g., cars, pedestrians, bicycles]

Task:
  i. Preprocess the image and prepare it for YOLO input.
  ii. Apply YOLO to detect objects in the image.
  iii. Implement NMS to filter out duplicate detections.
  iv. Visualize the detection results, including bounding boxes and class labels.
  v. Calculate the mAP (mean Average Precision) for the detection results.

Requirements:
  i. Use a latest YOLO variant and implement it using a deep learning framework (e.g., PyTorch, TensorFlow).
  ii. Code should be written in Python.
  iii. Provide a detailed report explaining the steps taken, results, and discussions.

Grading criteria:
  - Accuracy of object detection
  - Effective use of NMS and anchor points
  - Code quality and organization
  - Clarity of report and results

------------------------------------------------------------------------------------------------------------

ASSIGNMENT-03: 
Question: 
Apply transfer learning algorithms to detect diseases like tuberculosis (TB) in X-Ray images and brain tumors in MRI images and analyze the results. Analyze all results

Dataset:
  - Chest X-Ray Images (TB)
  - Brain MRI (Tumor Segmentation)
  - (Dataset download: Kaggle, GitHub, UCI repository)

Specific Requirements:
  1. Data Preprocessing:
   - Split each dataset into training (80%) and validation sets (20%).
   - Normalize image pixel values between 0 and 1.
   - Resize images to 256x256 pixels.
  2. Transfer Learning:
   - Use each of the following pre-trained models as a starting point:
   - ResNet50
   - DenseNet121
   - InceptionV3
   - Fine-tune the models by adjusting the following hyperparameters:
   - Learning rate (0.001, 0.01, 0.1)
   - Batch size (8, 16,, 32)
   - Number of epochs (50)
  3. Performance Evaluation:
   - Calculate the following metrics for each model on the validation set:
   - Accuracy
   - Precision
   - Recall
   - F1-score
   - AUC-ROC
  4. Comparative Analysis:
   - Create a table comparing the performance metrics of each model for both datasets.
   - Plot the ROC curves for each model.
   - Discuss the strengths and weaknesses of each model and identify the best-performing model for each disease.

Deliverables:
  i. A Python script or Jupyter Notebook implementing the task.
  ii. A table comparing the performance metrics of each model for both datasets.
  iii. A plot (e.g., bar chart or ROC curve) visualizing the performance comparison.
  iv. A brief report (4-5 pages) discussing the results, strengths, and weaknesses of each model, and concluding with the best-performing model for each disease.

Evaluation Criteria:
  1. Accuracy and efficiency of the transfer learning approach.
  2. Effectiveness of hyperparameter tuning.
  3. Thoroughness of the comparative analysis.
  4. Clarity and organization of the deliverables.

Additional Requirements:
  - Use techniques like data augmentation, regularization, or ensemble methods to improve model performance.
  - Explore the use of transfer learning with different architectures
  - Discuss the clinical implications and potential applications of the best-performing models in medical diagnosis.

# On the Importance of Feature Representation for Flood Mapping using Classical Machine Learning Approaches

Climate change has increased the severity and frequency of weather disasters all around the world so that efforts to aid disaster management activities and recovery operations are of high value. Flood inundation mapping based on earth observation data can help in this context, by providing cheap and accurate maps depicting the area affected by a flood event to emergency-relief units in near-real-time. Modern deep neural network architectures require vast amounts of labeled data for training and whilst a large amount of unlabeled data is available, accurately labeling this data is a time-consuming and expensive task.

Building upon the recent development of the Sen1Floods11 dataset, which provides a limited amount of hand-labeled high-quality training data, this paper evaluates the potential of five traditional machine learning approaches such as gradient boosted decision trees, support vector machines or quadratic discriminant analysis for leveraging this data source. By performing a grid-search-based hyperparameter optimization on 23 feature spaces we can show that all considered classifiers are capable of outperforming the current state-of-the-art neural network-based approaches in terms of total IoU on their best-performing feature spaces, despite our approaches being trained only on the small amount of hand labeled optical and SAR data available in this dataset for performing pixel-wise flood inundation mapping. We show that with total and mean IoU values of 0.8751 and 0.7031 compared to 0.70 and 0.5873 as the previous best-reported results, a simple gradient boosting classifier can significantly improve over the current state-of-the-art neural network based approaches on the Sen1Floods11 test set.

Furthermore, an analysis of the regional distribution of the Sen1Floods11 dataset reveals a problem of spatial imbalance. We show that traditional machine learning models can learn this bias and argue that modified metric evaluations are required to counter artifacts due to spatial imbalance. Lastly, a qualitative analysis shows that this pixel-wise classifier provides highly-precise surface water classifications indicating that a good choice of a feature space and pixel-wise classification can generate high-quality flood maps using optical and SAR data. To facilitate future use of the created feature spaces and the gradient boosting model, we make our code publicly available in this repository.

## Results

You can find a pre-print of our work, which has been submitted to Remote Sensing of Enviornment, on [arXiv](https://arxiv.org/abs/2303.00691).

# Running the code 

Please refer to the [associated README](src/README.md).

# AdvancedBirdSpeciesClassification

## Overview
This project focuses on the classification of bird species using TensorFlow and the Xception architecture. It demonstrates advanced techniques in image 
classification through preprocessing of TFRecord datasets and effective use of transfer learning.

## Features
- Image preprocessing from TFRecord datasets.
- Transfer learning with Xception model.
- Fine-tuning for improved accuracy.
- Training and validation phases with performance evaluation.

## Requirements
- TensorFlow
- Pandas
- Matplotlib

## Usage Instructions
1) **Preprocessing**: Start by running `preprocessDefinition.py`. This script contains the function for parsing and preprocessing the image data from TFRecord datasets. 
Ensure you have the TFRecord files (`birds-vs-squirrels-train.tfrecords`, `birds-vs-squirrels-validation.tfrecords`, etc.) in the correct directory.
2) **Model Training**: Next, run `buildAndTrainModel.py`. This script utilizes the Xception model from TensorFlow for transfer learning and trains the model on the 
preprocessed data. The script also includes fine-tuning layers after the initial training.
3) **Evaluation and Visualization**: Finally, the training and validation accuracy and loss are plotted using Matplotlib within `buildAndTrainModel.py`. Monitor these 
plots to evaluate the model's performance.

Ensure that TensorFlow, Pandas, and Matplotlib libraries are installed before running these scripts.


## Contributing
Contributions to improve the model or add new features are welcome.


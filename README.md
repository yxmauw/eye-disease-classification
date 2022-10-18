# :eye: Eye disease classification
A multiclass problem <br>
Option 2 for GA Capstone 22-Oct-2022


## Vision loss: A Public Healthcare Problem
More than 3.4 million Americans aged 40 years and older are blind (having a visual acuity of 20/200 or less or a visual field on 20 degrees or less) or visually impaired (having a visual acuity of 20/40 or less). Other estimates of “vision problems” range as high as 21 million, and a total of 80 million Americans have potentially blinding eye diseases. The major causes of vision loss are cataracts, age-related macular degeneration, diabetic retinopathy, and glaucoma. ([CDC et al](https://www.cdc.gov/visionhealth/basic_information/vision_loss.htm))

In Singapore, diabetic retinopathy is the leading cause of vision loss in working-age adults. According to a study led by researchers at the Singapore Eye Research Institute (SERI) in 2015, diabetic retinopathy has to date claimed the sight of more than 600 Singaporeans, the loss of an eye in 8,000, and visual impairment in a further 17,500. The problem worsens as the risk of blindness increases fifteen-fold for Singaporeans aged 50 to 80 and above. As age steals away the senses, vision loss is perhaps the most devastating, as it increases the risk of falls, depression and even premature death. ([SNEC et al](https://www.snec.com.sg/giving/singapores-eye-health))


## Problem Statement

_Fictional scenario_
> Singapore Health Promotion Board is searching for innovative, cutting-edge technological solutions to facilitate mass eye screening of common eye diseases in the general adult population. They hope to automate interpretation of retina screening images, shorten the time taken to flag high-risk persons for further health assessment. This would allow early intervention, lower risk of disease progression and lower overall healthcare cost burden. By minimising number of people in the population with severe eye disease, we minimise the use of more costly therapies.  

> Using 4,217 retinal images collected from various sources like IDRiD, Oculur recognition, HRF etc, construct a machine learning algorithm to help classify a RGB retina image to any of 4 classes (cataract, diabetic retinopathy, glaucoma and normal).

_**Potential stakeholders:** public health authorities, eye clinics, neighbourhood eye screening centres e.g. optical shops_


## Data
| Eye disease          | No. of images |
|:---------------------|:-------------:|
| Cataract             |     1038      |
| Diabetic retinopathy |     1098      |
| Glaucoma             |     1007      |
| Normal               |     1074      |
| ***                  | ***           |
| **Total**            |   **4217**    |


## Directory tree
```
Repo
 ├── model_notebooks
 |      ├── EfficientNet.ipynb
 |      └── InceptionResNetV2.ipynb
 ├── history
 |      ├── ENet_history.csv
 |      ├── ENet-a_history.csv
 |      ├── IRN_history.csv
 |      └── IRN-a_history.csv
 ├── working_models
 |      ├── ENet_ep20_val0.311.zip
 |      └── IRN_ep13_val0.347.zip
 ├── assets
 |      ├── EfficientNet.png
 |      └── InceptionResNetV2.png
 ├── README.md
 └── slides.pdf
```

## Notebooks
* EfficientNet [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yxmauw/eye-disease-classification/blob/main/model_notebooks/EfficientNet.ipynb)
* InceptionResnetV2  [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yxmauw/eye-disease-classification/blob/main/model_notebooks/InceptionResNetV2.ipynb)

_For simplicity, the following will be based on EfficientNet notebook since both notebooks are very similar but with different ML models_

## Contents
1. Import image dataset from Kaggle using Kaggle API
2. Explore images, image hashing
3. Train, test, validation datasets split
4. Visualise image augmentation sample
5. Use EfficientNetV2S for Transfer learning as Base model with custom metrics 
6. Evaluate loss and accuracy trends
7. Add data augmentation layers to model and run
8. Evaluate loss and accuracy trends
9. Confusion matrices, Classification report
10. Evaluate misclassified images
11. Saliency Maps
12. Activation Heatmaps
13. ROC, PR curves to compare with InceptionResNetV2 model
14. Recommendations

## Methods
1. Download dataset into local drive using `!kaggle datasets download gunavenkatdoddi/eye-diseases-classification` from [Kaggle](https://www.kaggle.com/datasets/gunavenkatdoddi/eye-diseases-classification)
1. Explore images
     * Image count for each class / total image count
     * Image attributes `(512, 512, 3)`, `RGB`, pixels range `(0, 255)`
     * Image hashing: `4215` unique images 
1. Split dataset using `splitfolders` library at a ratio of `(.5, .25, .25)`
     * Read folders into tensor dataset formats using `image_dataset_from_directory()`, set `shuffle=False` for prediction evaluation 
1. Visualise preprocessed images 
     * Resizing `(160,160)` for EfficientNet or `(299,299)` for InceptionResNetV2
     * Rescaling pixels to `(0, 1)` - allows faster model convergence
     * Data augmentation using `RandomFlip("horizontal")`, `RandomRotation(0.1)`, `RandomContrast(0.1)`
1. Create base model using transfer learning **without** data augmentation layers
     * EfficientNetV2S: `input_shape=(160,160,3)`, `include_top=False`, `weights='imagenet'`, `pooling='max'`
     * `base_model.trainable=True`
     * Compile model: `optimizer='Adamax'`, `loss='categorical_crossentropy'`
        * Metrics: `categorical_accuracy`, `precision`, `recall`, `auc`, custom function `f1_score`
1. Evaluate base model
     * Accuracy 93.4%
     * Loss 0.311
     * Accuracy generalisation 6.29%
1. Add data augmentation layers to model
1. Evaluate augmented model - took more epoch runs to reach minimal loss
     * Accuracy 94.2%
     * Loss 0.312
     * Accuracy generalisation 5.66%
1. Confusion matrix <br> ![](https://github.com/yxmauw/eye-disease-classification/blob/main/assets/cm.jpg)
1. Classification report <br> ![](https://github.com/yxmauw/eye-disease-classification/blob/main/assets/cls_rp.jpg)
1. Some misclassified images <br> ![](https://github.com/yxmauw/eye-disease-classification/blob/main/assets/misclassified.jpg) <br> Reasons for misclassification:
     * Poorly taken 
     * Wrong source labelled
     * Abnormality not obvious, too mild (e.g. mild cataract)
1. Saliency map - Glaucoma example <br> ![](https://github.com/yxmauw/eye-disease-classification/blob/main/assets/saliency_map.jpg)
1. Activation heatmap - Glaucoma example <br> ![](https://github.com/yxmauw/eye-disease-classification/blob/main/assets/heatmap.jpg) <br> Optic disc region is activating the model the most, which is desired for detecting glaucoma
1. Comparison between EfficientNetV2S and InceptionResNetV2 models <br> ![](https://github.com/yxmauw/eye-disease-classification/blob/main/assets/roc.jpg) <br> InceptionResNetV2 model is marginally superior to EfficientNet, however inferior in identifying images with glaucoma.


## Results
|   Class  | Test set metrics | EfficientNetV2S | InceptionResNetV2 |
|:---------|:----------------:|:---------------:|:-----------------:|
|     -    |     Accuracy     |      93.00%     |       92.43%      |
| Cataract |    Precision     |      94.07%     |       90.91%      |
|          |    Recall        |      91.54%     |       96.15%      |
|          |    F1 score      |      92.79%     |       93.46%      |
|  DRE     |    Precision     |      99.64%     |       99.28%      |
|          |    Recall        |      100%       |       100%        |
|          |    F1 score      |      99.82%     |       99.64%      |
| Glaucoma |    Precision     |      90.04%     |       86.96%      |
|          |    Recall        |      89.33%     |       86.96%      |
|          |    F1 score      |      89.68%     |       86.96%      |
|  Normal  |    Precision     |      88.09%     |       92.06%      |
|          |    Recall        |      90.71%     |       86.25%      |
|          |    F1 score      |      89.38%     |       89.06%      |            


_DRE: diabetic retinopathy_


## Conclusions & Recommendations
__Limitations:__
1. Model cannot be used to identify retina images with multiple abnormalities. e.g. retina with both diabetic neuropathy and cataract. It would only be able to give 1 class as output, not multiple classes
1. Model is unable to pinpoint the region or abnormal feature found in the retina image - can only classify image as a whole.
1. Retina images with poor resolution, poor image capture (no retina or optic disc visualised) have higher chance of being wrongly classified.

__Conclusions:__
1. As most of the probabilities for wrongly classified images are quite high (>0.9), increasing probability threshold may not be useful in improving evaluation metrics.
1. This model is best used with other screening modalities to increase precision and accuracy of diagnosis e.g.
      * tonometer (for anterior chamber pressure) for glaucoma
      * snellen chart (visual acuity) for overall vision ability
      * auto perimeter visual field analyser (glaucoma)
      * Optical coherence tomography (OCT) - (identify retina layers, and subretinal deposits)
      
__Suggestions:__
1. Try individual colour channels for input images, e.g. just using green channels as input. May help model train better
1. Remove black background for input images, by adding alpha channel for transparency. May help model learn diabetic neuropathy more accurately.
1. Train a different model on just the diabetic neuropathy images vs normal images (keeping the black background), and see if the model is activated by correct features in the images with diabetic neuropathy rather than background.
1. Improve the input images by trimming off images with non-obvious abnormalities, distortion, or features not distinguishable for the particular class. E.g. glaucoma - need to have distinct optic disc cataract - need to have distinct blurring or dullness of retina normal - need to have good quality images, no confusing peripheral dullness or blurring or obscured optic disc that can be easily confused with glaucoma or cataract.
1. Try Meta Pseudo Label model, a semi-supervised learning method that achieves a new state-of-the-art top-1 accuracy of 90.2% on ImageNet, and see if model learns better with less confusion and less background activation


## App Deployment
* Link to app: [![Generic badge](https://img.shields.io/badge/🤗-Open%20In%20Spaces-blue.svg)](https://huggingface.co/spaces/yxmauw/eye-disease-clf-app)
* Link to app repository: [![Generic badge](https://badgen.net/badge/icon/Open%20Github%20Repo/blue?icon=github&label)](https://github.com/yxmauw/eye-disease-clf-app)

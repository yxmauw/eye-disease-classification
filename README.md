# :eye: Eye disease classification
A multiclass problem <br>
Option 2 for GA Capstone 22-Oct-2022


## Vision loss: A Public Healthcare Problem
More than 3.4 million Americans aged 40 years and older are blind (having a visual acuity of 20/200 or less or a visual field on 20 degrees or less) or visually impaired (having a visual acuity of 20/40 or less). Other estimates of “vision problems” range as high as 21 million, and a total of 80 million Americans have potentially blinding eye diseases. The major causes of vision loss are cataracts, age-related macular degeneration, diabetic retinopathy, and glaucoma. ([CDC et al](https://www.cdc.gov/visionhealth/basic_information/vision_loss.htm))

In Singapore, diabetic retinopathy is the leading cause of vision loss in working-age adults. According to a study led by researchers at the Singapore Eye Research Institute (SERI) in 2015, diabetic retinopathy has to date claimed the sight of more than 600 Singaporeans, the loss of an eye in 8,000, and visual impairment in a further 17,500. The problem worsens as the risk of blindness increases fifteen-fold for Singaporeans aged 50 to 80 and above. As age steals away the senses, vision loss is perhaps the most devastating, as it increases the risk of falls, depression and even premature death. ([SNEC et al](https://www.snec.com.sg/giving/singapores-eye-health))


## Problem Statement

_Fictional scenario_
> Singapore Health Promotion Board is searching for innovatve, cutting-edge technological solutions to facilitate mass eye screening of commom eye diseases in the general population. They hope to automate interpretation of retina screening images, shorten the time taken to flag high-risk persons for further health assessment. This would allow early intervention, lower risk of disease progression and lower overall healthcare cost burden. By minimising number of people in the population with severe eye disease, we minimise the use of more costly therapies.  

> Using 4217 retinal images collected from various sources like IDRiD, Oculur recognition, HRF etc, construct a machine learning algorithm to help classify an RGB retina image to any of 4 classes (cataract, diabetic retinopathy, glaucoma and normal).

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


---
## Conclusions


---
## App Deployment
* Link to app: [![Generic badge](https://img.shields.io/badge/🤗-Open%20In%20Spaces-blue.svg)](https://huggingface.co/spaces/yxmauw/eye-disease-clf-app)
* Link to app repository: [![Generic badge](https://badgen.net/badge/icon/Open%20Github%20Repo/blue?icon=github&label)](https://github.com/yxmauw/eye-disease-clf-app)

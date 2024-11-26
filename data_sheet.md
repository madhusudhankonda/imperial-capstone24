# Data Sheet for Predicting Damage from Natural Disasters

## Project Overview

This project aims to leverage machine learning techniques to analyze satellite imagery for predicting damage from natural disasters, specifically focusing on urban areas like Rio de Janeiro. The insights generated can assist urban planners and policymakers in making informed decisions regarding disaster management and urban development.

## Training Data

The model was trained on the [AOI 1 - Rio](https://spacenet.ai/spacenet-buildings-dataset-v1/) dataset, which includes satellite images and building boundaries in Rio de Janeiro, covering over 2,500 sq. km. The training set consists of approximately 7,000 images, each covering an area of 200m x 200m with a resolution of about 400px x 400px. The building boundaries are provided in vector shapefile format.

## Model Architecture

The project utilizes a vanilla U-net architecture, which includes four downsampling encoders and four upsampling decoders. This architecture is effective for image segmentation tasks, allowing the model to extract important features from the input images and accurately predict building boundaries.

## Performance Metrics

The model's performance is evaluated using the Intersection over Union (IoU) metric, which measures the overlap between the predicted building areas and the ground truth. The U-net model achieved an IoU of 0.693, outperforming other models like Random Forest and Multi-task Network Cascades.

## Generalizability

The models were tested on satellite images from Antalya, Turkey, to assess their generalizability. The U-net model demonstrated strong performance, correctly identifying buildings in 7 out of 10 images in 2022 and 9 out of 10 in 2024.

## Conclusion

This data sheet summarizes the key aspects of the project, including the data used, model architecture, and performance metrics. The insights gained from this analysis can significantly contribute to disaster management and urban planning efforts.


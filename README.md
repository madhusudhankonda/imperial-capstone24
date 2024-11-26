# Predicting Damage from Natural Disasters

## Overview

This project is an AI/ML capstone initiative aimed at predicting damage from natural disasters using machine learning techniques. By analyzing satellite imagery, particularly focusing on urban areas like Rio de Janeiro, the project seeks to provide valuable insights for urban planners and policymakers to enhance disaster management and urban development strategies.

## Project Goals

- Leverage machine learning to analyze satellite imagery.
- Classify land use patterns and monitor environmental changes.
- Provide actionable insights for urban planning and disaster management.

## Data

The model is trained on the [AOI 1 - Rio](https://spacenet.ai/spacenet-buildings-dataset-v1/) dataset, which includes:
- Satellite images and building boundaries in Rio de Janeiro.
- Coverage of over 2,500 sq. km.
- Approximately 7,000 images, each covering an area of 200m x 200m with a resolution of about 400px x 400px.
- Building boundaries provided in vector shapefile format.

## Model Architecture

The project employs a vanilla U-net architecture, which consists of:
- Four downsampling encoders.
- Four upsampling decoders.
This architecture is well-suited for image segmentation tasks, allowing for effective feature extraction and accurate prediction of building boundaries.

## Performance Metrics

The model's performance is evaluated using the Intersection over Union (IoU) metric, achieving an IoU of 0.693, which indicates strong performance compared to other models like Random Forest and Multi-task Network Cascades.

## Generalizability

The models were tested on satellite images from Antalya, Turkey, to assess their generalizability. The U-net model showed strong performance, correctly identifying buildings in 7 out of 10 images in 2022 and 9 out of 10 in 2024.

## Conclusion

This project demonstrates the potential of machine learning in analyzing satellite imagery for disaster management and urban planning. The insights gained can significantly contribute to informed decision-making in these critical areas.

## References

- [SpaceNet Dataset](https://spacenet.ai/spacenet-buildings-dataset-v1/)
- [U-net Architecture](https://link.springer.com/chapter/10.1007/978-3-319-24574-4_28) 


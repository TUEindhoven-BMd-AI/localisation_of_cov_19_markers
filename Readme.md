# Readme about segmentation models

This repo provides the code for the frame-based segmentation models presented in our paper:
[Deep learning for classification and localization of COVID-19 markers in point-of-care lung ultrasound]

## How to run the code

### Dependencies
Download the [anaconda](https://www.anaconda.com/) package. 

In the anaconda prompt run:
```
conda env create -f COVID19-environment.yml
```

and then activate the environment:
```
conda activate COVID19
```

### Pixel-wise segmentation models
- Run PreprocessingData/mainPreprocessing.py to preprocess the datafolder containing the raw data.
- Run main.py to train one of the models.
This main contains the settings as used in the paper, but the --segmentation_model parameter should be set to indicate the model that you want to train.
- Run inference/main_inference.py to run inference on a trained model.


## Citation

Please cite our paper if you use this database or code in your own work:

```
@article{roy2020deep,
  title={Deep learning for classification and localization of COVID-19 markers in point-of-care lung ultrasound.},
  author={Roy, S and Menapace, W and Oei, S and Luijten, B and Fini, E and Saltori, C and Huijben, I and Chennakeshava, N and Mento, F and Sentelli, A and others},
  journal={IEEE Transactions on Medical Imaging},
  year={2020}
}
```
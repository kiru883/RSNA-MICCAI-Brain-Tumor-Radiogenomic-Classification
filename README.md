# RSNA-MICCAI-Brain-Tumor-Radiogenomic-Classification
Kaggle competition on predict MGMT gen by brain CT images, 
[competition link](https://www.kaggle.com/c/rsna-miccai-brain-tumor-radiogenomic-classification).
All experiment part adapted for *kaggle* notebooks! **Write** to provide a particular model.
Sorry for quality of code...

This repository created for to show how i solved this competition and solution have __raw__ format.

# Result
* __Place__ - 128 / 1555, bronze. 
* __Scores__ - 0.59778(LB) / 0.57695(PB), max PB score - 0.62174

# EDA
```
/experiments/EDA/
```
For using EDA you need to attach competition data. In EDA im show MGMT label distribution,
examples of DICOM CT scans and example of PSEUDO-RGB preprocessed images. 

# Experiments
## A
```
/experiments/A/
```
__Preprocessing__:

```
/experiments/A/get_data_pseudorgb_v1_flair_train.py
```
Try use PSEUDO RGB image(only FLAIR type of scans) preprocessing, for this preprocess each 
channel of 3d image:
1. First channel(R) is mean value of 512 channels.
2. Second channel(G) is max value of 512 channels.
3. Last channel(B) is std value of 512 channels.

After receiving all channels, normalization was performed for each of the 3 channels. 
Thus, I changed the dimension of the image from 512x512x512 to 512x512x3.

__Model training__:
```
/experiments/A/train-resnet-50-pseudorgb-v1.ipynb
```
Im build(from scratch) ResNet50, pytorch lightning wrapper from model and torch dataset class.
Train parameters - max number of epochs - 60, loss - MSE, initial learning rate - 1e-6, save callback - 
save model with minimum MSE loss, test size - 0.3, size of batch - 16.

__Submit__:
```
/experiments/A/submit-a.ipynb
```
**Result** - 0.58086(LB)

## A.1
```
/experiments/A.1/submit-resnet50-pseudorgb-3f-v1.ipynb
/experiments/A.1/train-resnet50-pseudorgb-v1-3f.ipynb
```
All the same as in A exp., but try use CV(3 folds), when predict use 3 models and average 3 predicts.

**Result** - 0.58298(LB) / 0.57275(PB)

## B
```
/experiments/B/
```
__Preprocessing__:

Open DICOM **FLAIR** ct image and get only 64 central slices, add additional zeros 
slices(must be 256 slices) if nessesary.

__Model training__:
```
/experiments/B/train-3d-custom-v1.ipynb
```
Train parameters - 30 epochs, lr - 3e-5, batch size - 2, test size - 0.3.
Model code:
```
class Custom3DNet(nn.Module):
    def __init__(self):
        super(Custom3DNet, self).__init__()
        self.block1 = self.__gen_block(1, 64, 3, 2, 0.01)
        self.block2 = self.__gen_block(64, 128, 3, 2, 0.02)
        self.block3 = self.__gen_block(128, 256, 3, 2, 0.03)
        self.block4 = self.__gen_block(256, 512, 3, 2, 0.04)
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten(),
            nn.Linear(512, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout3d(p=0.08),
            nn.Linear(1024, 1),
            nn.Sigmoid()
        )
        
        
    def forward(self, inp):
        x = self.block1(inp)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        #print(nn.AdaptiveAvgPool3d(1)(x).shape)
        return self.classifier(x)
        
        
    def __gen_block(self, in_channels, out_channels, kernel_size, pool_size, dropout=None):
        layers = [
            nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=pool_size) ,
            nn.BatchNorm3d(num_features=out_channels)
        ]
        if not dropout is None:
            layers += [nn.Dropout3d(p=dropout)]
        return nn.Sequential(*layers)
```

__Submit__:
```
/experiments/B/submit-3d-custom-v1.ipynb
```
**Result** - 0.59778(LB) / 0.57694(PB)

# B.1
```
/experiments/B.1/submit-3d-custom-3f-v1.ipynb
/experiments/B.1/train-3d-custom-v1-3f.ipynb
```
All the same as in B exp., but try use CV(3 folds), when predict use 3 models and average 3 predicts.

**Result** - 0.53857(LB)

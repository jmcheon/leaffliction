# Leaffliction - Computer vision
>*_Summary: Image classification by disease recognition on leaves._*

| Requirements | Skills |
|--------------|--------|
| - `python3.10`<br> - `torch`<br> - `torchvision`<br> - `opencv`<br> - `plantcv`<br> - `numpy`<br>  - `matplotlib`<br>  | - `Rigor`<br> - `Group & interpersonal`<br> - `Algorithms & AI` |

## Usage
```bash
# Download image dataset and generate distribution chart image
python3  01.Distribution.py  apple  grape

# Augment unbalanced image dataset
python3  02.Augmentation.py

# Save transformed image plots
python3  03.Transformation.py  -src [SRC_PATH] -dst [DST_PATH]

# Print the accuracy on validation dataset
python3  04.Classification
```

### Tensorboard
To visualize the learning curves using tensorboard, execute the following command.
```
tensorboard --logdir runs
```

## Implementation

### Leaf Classifier CNN Model
The model is designed to classify leaf diseases based on images of leaves. The model is implemented using Pytorch and consists of 4 convolutional layers followed by max pooling, along with 2 fully connected layers. The final output is produced using a softmax function for multi-class classification.

#### Model Architecture
1. Input layer
	- Input: Leaf images with a shape of (256, 256, 3) corresponding to 256 x 256 RGB images.

2. Convolutional layers
	- Conv Layer 1
		- Input channels: 3 (RGB)
		- Output channels: 32
		- Kernel size: 3 x 3
		- Activation function: ReLU
		- Max Pooling: 2 x 2
	- Conv Layer 2
		- Input channels: 32
		- Output channels: 64
		- Kernel size: 3 x 3
		- Activation function: ReLU
		- Max Pooling: 2 x 2
	- Conv Layer 3
		- Input channels: 64
		- Output channels: 128
		- Kernel size: 3 x 3
		- Activation function: ReLU
		- Max Pooling: 2 x 2
	- Conv Layer 4
		- Input channels: 128
		- Output channels: 256
		- Kernel size: 3 x 3
		- Activation function: ReLU
		- Max Pooling: 2 x 2

3. Fully connected layers
	- FC Layer 1
		- Input: Flattened tensor from the previous convolutional layers (256 * 14 * 14 = 50176 units)
		- Output: 512 units
		- Activation function: ReLU
		- Dropout: 0.5
	- FC Layer 2
		- Input: 512 units
		- Output: `NUM_CLASSES` units (representing the number of disease classes)
		- Activation function: Softmax

## Visualization

### Test Accuracy
We have 10 test images and the model has 100% accuracy 
<img src="https://github.com/jmcheon/leaffliction/blob/main/assets/leaffliction_output_ex1.png" alt="output example1">

### Validation Accuracy
<img src="https://github.com/jmcheon/leaffliction/blob/main/assets/validation_accuracy_ex1.png" alt="validation accuracy">


## Resources
- [Youtube Coursera CNN](https://www.youtube.com/playlist?list=PLkDaE6sCZn6Gl29AoE31iwdVwSG-KnDzF)

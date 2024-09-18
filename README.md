# Leaffliction - Computer vision
>*_Summary: Image classification by disease recognition on leaves._*

| Requirements | Skills |
|--------------|--------|
| - `python3.10`<br> - `torch`<br> - `torchvision`<br> - `opencv`<br> - `plantcv`<br> - `numpy`<br>  - `matplotlib`<br>  | - `Rigor`<br> - `Group & interpersonal`<br> - `Algorithms & AI` |

## Usage
There are 4 distinct parts in this project, `01. Distribution`, `02. Augmentation`, `03. Transformation`, and `04. Classification`.

### 01. Distribution
Download image dataset and generate distribution chart image
```bash
usage: 01.Distribution.py [-h] directories [directories ...]

A program to analyze plant images and generate charts.

positional arguments:
  directories  The directories to store extracted images and save the charts (ex: 01.Distribution apple)

options:
  -h, --help   show this help message and exit
```

#### Example
```bash
python3  01.Distribution.py  apple  grape
```

<br/>

### 02. Augmentation
Augment unbalanced image dataset
```bash
usage: 02.Augmentation.py [-h] [file_path]

A program to augment images samples by applying 6 types of transformation.

positional arguments:
  file_path   Image file path to transform to 6 different types.

options:
  -h, --help  show this help message and exit
```

#### Example
```bash
python3  02.Augmentation.py
```

<br/>

### 03. Transformation
Save transformed image plots

```bash
usage: 03.Transformation.py [-h] -src [SRC_PATH] [-dst [DST_PATH]] [-gaussian] [-mask] [-roi] [-analyze] [-pseudo] [-hist]

A program to display image transformation.

options:
  -h, --help            show this help message and exit
  -src [SRC_PATH], --src_path [SRC_PATH]
                        Image file path.
  -dst [DST_PATH], --dst_path [DST_PATH]
                        Destination directory path.
  -gaussian, --gaussian_blur
                        Gaussian Transform
  -mask                 Mask Transform
  -roi, --roi_objects   Roi Transform
  -analyze, --analyze_object
                        Analyze Transform
  -pseudo, --pseudolandmarks
                        Psudolandmark Transform
  -hist, --color_histogram
                        Color histogram Transform
```

#### Example
```bash
python3  03.Transformation.py  -src [SRC_PATH] -dst [DST_PATH]
```

<br/>

### 04. Classification
Print the accuracy on validation dataset

```bash
usage: 04.Classification.py [-h] [folder_path]

A program to classify a type of leaf from validation set.

positional arguments:
  folder_path  Image folder path.

options:
  -h, --help   show this help message and exit
```

#### Example
```bash
python3  04.Classification
```

<br/>
<br/>

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

### 01. Distribution
There are 2 distinct leaf types; `apple` and `grape`, each of which consists of 4 labels.
<table>
	<tr>
		<th>
			Apple Image Distribution
		</th>
		<th>
			Grape Image Distribution
		</th>
	</tr>
	<tr>
		<td>
			<img src="https://github.com/jmcheon/leaffliction/blob/main/assets/apple_combined_chart.png" alt="apple image distribution">
		</td>
  		<td>
			<img src="https://github.com/jmcheon/leaffliction/blob/main/assets/grape_combined_chart.png" alt="grape image distribution">	
		</td>
	</tr>
</table>

### 02. Augmentation
The following 6 image augmentation techniques are applied to one single-leaf image labeled `apple black rot`.

<table>
	<tr>
		<th>
			Brightness
		</th>
		<th>
			Contrast
		</th>
		<th>
			Flip
		</th>
		<th>
			Perspective
		</th>
		<th>
			Rotate
		</th>
		<th>
			Saturation
		</th>
	</tr>
	<tr>
		<td>
			<img src="https://github.com/jmcheon/leaffliction/blob/main/assets/apple_black_rot_image (100)_Brightness.JPG" alt="augmentation brightness image" width=175px height=175px>
		</td>
		<td>
			<img src="https://github.com/jmcheon/leaffliction/blob/main/assets/apple_black_rot_image (100)_Contrast.JPG" alt="augmentation contrast image" width=175px height=175px>
		</td>
		<td>
			<img src="https://github.com/jmcheon/leaffliction/blob/main/assets/apple_black_rot_image (100)_Flip.JPG" alt="augmentation flip image" width=175px height=175px>
		</td>
		<td>
			<img src="https://github.com/jmcheon/leaffliction/blob/main/assets/apple_black_rot_image (100)_Perspective.JPG" alt="augmentation perspective image" width=175px height=175px>
		</td>
		<td>
			<img src="https://github.com/jmcheon/leaffliction/blob/main/assets/apple_black_rot_image (100)_Rotate.JPG" alt="augmentation rotate image" width=175px height=175px>
		</td>
		<td>
			<img src="https://github.com/jmcheon/leaffliction/blob/main/assets/apple_black_rot_image (100)_Saturation.JPG" alt="augmentation saturation image" width=175px height=175px>
		</td>
	</tr>
</table>

### 03. Transformation
The following 6 image transformation techniques are applied to one single-leaf image labeled `apple black rot`.

<table>
	<tr>
		<th>
			Mask
		</th>
		<th>
			Gaussian Blur
		</th>
		<th>
			Roi objects
		</th>
		<th>
			Analyze object
		</th>
		<th>
			Pseudolandmarks
		</th>
	</tr>
	<tr>
		<td>
			<img src="https://github.com/jmcheon/leaffliction/blob/main/assets/apple_black_rot_image (100)_mask.JPG" alt="transformation mask image" width=175px height=175px>
		</td>
		<td>
			<img src="https://github.com/jmcheon/leaffliction/blob/main/assets/apple_black_rot_image (100)_gaussian_blur.JPG" alt="transformation gaussian blur image" width=175px height=175px>
		</td>
		<td>
			<img src="https://github.com/jmcheon/leaffliction/blob/main/assets/apple_black_rot_image (100)_roi_objects.JPG" alt="transformation roi objects image" width=175px height=175px>
		</td>
		<td>
			<img src="https://github.com/jmcheon/leaffliction/blob/main/assets/apple_black_rot_image (100)_analyze_object.JPG" alt="transformation analyze object image" width=175px height=175px>
		</td>
		<td>
			<img src="https://github.com/jmcheon/leaffliction/blob/main/assets/apple_black_rot_image (100)_pseudolandmarks.JPG" alt="transformation pseudolandmarks image" width=175px height=175px>
		</td>
	</tr>
</table>

<br/>
<br/>

<table align="center">
	<tr>
		<th>
			Color Histogram
		</th>
	</tr>
	<tr>
  		<td>
			<img src="https://github.com/jmcheon/leaffliction/blob/main/assets/apple_black_rot_image (100)_color_histogram.JPG" alt="color histogram image" width=600px height=400px>
		</td>
	</tr>
</table>

### 04. Classification

#### Tensorboard
To visualize the learning curves using tensorboard, execute the following command.
```
tensorboard --logdir runs
```

#### Validation Accuracy
<img src="https://github.com/jmcheon/leaffliction/blob/main/assets/validation_accuracy_ex1.png" alt="validation accuracy">

#### Test Accuracy
We have 10 test images and the model has 100% accuracy 
<div align="center">
	<img src="https://github.com/jmcheon/leaffliction/blob/main/assets/predicted.png" alt="predicted example1">
</div>


## Resources
- [Youtube Coursera CNN](https://www.youtube.com/playlist?list=PLkDaE6sCZn6Gl29AoE31iwdVwSG-KnDzF)

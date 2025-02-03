# Lung & Liver Segmentation

This project was developed as part of the Signal, Image, and Video course for the Master's Degree in Artificial Intelligence Systems at the University of Trento. The goal of this project is to perform segmentation of the lungs and liver in medical images using various image analysis techniques.

## Lung Segmentation

The lung segmentation code is available in the notebook [Lung_segmentation.ipynb](Lung_Segmentation/lung_segmentation.ipynb).

<img src="Miscellaneous/lungs.png" alt="3D Lungs" width="500"/>

## Liver Segmentation

The liver segmentation code is available in the [Liver_Segmentation](Liver_Segmentation) folder.

### Exploration, Flood, and Watershed Approaches

These approaches are implemented in the [Processing_Segmentation](Liver_Segmentation/Processing_Segmentation) folder. The file [segmentation_functions.py](Liver_Segmentation/Processing_Segmentation/segmentation_functions.py) contains the three functions for segmentation. The file [Combined_Segmentation_GUI.py](Liver_Segmentation/Processing_Segmentation/Combined_Segmentation_GUI.py) provides a GUI for conducting segmentation on medical images.

<img src="Miscellaneous/gui.png" alt="Segmentation GUI" width="500"/>


### U-Net

The U-Net approach is implemented in the [Unet_Segmentation](Liver_Segmentation/Unet_Segmentation) folder.

<img src="Miscellaneous/output.png" alt="Liver slice" width="1000"/>

## Installation

### Using Conda

1. Clone the repository:
   ```bash
   git clone https://github.com/avendramini/Lung_Liver_Segmentation.git
   cd Lung_Liver_Segmentation
   ```
2. Create the environment:
   ```bash
   conda env create -f environment.yml
   ```
3. Activate the environment:
   ```bash
   conda activate siv
   ```

### Using Pip

1. Clone the repository:
   ```bash
   git clone https://github.com/avendramini/Lung_Liver_Segmentation.git
   cd Lung_Liver_Segmentation
   ```
2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Lung Segmentation

To run the lung segmentation, open the [Lung_segmentation.ipynb](Lung_Segmentation/lung_segmentation.ipynb) notebook and follow the instructions.

### Liver Segmentation

#### GUI for Segmentation

To use the GUI for liver segmentation, run the [Combined_Segmentation_GUI.py](Liver_Segmentation/Processing_Segmentation/Combined_Segmentation_GUI.py) script. This GUI allows you to load medical images and perform segmentation using the Exploration, Flood, and Watershed approaches.

#### U-Net Segmentation
To use the U-Net segmentation, follow the instructions in the [Unet_Liver.ipynb](Liver_Segmentation/Unet_Segmentation/Unet_Liver.ipynb) notebook.

## Results

The results of the segmentation can be visualized using the provided GUI or by running the respective scripts and notebooks. The Dice scores and other metrics are calculated to evaluate the performance of the segmentation approaches.

## Performance Metrics

### Lung Segmentation

The performance of the lung segmentation approach is evaluated using the Dice coefficient. The following graph shows the Dice scores for lung segmentation:

<img src="Miscellaneous/lung_dice.png" alt="Lung Segmentation Dice Scores" width="500"/>

### Liver Segmentation

#### Exploration, Flood, and Watershed Approaches

The performance of the liver segmentation using the Exploration, Flood, and Watershed approaches is evaluated using the Dice coefficient. The following graph shows the Dice scores for these approaches:

<img src="Miscellaneous/3_approaches_dice.png" alt="Liver Segmentation Dice Scores for 3 Approaches" width="500"/>

Note: The models were trained with limited computational power, which may affect the performance results.

#### U-Net Segmentation

The performance of the U-Net segmentation approach is evaluated using the Dice coefficient. The following graph shows the Dice scores for U-Net segmentation:

<img src="Miscellaneous/dice_scores_unet.png" alt="U-Net Segmentation Dice Scores" width="500"/>




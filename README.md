# Lung & Liver Segmentation

This project was developed as part of the Signal, Image, and Video course for the Master's Degree in Artificial Intelligence Systems at the University of Trento.
The goal of this project is to perform segmentation of the lungs and liver in medical images using image analysis techniques.
## Lung Segmentation
The Lung segmentation code is entirely available in the notebook [Lung_segmentation.ipynb](Lung_Segmentation/lung_.ipynb).

## Liver Segmentation
The Liver segmentation code is entirely available in the [Liver_Segmentation](Liver_Segmentation) folder.
### Exploration, Flood and Watershed approaches
These approaches' code is in the [Processing_Segmentation](Liver_Segmentation/Processing_Segmentation) folder.
The file [Segmentation_Functions.py](Liver_Segmentation/Processing_Segmentation/segmentation_functions.py) contains the 3 functions for segmentation used.
The file [Combined_Segmentation_GUI.py](Liver_Segmentation/Processing_Segmentation/Combined_Segmentation_GUI.py) is the code of a GUI for conducting segmentation.given a medical image.
### U-Net
These approach's code is in the [Unet_Segmentation](Liver_Segmentation/Unet_Segmentation) folder.

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



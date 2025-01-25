import os
import nibabel as nib
import numpy as np

def split_nii_by_64(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        if filename.endswith('.nii.gz'):
            filepath = os.path.join(input_folder, filename)
            img = nib.load(filepath)
            data = img.get_fdata()
            
            num_slices = data.shape[2]
            num_chunks = num_slices // 64
            
            for i in range(num_chunks):
                chunk_data = data[:, :, i*64:(i+1)*64]
                chunk_img = nib.Nifti1Image(chunk_data, img.affine)
                chunk_filename = f"{os.path.splitext(filename)[0]}_chunk_{i}.nii.gz"
                nib.save(chunk_img, os.path.join(output_folder, chunk_filename))

def process_folders(base_folder):
    folders = ['TestSegmentation', 'TrainSegmentation', 'TestVolumes', 'TrainVolumes']
    for folder in folders:
        input_folder = os.path.join(base_folder, folder)
        output_folder = os.path.join(base_folder, f"{folder}_splitted")
        split_nii_by_64(input_folder, output_folder)

if __name__ == "__main__":
    base_folder = '.\\'
    process_folders(base_folder)
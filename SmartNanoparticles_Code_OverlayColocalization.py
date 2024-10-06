# -*- coding: utf-8 -*-
"""
Created on Fri Aug  2 10:46:45 2024

@author: Home
"""


#%%

# Step 1 : Pre- processing. 
# Convert the grayscale images to colored images. (Raw images is given as a grayscale).
# The output folder named as colored images.

import os
import numpy as np
from PIL import Image
import easygui

def apply_lut(image, color):
    # Convert image to numpy array
    img_array = np.array(image)
    
    # Ensure image is 2D (grayscale)
    if len(img_array.shape) > 2:
        img_array = img_array[:,:,0]  # Take first channel if it's multi-channel
    
    # Create colored image
    colored = np.zeros((img_array.shape[0], img_array.shape[1], 3), dtype=np.uint8)
    
    if color == 'blue':
        colored[:,:,2] = img_array
    elif color == 'green':
        colored[:,:,1] = img_array
    elif color == 'red':
        colored[:,:,0] = img_array
    elif color == 'yellow':
        colored[:,:,0] = img_array
        colored[:,:,1] = img_array
    
    return Image.fromarray(colored)

def process_images():
    input_folder = easygui.diropenbox("Select the folder containing TIFF images")
    
    if not input_folder:
        print("No folder selected. Exiting.")
        return

    output_folder = os.path.join(input_folder, "colored_images")
    os.makedirs(output_folder, exist_ok=True)

    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.tif', '.tiff')):
            file_path = os.path.join(input_folder, filename)
            
            if filename.endswith('_1.tif') or filename.endswith('_1.tiff'):
                color = 'blue'
            elif filename.endswith('_2.tif') or filename.endswith('_2.tiff'):
                color = 'green'
            elif filename.endswith('_3.tif') or filename.endswith('_3.tiff'):
                color = 'red'
            elif filename.endswith('_4.tif') or filename.endswith('_4.tiff'):
                color = 'yellow'
            else:
                continue

            try:
                with Image.open(file_path) as img:
                    print(f"Processing {filename}: Mode = {img.mode}, Size = {img.size}, Format = {img.format}")
                    
                    colored_img = apply_lut(img, color)
                
                output_filename = f"{os.path.splitext(filename)[0]}_{color}.tif"
                output_path = os.path.join(output_folder, output_filename)
                colored_img.save(output_path)
                
                print(f"Processed: {filename} -> {output_filename}")
            except Exception as e:
                print(f"Error processing {filename}: {str(e)}")

    print(f"All images processed. Colored images saved in: {output_folder}")

if __name__ == "__main__":
    process_images()


#%%

# Step 2 : Specific region Segmentation - red color
# Using Roboflow to segment regions - Converting COCO format to labeled masks - Segmentation (with & without contours).
# Three output folders are generated and named as follows:  1) segmented_images_with_contours, 2) segmented_RGB_color, 3) segmented_images_without_contours.

import os
import json
import cv2
import numpy as np
import re
import easygui

def create_output_folders(base_dir):
    seg_output_dir = os.path.join(base_dir, 'segmented_images_with_contours')
    rgb_output_dir = os.path.join(base_dir, 'segmented_RGB_color')
    no_contour_dir = os.path.join(base_dir, 'segmented_images_without_contours')
    os.makedirs(seg_output_dir, exist_ok=True)
    os.makedirs(rgb_output_dir, exist_ok=True)
    os.makedirs(no_contour_dir, exist_ok=True)
    return seg_output_dir, rgb_output_dir, no_contour_dir

def clean_filename(filename):
    # Remove the '.rf...' part and everything after it, and remove '_png'
    cleaned = re.sub(r'\.rf\.[^.]+', '', filename)
    cleaned = cleaned.replace('_png', '')
    
    # Remove any file extension in the middle (like .jpg)
    cleaned = re.sub(r'\.(jpg|jpeg|png|gif|bmp)\.', '.', cleaned, flags=re.IGNORECASE)
    
    # Ensure the filename ends with .png
    if not cleaned.lower().endswith('.png'):
        cleaned = os.path.splitext(cleaned)[0] + '.png'
    
    return cleaned

def save_images_with_coco_annotations(image_paths, annotations, seg_output_dir, rgb_output_dir, no_contour_dir):
    for img_path in image_paths:
        # Load image using OpenCV and convert it from BGR to RGB color space
        image = cv2.imread(img_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Get image filename to match with annotations
        img_filename = os.path.basename(img_path)
        cleaned_filename = clean_filename(img_filename)
        img_id = next(item for item in annotations['images'] if item["file_name"] == img_filename)['id']
        
        # Filter annotations for the current image
        img_annotations = [ann for ann in annotations['annotations'] if ann['image_id'] == img_id]
        
        # Create a mask for instance segmentation
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        
        # Create copies of the original image for segmentation overlay
        overlay_image = image_rgb.copy()
        no_contour_image = image_rgb.copy()
        
        for ann in img_annotations:
            # Display segmentation polygon
            for seg in ann['segmentation']:
                poly = [(seg[i], seg[i+1]) for i in range(0, len(seg), 2)]
                
                # Draw thick blue contour on the overlay image
                cv2.polylines(overlay_image, [np.array(poly, dtype=np.int32)], 
                              isClosed=True, color=(0, 255, 0), thickness=10)
                
                # Fill the polygon in the mask
                cv2.fillPoly(mask, [np.array(poly, dtype=np.int32)], 255)

        # Apply mask to the no_contour_image
        no_contour_image = cv2.bitwise_and(no_contour_image, no_contour_image, mask=mask)

        # Save the image with segmentation overlay
        seg_output_path = os.path.join(seg_output_dir, cleaned_filename)
        cv2.imwrite(seg_output_path, cv2.cvtColor(overlay_image, cv2.COLOR_RGB2BGR))

        # Save the segmented image without contour
        no_contour_output_path = os.path.join(no_contour_dir, cleaned_filename)
        cv2.imwrite(no_contour_output_path, cv2.cvtColor(no_contour_image, cv2.COLOR_RGB2BGR))

        # Create and save the instance segmentation image
        instance_seg = cv2.bitwise_and(image_rgb, image_rgb, mask=mask)
        rgb_output_path = os.path.join(rgb_output_dir, cleaned_filename)
        cv2.imwrite(rgb_output_path, cv2.cvtColor(instance_seg, cv2.COLOR_RGB2BGR))

# Prompt user to select input directory using easygui
input_dir = easygui.diropenbox(title="Select Input Folder", msg="Choose the folder containing images and annotations")

if input_dir is not None:
    annotations_path = os.path.join(input_dir, '_annotations.coco.json')
    
    # Load COCO annotations
    with open(annotations_path, 'r') as f:
        annotations = json.load(f)

    # Create output folders
    seg_output_dir, rgb_output_dir, no_contour_dir = create_output_folders(os.path.dirname(input_dir))

    # Get all image files
    all_image_files = [os.path.join(input_dir, img['file_name']) for img in annotations['images']]

    # Process all images
    save_images_with_coco_annotations(all_image_files, annotations, seg_output_dir, rgb_output_dir, no_contour_dir)

    print(f"Segmented images with contours saved in: {seg_output_dir}")
    print(f"Segmented images without contours saved in: {no_contour_dir}")
    print(f"RGB instance segmentation images saved in: {rgb_output_dir}")
else:
    print("No folder selected. Exiting the program.")
    
#%%


# Step 3: apply the previuos segmentation at the same specific red- segmentaed location for the other colors (blue, green and yellow).
# First choose the segmented red color folder and then choose each time a specific color folder to apply the segmentation.

import cv2
import os
import numpy as np
import easygui

def segment_and_overlay_images():
    # Select the folder containing red images
    red_folder = easygui.diropenbox(title="Select Folder with Red Images")
    
    # Select the folder containing other color images
    other_colors_folder = easygui.diropenbox(title="Select Folder with Other Color Images")
    
    # Create an output folder to save the results
    output_folder = os.path.join(other_colors_folder, "Overlayed_Images")
    os.makedirs(output_folder, exist_ok=True)
    
    # List all images in the red folder
    red_images = sorted([f for f in os.listdir(red_folder) if f.endswith(('.png', '.jpg', '.jpeg', '.tiff'))])
    
    # List all images in the other colors folder
    other_images = sorted([f for f in os.listdir(other_colors_folder) if f.endswith(('.png', '.jpg', '.jpeg', '.tiff'))])
    
    # Ensure both folders have the same number of images
    if len(red_images) != len(other_images):
        print("The number of images in both folders should be the same.")
        return
    
    for red_image_name, other_image_name in zip(red_images, other_images):
        # Read the red image
        red_image_path = os.path.join(red_folder, red_image_name)
        red_image = cv2.imread(red_image_path)
        
        # Read the corresponding other color image
        other_image_path = os.path.join(other_colors_folder, other_image_name)
        other_image = cv2.imread(other_image_path)
        
        # Check if images are of the same size
        if red_image.shape != other_image.shape:
            print(f"Image sizes do not match for {red_image_name} and {other_image_name}. Skipping.")
            continue
        
        # Assume the segmentation is already done in the red channel, create a mask
        red_channel = red_image[:, :, 2]  # Extract the red channel
        _, mask = cv2.threshold(red_channel, 1, 255, cv2.THRESH_BINARY)  # Create a binary mask
        
        # Overlay the red mask on the other image
        overlayed_image = cv2.bitwise_and(other_image, other_image, mask=mask)
        
        # Save the overlayed image
        output_image_path = os.path.join(output_folder, f"overlayed_{other_image_name}")
        cv2.imwrite(output_image_path, overlayed_image)
        
        print(f"Overlayed image saved: {output_image_path}")

if __name__ == "__main__":
    segment_and_overlay_images()


#%%

# Step 4 : Brightness the segmented images from the previous step, specifically the blue, green and yellow colors by factor of 2.
# The output folder is called brightened_images.

import os
import easygui
from PIL import Image, ImageEnhance

def increase_brightness(image_path, output_path, factor=2.0):
    with Image.open(image_path) as img:
        enhancer = ImageEnhance.Brightness(img)
        bright_img = enhancer.enhance(factor)
        bright_img.save(output_path)

def process_images():
    # Select input folder
    input_folder = easygui.diropenbox("Select the folder containing images")
    
    if not input_folder:
        print("No folder selected. Exiting.")
        return

    # Create output folder
    output_folder = os.path.join(input_folder, "brightened_images")
    os.makedirs(output_folder, exist_ok=True)

    # Process images
    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, f"bright_{filename}")
            
            try:
                increase_brightness(input_path, output_path)
                print(f"Processed: {filename}")
            except Exception as e:
                print(f"Error processing {filename}: {str(e)}")

    print(f"All images processed. Brightened images saved in: {output_folder}")

if __name__ == "__main__":
    process_images()
    
    
    
#%%

# Step 5 : Nuclei Segmentation for each channel seperatly - Apply colors for the masks in FiJi
# Utilizing StarDist algorithm.
# The results will be saved in a folder called as stardist_output. (apply for each color seperatly).

from stardist.models import StarDist2D 
from csbdeep.utils import normalize
from skimage import io, measure, color
import numpy as np
import os
from tqdm import tqdm
import easygui

# Load the model
model = StarDist2D.from_pretrained('2D_versatile_fluo')

# Use easygui to choose the input folder
input_folder = easygui.diropenbox(title="Select Input Folder")
if not input_folder:
    print("No folder selected. Exiting.")
    exit()

# Set up output folder
output_folder = os.path.join(os.path.dirname(input_folder), "stardist_output")
os.makedirs(output_folder, exist_ok=True)

# Adjust these parameters
prob_thresh = 0.8  # Increase this to be more selective
nms_thresh = 0.4   # Decrease this to allow more overlap
min_size = 50  # minimum area of object to keep
max_size = 800  # maximum area of object to keep

# Get list of image files
image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.png', '.tif', '.jpg', '.jpeg'))]

# Process all images in the input folder
for filename in tqdm(image_files, desc="Processing images"):
    # Load image
    img_path = os.path.join(input_folder, filename)
    my_fl_img = io.imread(img_path, as_gray=True)

    # Predict with adjusted parameters
    my_fl_labels, my_fl_details = model.predict_instances(normalize(my_fl_img), 
                                                          prob_thresh=prob_thresh,
                                                          nms_thresh=nms_thresh)

    # Size filtering
    labeled_image = measure.label(my_fl_labels)
    regions = measure.regionprops(labeled_image)
    filtered_labels = np.zeros_like(my_fl_labels)

    for region in regions:
        if min_size <= region.area <= max_size:
            filtered_labels[labeled_image == region.label] = region.label

    # Create overlay image
    overlay = color.label2rgb(filtered_labels, my_fl_img, bg_label=0, alpha=0.5)
    
    # Save overlay as PNG
    overlay_path = os.path.join(output_folder, f"overlay_{filename}")
    io.imsave(overlay_path, (overlay * 255).astype(np.uint8))

    # Save filtered prediction as binary mask
    binary_mask = (filtered_labels > 0).astype(np.uint8) * 255
    mask_path = os.path.join(output_folder, f"mask_{filename}")
    io.imsave(mask_path, binary_mask)

print("Processing complete. Results saved in:", output_folder)



#%%

# Step 6 : overlap red-blue colors - ratio (%).
# The output folder will be named as merged_Blue-Red and the excel file overlap_results.xlsx

import cv2
import numpy as np
import pandas as pd
import glob
import os
import easygui

# Function to calculate overlap ratio and create merged image
def calculate_and_merge(blue_image, red_image):
    # Convert both images to grayscale
    blue_gray = cv2.cvtColor(blue_image, cv2.COLOR_BGR2GRAY)
    red_gray = cv2.cvtColor(red_image, cv2.COLOR_BGR2GRAY)

    # Convert both masks to binary
    _, blue_binary = cv2.threshold(blue_gray, 1, 255, cv2.THRESH_BINARY)
    _, red_binary = cv2.threshold(red_gray, 1, 255, cv2.THRESH_BINARY)

    # Count non-zero pixels in both images
    red_pixels = np.count_nonzero(red_binary)

    # Calculate overlap (intersection of blue and red)
    overlap = cv2.bitwise_and(blue_binary, red_binary)
    overlap_pixels = np.count_nonzero(overlap)

    # Calculate overlap ratio as a percentage
    if red_pixels > 0:
        overlap_ratio = (overlap_pixels / red_pixels) * 100
    else:
        overlap_ratio = 0

    # Create merged image (overlay blue on red)
    merged_image = cv2.addWeighted(blue_image, 0.5, red_image, 0.5, 0)

    return overlap_ratio, merged_image

# Main script to process images
def process_images():
    # Use EasyGUI to choose folders
    blue_folder = easygui.diropenbox(msg="Choose the folder containing Blue images")
    red_folder = easygui.diropenbox(msg="Choose the folder containing Red images")

    # Use EasyGUI to choose output directory
    output_folder = easygui.diropenbox(msg="Choose the folder to save the output")

    # Create output folder for merged images inside the user-chosen folder
    merged_folder = os.path.join(output_folder, "merged_Blue-Red")
    if not os.path.exists(merged_folder):
        os.makedirs(merged_folder)

    # Excel file will be saved inside the merged folder
    output_excel = os.path.join(merged_folder, "overlap_results.xlsx")

    # Load all image pairs
    blue_images = sorted(glob.glob(os.path.join(blue_folder, "*.png")))
    red_images = sorted(glob.glob(os.path.join(red_folder, "*.png")))

    # List to store results for Excel
    results = []

    # Ensure there is a matching red image for each blue image
    for blue_image_path, red_image_path in zip(blue_images, red_images):
        # Extract image names
        red_image_name = os.path.basename(red_image_path)

        # Read images
        blue_image = cv2.imread(blue_image_path)
        red_image = cv2.imread(red_image_path)

        # Calculate overlap ratio and generate merged image
        overlap_ratio, merged_image = calculate_and_merge(blue_image, red_image)

        # Save merged image
        merged_image_path = os.path.join(merged_folder, red_image_name)
        cv2.imwrite(merged_image_path, merged_image)

        # Append result to list
        results.append({
            "Image": red_image_name,
            "Overlap Ratio (%)": overlap_ratio
        })

    # Save results to Excel inside the merged folder
    df = pd.DataFrame(results)
    df.to_excel(output_excel, index=False)
    print(f"Results saved to {output_excel}")
    print(f"Merged images saved to {merged_folder}")

# Run the script
process_images()

#%%

# Step 7: Thresholding green color and apply LUT green.
# The output folder will be named as Binary masks green.

import cv2
import os
import easygui
from tqdm import tqdm
import numpy as np

# Function to create binary mask with a given pixel range
def create_binary_mask(image, lower_threshold, upper_threshold):
    # Ensure the image is grayscale
    if len(image.shape) == 3:
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray_image = image
    
    # Apply the thresholding (range 21 to 149 in your example)
    binary_mask = cv2.inRange(gray_image, lower_threshold, upper_threshold)
    
    return binary_mask

# Function to apply green color to the binary mask
def apply_green_color(binary_mask):
    # Create a 3-channel image initialized to zero (black)
    green_image = np.zeros((binary_mask.shape[0], binary_mask.shape[1], 3), dtype=np.uint8)
    
    # Set the green channel to the binary mask values
    green_image[:, :, 1] = binary_mask  # Green channel
    
    return green_image

# Main function to process images in a folder
def process_images_in_folder():
    # Use EasyGUI to choose the input folder
    input_folder = easygui.diropenbox(title="Select Input Folder for Green Channel Images")
    if not input_folder:
        print("No folder selected. Exiting.")
        exit()

    # Create an output folder called 'Binary masks green'
    output_folder = os.path.join(os.path.dirname(input_folder), "Binary masks green")
    os.makedirs(output_folder, exist_ok=True)

    # Get a list of image files (png, jpg, jpeg, tif)
    image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff'))]

    if not image_files:
        print("No image files found in the selected folder.")
        exit()

    # Define the threshold range (21 to 149 in your case)
    lower_threshold, upper_threshold = 21, 149

    # Process each image in the folder
    for filename in tqdm(image_files, desc="Processing images"):
        # Load the image
        img_path = os.path.join(input_folder, filename)
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        if image is None:
            print(f"Failed to load image: {filename}")
            continue

        # Create a binary mask for the image
        binary_mask = create_binary_mask(image, lower_threshold, upper_threshold)

        # Apply the green color to the binary mask
        green_image = apply_green_color(binary_mask)

        # Save the green image as a new image in the output folder
        save_path = os.path.join(output_folder, f"threshold_{filename}")
        cv2.imwrite(save_path, green_image)

    print(f"Processing complete. Green-colored binary masks saved in: {output_folder}")

# Run the script
if __name__ == "__main__":
    process_images_in_folder()


#%%

# Step 8: Merge red and green channel - calculate ratio of overlapping.

import cv2
import numpy as np
import pandas as pd
import glob
import os
import easygui

# Function to calculate overlap ratio and create merged image
def calculate_and_merge(green_image, red_image):
    # Convert both images to grayscale
    green_gray = cv2.cvtColor(green_image, cv2.COLOR_BGR2GRAY)
    red_gray = cv2.cvtColor(red_image, cv2.COLOR_BGR2GRAY)

    # Convert both masks to binary
    _, green_binary = cv2.threshold(green_gray, 1, 255, cv2.THRESH_BINARY)
    _, red_binary = cv2.threshold(red_gray, 1, 255, cv2.THRESH_BINARY)

    # Count non-zero pixels in the red image
    red_pixels = np.count_nonzero(red_binary)

    # Calculate overlap (intersection of green and red)
    overlap = cv2.bitwise_and(green_binary, red_binary)
    overlap_pixels = np.count_nonzero(overlap)

    # Calculate overlap ratio as a percentage
    if red_pixels > 0:
        overlap_ratio = (overlap_pixels / red_pixels) * 100
    else:
        overlap_ratio = 0

    # Create merged image (overlay green on red)
    merged_image = cv2.addWeighted(green_image, 0.5, red_image, 0.5, 0)

    return overlap_ratio, merged_image

# Main script to process images
def process_images():
    # Use EasyGUI to choose folders
    green_folder = easygui.diropenbox(msg="Choose the folder containing Green images")
    red_folder = easygui.diropenbox(msg="Choose the folder containing Red images")

    # Use EasyGUI to choose output directory
    output_folder = easygui.diropenbox(msg="Choose the folder to save the output")

    # Create output folder for merged images inside the user-chosen folder
    merged_folder = os.path.join(output_folder, "merged_Green-Red")
    if not os.path.exists(merged_folder):
        os.makedirs(merged_folder)

    # Excel file will be saved inside the merged folder
    output_excel = os.path.join(merged_folder, "overlap_results.xlsx")

    # Load all image pairs
    green_images = sorted(glob.glob(os.path.join(green_folder, "*.png")))
    red_images = sorted(glob.glob(os.path.join(red_folder, "*.png")))

    # List to store results for Excel
    results = []

    # Ensure there is a matching red image for each green image
    for green_image_path, red_image_path in zip(green_images, red_images):
        # Extract image names
        red_image_name = os.path.basename(red_image_path)

        # Read images
        green_image = cv2.imread(green_image_path)
        red_image = cv2.imread(red_image_path)

        # Calculate overlap ratio and generate merged image
        overlap_ratio, merged_image = calculate_and_merge(green_image, red_image)

        # Save merged image
        merged_image_path = os.path.join(merged_folder, red_image_name)
        cv2.imwrite(merged_image_path, merged_image)

        # Append result to list
        results.append({
            "Image": red_image_name,
            "Overlap Ratio (%)": overlap_ratio
        })

    # Save results to Excel inside the merged folder
    df = pd.DataFrame(results)
    df.to_excel(output_excel, index=False)
    print(f"Results saved to {output_excel}")
    print(f"Merged images saved to {merged_folder}")

# Run the script
process_images()

#%%

# Step 9 :  Yellow and red overlap excluding_green
# The blue color in these images actually represents the yellow color (CY5). Why did I do this? 
#For an important reason: The overlap between green and red colors creates yellow. 
#To prevent confusion between the yellow created by this overlap and the original yellow (CY5), I decided to display the CY5 in blue.
    
import cv2
import numpy as np
import os
import easygui

# Use EasyGUI to select the input folder
input_folder = easygui.diropenbox("Select the folder containing binary mask images")

# Create an output folder
output_folder = os.path.join(input_folder, "blue_lut_output")
os.makedirs(output_folder, exist_ok=True)

# Create a blue LUT
blue_lut = np.zeros((256, 1, 3), dtype=np.uint8)
blue_lut[:, 0, 0] = np.arange(256)  # Blue channel (BGR order in OpenCV)

# Process each image in the input folder
for filename in os.listdir(input_folder):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
        # Read the image
        img_path = os.path.join(input_folder, filename)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        
        # Ensure the image is binary
        _, img_binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
        
        # Convert to 3-channel image
        img_3channel = cv2.cvtColor(img_binary, cv2.COLOR_GRAY2BGR)
        
        # Apply the blue LUT
        blue_img = cv2.LUT(img_3channel, blue_lut)
        
        # Save the result
        output_path = os.path.join(output_folder, f"blue_{filename}")
        cv2.imwrite(output_path, blue_img)

print(f"Processing complete. Blue LUT images saved in: {output_folder}")


#%%

# Step 10 : Yellow & Red overlap_excluding_green
    
import cv2
import numpy as np
import pandas as pd
import glob
import os
import easygui

def calculate_and_merge(blue_image, red_image, green_image):
    # Convert images to grayscale
    blue_gray = cv2.cvtColor(blue_image, cv2.COLOR_BGR2GRAY)
    red_gray = cv2.cvtColor(red_image, cv2.COLOR_BGR2GRAY)
    green_gray = cv2.cvtColor(green_image, cv2.COLOR_BGR2GRAY)

    # Convert masks to binary
    _, blue_binary = cv2.threshold(blue_gray, 1, 255, cv2.THRESH_BINARY)
    _, red_binary = cv2.threshold(red_gray, 1, 255, cv2.THRESH_BINARY)
    _, green_binary = cv2.threshold(green_gray, 1, 255, cv2.THRESH_BINARY)

    # Count non-zero pixels in the red image
    red_pixels = np.count_nonzero(red_binary)

    # Calculate overlap between blue and red, excluding green
    overlap = cv2.bitwise_and(blue_binary, red_binary)
    overlap = cv2.bitwise_and(overlap, cv2.bitwise_not(green_binary))
    overlap_pixels = np.count_nonzero(overlap)

    # Calculate overlap ratio as a percentage
    if red_pixels > 0:
        overlap_ratio = (overlap_pixels / red_pixels) * 100
    else:
        overlap_ratio = 0

    # Create merged image (overlay blue, red, and green)
    merged_image = cv2.addWeighted(blue_image, 0.33, red_image, 0.33, 0)
    merged_image = cv2.addWeighted(merged_image, 1, green_image, 0.33, 0)

    return overlap_ratio, merged_image

def process_images():
    # Use EasyGUI to choose folders
    blue_folder = easygui.diropenbox(msg="Choose the folder containing Blue images")
    red_folder = easygui.diropenbox(msg="Choose the folder containing Red images")
    green_folder = easygui.diropenbox(msg="Choose the folder containing Green images")
    output_folder = easygui.diropenbox(msg="Choose the folder to save the output")

    # Create output folder for merged images
    merged_folder = os.path.join(output_folder, "merged_Blue-Red-Green")
    if not os.path.exists(merged_folder):
        os.makedirs(merged_folder)

    # Excel file path
    output_excel = os.path.join(merged_folder, "overlap_results_Blue-Red.xlsx")

    # Load all image pairs
    blue_images = sorted(glob.glob(os.path.join(blue_folder, "*.png")))
    red_images = sorted(glob.glob(os.path.join(red_folder, "*.png")))
    green_images = sorted(glob.glob(os.path.join(green_folder, "*.png")))

    results = []

    # Process each set of images
    for blue_image_path, red_image_path, green_image_path in zip(blue_images, red_images, green_images):
        # Extract image names
        image_name = os.path.basename(blue_image_path)

        # Read images
        blue_image = cv2.imread(blue_image_path)
        red_image = cv2.imread(red_image_path)
        green_image = cv2.imread(green_image_path)

        # Calculate overlap ratio and generate merged image
        overlap_ratio, merged_image = calculate_and_merge(blue_image, red_image, green_image)

        # Save merged image
        merged_image_path = os.path.join(merged_folder, image_name)
        cv2.imwrite(merged_image_path, merged_image)

        # Append result to list
        results.append({
            "Image": image_name,
            "Overlap Ratio yellow-Red (%)-excluding green": overlap_ratio
        })

    # Save results to Excel
    df = pd.DataFrame(results)
    df.to_excel(output_excel, index=False)
    print(f"Results saved to {output_excel}")
    print(f"Merged images saved to {merged_folder}")

# Run the script
process_images()














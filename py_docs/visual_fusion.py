import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import morphology, exposure
from skimage.filters import gaussian
from skimage.morphology import disk, binary_closing, binary_opening
from scipy.ndimage import binary_fill_holes

from helper import load_and_select_reference, interactive_crop, avg_intensity, OUTPUT_FOLDER

## Checking Reference Frame ##

print("Loading stack and selecting reference frame...")
R_stack, G1_stack, G2_stack, B_stack, ref_idx = load_and_select_reference(apply_white_balance=True)

R_ref  = R_stack[ref_idx]
G1_ref = G1_stack[ref_idx]
G2_ref = G2_stack[ref_idx]
B_ref  = B_stack[ref_idx]
G_ref = (G1_ref + G2_ref) / 2

# Visualization 
plt.figure(figsize=(5, 5))
plt.imshow(R_ref, cmap='gray')
plt.title(f"Chosen Reference")
plt.axis("off")
plt.show()

## SIFT Feautre Matching ##

# Creating rgb Reference image for sift and filtering pixels
ref_intensity = avg_intensity(R_ref, G1_ref, G2_ref, B_ref) # creating image that takes average intensity from each pixel
ref_gray = (ref_intensity / (np.percentile(ref_intensity, 99) + 1e-6) * 255).astype(np.uint8) # Converting to grayscale and converting to 8 bit image
valid = ((ref_intensity > np.percentile(ref_intensity,5)) & (ref_intensity < np.percentile(ref_intensity,99.5))) # ensures only pixels brighter that lowest 5% and darker than brightest 0.5% are used for features
ref_gray[~valid] = 0 # all invalid pixels are 0

# Setting Up SIFT
sift = cv2.SIFT_create(nfeatures=5000, edgeThreshold=10) # Sets up 3000 key points
kp_ref, des_ref = sift.detectAndCompute(ref_gray, None) # defining corrdinates for key points descriptive vectors
FLANN_INDEX_KDTREE = 1 # algorium used
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5) # Builts KDTREE Index (change # of tree to adjust accruacy and speed)
search_params = dict(checks=50) # checks up to 50 key point and descriptors
matcher = cv2.FlannBasedMatcher(index_params, search_params) # Matches points and vectors

# Holding appendix for each channel starting with ref as 0
R_aligned  = [R_ref]
G1_aligned = [G1_ref]
G2_aligned = [G2_ref]
B_aligned  = [B_ref]

for i in range(len(R_stack)):
    if i == ref_idx: # if image is ref skip
        continue
    # if it is not ref then pick image channels
    R  = R_stack[i]
    G1 = G1_stack[i]
    G2 = G2_stack[i]
    B  = B_stack[i]

    I = avg_intensity(R, G1, G2, B) # find average intensity
    gray = (I / (np.percentile(I, 99) + 1e-6) * 255).astype(np.uint8) # converts to gray for SIFT
    valid = ((I > np.percentile(I,5)) &(I < np.percentile(I,99.5))) # filtering pixels the same way as reference
    gray[~valid] = 0 # setting invaid pixels to 0
    
    kp, des = sift.detectAndCompute(gray, None) # key points and descriptors for current frame

    # if no frames detected then skip
    if des is None or des_ref is None:
        print(f"Frame {i}: skipped (no features)")
        continue

    matches = matcher.knnMatch(des_ref, des, k=2) # check closest desciptors to reference from current frame
    
    # keeping only best matches
    good = []
    for m,n in matches:
        if m.distance < 0.7 * n.distance:
            good.append(m)
    
    # frame has 5 or less matches then bad frame
    if len(good) < 10:
        print(f"Frame {i}: not enough matches")
        continue

    src_pts = np.float32([kp_ref[m.queryIdx].pt for m in good]).reshape(-1,2) # reference frame coordinates
    dst_pts = np.float32([kp[m.trainIdx].pt for m in good]).reshape(-1,2) # current frame coordinates

    M, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, ransacReprojThreshold=3.0)

    if M is None:
        print(f"Frame {i}: Homography failed")
        continue

# Apply the perspective warp
    shape = R.shape
    R_w  = cv2.warpPerspective(R, M, (shape[1], shape[0]))
    G1_w = cv2.warpPerspective(G1, M, (shape[1], shape[0]))
    G2_w = cv2.warpPerspective(G2, M, (shape[1], shape[0]))
    B_w  = cv2.warpPerspective(B, M, (shape[1], shape[0]))
    
    # Saving for each frame
    R_aligned.append(R_w)
    G1_aligned.append(G1_w)
    G2_aligned.append(G2_w)
    B_aligned.append(B_w)

    print(f"Frame {i} aligned with {np.sum(mask)} features")

# Stacking aligned channels
R_aligned = np.stack(R_aligned)
G1_aligned = np.stack(G1_aligned)
G2_aligned = np.stack(G2_aligned)
B_aligned = np.stack(B_aligned)

# Cropping
ymin, ymax, xmin, xmax =interactive_crop(R_aligned[0])

# Checking coordinates
print(f"Cropping Coordinates -> Min = {ymin}: {xmin}, Max = {ymax}: {xmax}")

# Cropping All images and channels
R_crop = R_aligned[:, ymin:ymax, xmin:xmax]
G1_crop = G1_aligned[:, ymin:ymax, xmin:xmax]
G2_crop = G2_aligned[:, ymin:ymax, xmin:xmax]
B_crop = B_aligned[:, ymin:ymax, xmin:xmax]

# Visualize the cropped reference frame
plt.figure(figsize=(5, 5))
plt.imshow(R_crop[0], cmap='gray')
plt.title(f"Cropped Reference")
plt.axis("off")
plt.show()

# Stacking all new cropped channels
channel_crop_stack = np.stack([R_crop, G1_crop, G2_crop,B_crop], axis=-1)

## Identifying Lesion ##

# 1. Get grayscale and inverted blue intensity for the interior
ref_gray = np.mean(channel_crop_stack[0, ..., :3], axis=-1)
blue = channel_crop_stack[0, ..., 3]
inverted_blue_norm = cv2.normalize(1 - blue, None, 0.0, 1.0, cv2.NORM_MINMAX)

# 2. Compute Sobel structural edges
sobelx = cv2.Sobel(ref_gray, cv2.CV_32F, 1, 0, ksize=3)
sobely = cv2.Sobel(ref_gray, cv2.CV_32F, 0, 1, ksize=3)
structural_map = np.sqrt(sobelx**2 + sobely**2)
structural_weight = cv2.normalize(structural_map, None, 0.0, 1.0, cv2.NORM_MINMAX)

# 3. Combine edges AND intensity so the whole inside is seeded
# This highlights the edges AND keeps the dark body of the mole
combined_map = inverted_blue_norm + (structural_weight * 1.5)

# 4. Threshold the combined map to get a solid body
core_thresh = np.percentile(combined_map, 75) # Lower threshold to capture full interior
lesion = combined_map > core_thresh

# 5. Clean and fill solidly
lesion = morphology.remove_small_objects(lesion, min_size=500)
lesion = morphology.binary_closing(lesion, morphology.disk(10)) # Seals any minor gaps
lesion = binary_fill_holes(lesion) # Now that it's fully sealed, this will solid-fill the interior
lesion = morphology.binary_erosion(lesion, morphology.disk(5))

# Visualization
plt.figure(figsize=(10, 4)) 
plt.subplot(1, 3, 1) 
plt.imshow(channel_crop_stack[0][..., :3]) 
plt.title("Reference Image") 
plt.axis("off")

plt.subplot(1, 3, 2) 
plt.imshow(inverted_blue_norm, cmap='magma') 
plt.title("B-channel Map") 
plt.axis("off")

plt.subplot(1, 3, 3) 
plt.imshow(inverted_blue_norm, cmap='magma') # Background image
# Overlay the boolean mask with transparency (alpha=0.4) and a distinct color map
plt.imshow(lesion, cmap='autumn', alpha=0.2) 
plt.title("Mask Overlay") 
plt.axis("off")

plt.tight_layout() 
plt.show()

## Glare Detection ##

# Glare and shadow Thresholds
SKIN_GLARE_FACTOR = 1.5
LESION_GLARE_FACTOR = 1.2
SHADOW_FACTOR = 0.65

img_gray = channel_crop_stack[0].mean(axis=-1) # Converting to grayscale
local_baseline = gaussian(img_gray, sigma=15) # gaussian blur to detect just lighting changes

skin_glare = (img_gray > (local_baseline * SKIN_GLARE_FACTOR)) # Glare on all pixels 
lesion_glare = (img_gray > (local_baseline * LESION_GLARE_FACTOR)) & lesion # glare on Lesion pixels
glare_mask = binary_closing(skin_glare | lesion_glare, disk(8)) # smoothing with pixel radius of 8
#glare_mask = binary_closing(skin_glare, disk(8)) # smoothing with pixel radius of 8
shadow_mask = (img_gray < (local_baseline * SHADOW_FACTOR)) # shadow on all pixels

# shadow smoothing
shadow_mask = binary_opening(shadow_mask, disk(2))
shadow_mask = binary_closing(shadow_mask, disk(5))

# glare and shodow are comained and smoothed
raw_artifacts_mask = binary_closing(glare_mask | shadow_mask, disk(5)) # glare and shadow combined into mask
master_patch_mask = raw_artifacts_mask & (~lesion) # lesion eliminated from mask

patched_stack = channel_crop_stack[0].copy() # making copy to do patches

# Setting Regions
background_patch_mask = (raw_artifacts_mask) & (~lesion) # Background
lesion_patch_mask = lesion_glare # Lesion

# Pixel picking 
# replacing glare pixels (replacing buffer of 3 pixels inside lesion)
best_frame_indices = np.argmin(channel_crop_stack.mean(axis=-1), axis=0) # finding pixels with lowest brightness on a pixel basis
dilated_lesion_mask = morphology.binary_dilation(lesion_patch_mask, morphology.disk(3))
# Pixel Patching Loop
for y in range(patched_stack.shape[0]):
    for x in range(patched_stack.shape[1]):
        if background_patch_mask[y, x]: # For background
            frame_idx = best_frame_indices[y, x] # replace with better pixel in background
            patched_stack[y, x] = channel_crop_stack[frame_idx, y, x] # copy info inot current stack
            
        elif dilated_lesion_mask[y, x]: # For Lesion 
            frame_idx = best_frame_indices[y, x]
            patched_stack[y, x] = channel_crop_stack[frame_idx, y, x]

# Rebuilding channels (final channels)
R = patched_stack[..., 0]
G1 = patched_stack[..., 1]
G2 = patched_stack[..., 2]
B = patched_stack[..., 3]


# Visulaizing
G_combined = (G1 + G2) / 2.0
rgb_manual = np.stack([R, G_combined, B], axis=-1) # RGB stack
clean_mask = ~master_patch_mask # Using Clean pixles for white balance

# White balance
avg_r = np.mean(rgb_manual[clean_mask, 0])
avg_g = np.mean(rgb_manual[clean_mask, 1])
avg_b = np.mean(rgb_manual[clean_mask, 2])
avg_gray = (avg_r + avg_g + avg_b) / 3
rgb_balanced = rgb_manual.copy()
rgb_balanced[:,:,0] *= (avg_gray / avg_r)
rgb_balanced[:,:,1] *= (avg_gray / avg_g)
rgb_balanced[:,:,2] *= (avg_gray / avg_b)

# Contrast ( 1> more contrast, 1 < less contrast)
gamma = 1.1
rgb_final = exposure.adjust_gamma(rgb_balanced, gamma=gamma) # applying contrast
rgb_final = np.clip(rgb_final / np.percentile(rgb_final, 98), 0, 1) # final Normalizing

# Plotting Background mask ( Lesion is )
plt.figure(figsize=(10, 5))

plt.subplot(1, 3, 1)
plt.imshow(rgb_final)
plt.title("Glare Free Image")

plt.subplot(1, 3, 2)
plt.imshow(master_patch_mask, cmap='gray')
plt.title("Background Mask")

plt.show()

## Comparing to Reference ##

# Color correcting Reference cropped 
ref_crop_raw = channel_crop_stack[0]
R = ref_crop_raw[..., 0]
G1 = ref_crop_raw[..., 1]
G2 = ref_crop_raw[..., 2]
B = ref_crop_raw[..., 3]
G = (G1 + G2) / 2.0
rgb = np.stack([R, G, B], axis=-1)
    
avg_r = np.mean(rgb[:,:,0])
avg_g = np.mean(rgb[:,:,1])
avg_b = np.mean(rgb[:,:,2])
avg_gray = (avg_r + avg_g + avg_b) / 3
    
rgb_balanced = rgb.copy()
rgb_balanced[:,:,0] *= (avg_gray / avg_r)
rgb_balanced[:,:,1] *= (avg_gray / avg_g)
rgb_balanced[:,:,2] *= (avg_gray / avg_b)
    
rgb_og_cropped = np.clip(rgb_balanced / np.percentile(rgb_balanced, 98), 0, 1)

# Plotting Reconstructed vs Reference
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(rgb_og_cropped)
plt.title("Reference Image")
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(rgb_final)
plt.title("Glare-free Image")
plt.axis("off")

#plt.savefig(r"C:\Users\hbvis\SkinSavvy\fusion_images\glare_comparison4.jpg", format='jpg', dpi=300, bbox_inches='tight')
plt.show()

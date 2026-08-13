# helper.py
import os
from pathlib import Path
import numpy as np
import rawpy
import matplotlib.pyplot as plt

OUTPUT_FOLDER = r"C:\Users\hbvis\SkinSavvy\fusion_images"
INPUT_FOLDER = r"C:\Users\hbvis\SkinSavvy\raw_images"

def setup_environment():
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

def avg_intensity(R, G1, G2, B):
    return (R + G1 + G2 + B) / 4.0

def color_variance(R, G1, G2, B):
    eps = 1e-6
    mu = (R + G1 + G2 + B) / 4.0
    var = ((R - mu)**2 + (G1 - mu)**2 + (G2 - mu)**2 + (B - mu)**2) / 4.0
    return var / (mu**2 + eps)

def load_and_select_reference(apply_white_balance=True):

    setup_environment()
    folder = Path(INPUT_FOLDER)
    files = sorted(list(folder.glob("*.dng")))
    
    if not files:
        raise FileNotFoundError(f"No .dng files found in {INPUT_FOLDER}")

    R_imgs, G1_imgs, G2_imgs, B_imgs = [], [], [], []
    
    for f in files:
        with rawpy.imread(str(f)) as raw:
            sensor = raw.raw_image_visible.astype(np.float32)
            black = raw.black_level_per_channel

            R_bayer  = sensor[0::2,0::2]
            G1_bayer = sensor[0::2,1::2]
            G2_bayer = sensor[1::2,0::2]
            B_bayer  = sensor[1::2,1::2]

            if apply_white_balance:
                b_val = black[0]
                white = float(raw.white_level)

                R = np.clip(R_bayer - black[0], 0, None)
                G1 = np.clip(G1_bayer - black[1], 0, None)
                G2 = np.clip(G2_bayer - black[2], 0, None)
                B = np.clip(B_bayer - black[3], 0, None)
                
                # Normalizing by white level   
                R /= white
                G1 /= white
                G2 /= white
                B /= white
                
            else:

                R  = R_bayer - black[0]
                G1 = G1_bayer - black[1]
                G2 = G2_bayer - black[2]
                B  = B_bayer - black[3]
            
            R_imgs.append(R)
            G1_imgs.append(G1)
            G2_imgs.append(G2)
            B_imgs.append(B)
            
    R_stack = np.stack(R_imgs)
    G1_stack = np.stack(G1_imgs)
    G2_stack = np.stack(G2_imgs)
    B_stack = np.stack(B_imgs)

    # Automatically compute glare scores to find best reference frame
    glare_scores = []
    for i in range(len(R_stack)):
        I = avg_intensity(R_stack[i], G1_stack[i], G2_stack[i], B_stack[i])
        C = color_variance(R_stack[i], G1_stack[i], G2_stack[i], B_stack[i])
        glare_mask = (C < 0.05) & (I > np.percentile(I, 95))
        glare_scores.append(np.sum(glare_mask))

    ref_idx = np.argmin(glare_scores)
    print(f"Helper selected Reference Frame index: {ref_idx}")

    return R_stack, G1_stack, G2_stack, B_stack, ref_idx

def interactive_crop(img_reference):
    coords = []
    def onclick(event):
        if event.xdata is None or event.ydata is None:
            return
        coords.append((int(event.xdata), int(event.ydata)))
        if len(coords) == 2:
            plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(img_reference, cmap='gray')
    ax.set_title("Click Top-Left and Bottom-Right corners to crop")
    cid = fig.canvas.mpl_connect('button_press_event', onclick)
    plt.show()

    if len(coords) < 2:
        raise ValueError("Cropping aborted: You must click twice on the image window.")

    y1, x1 = coords[0][1], coords[0][0]
    y2, x2 = coords[1][1], coords[1][0]
    ymin, ymax = sorted([y1, y2])
    xmin, xmax = sorted([x1, x2])
    
    return ymin, ymax, xmin, xmax
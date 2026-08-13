# spectral.py
import os
import numpy as np
import matplotlib.pyplot as plt
from helper import load_and_select_reference, OUTPUT_FOLDER

R_stack, G1_stack, G2_stack, B_stack, ref_idx = load_and_select_reference(apply_white_balance=False)

R_ref  = R_stack[ref_idx]
G_ref  = (G1_stack[ref_idx] + G2_stack[ref_idx]) / 2.0
B_ref  = B_stack[ref_idx]

# ratios
rg = R_ref / (G_ref + 1e-6) # getting dividing by zero issues
rb = R_ref - B_ref

# Limits outliers (otherwise no detail)
rg_vmin, rg_vmax = np.percentile(rg,[2, 98])
rb_vmin, rb_vmax = np.percentile(rb, [2, 98])

# Plotting
fig, axes = plt.subplots(1, 2, figsize=(10, 4))

im0 = axes[0].imshow(rg, cmap='magma', vmin=rg_vmin, vmax=rg_vmax)
axes[0].set_title("Red / Green")
axes[0].axis("off")
fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

im1 = axes[1].imshow(rb, cmap='magma', vmin=rb_vmin, vmax=rb_vmax)
axes[1].set_title("Red-Blue")
axes[1].axis("off")
fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
plt.tight_layout()
plt.show()
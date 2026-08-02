# Hybrid Data Generation Pipeline: StyleGAN2-ADA + CutPaste

This guide details the strategy for generating synthetic, perfectly labeled defective images for the Himalaya label inspection project. 

Given the small dataset (especially only ~20 defective samples per ROI), pure generative models struggle to create specific, controlled defects. This hybrid approach solves that by splitting the problem: generating infinite clean backgrounds first, then programmatically adding real defects.

---

## Stage 1: Generate Clean "Good" Labels (StyleGAN2-ADA)

**Status:** *Current Priority. Must be completed before proceeding to Stage 2.*

**Objective:** Train StyleGAN2-ADA to understand the distribution of normal labels (lighting variations, slight positional shifts, print texture) and generate endless variations of perfect labels.

### 1. Data Preparation
*   **Source:** Your ~80 real "good" ROI crops (e.g., `ROI_1` logo area).
*   **Formatting:** StyleGAN requires square images and specific dataset formats. You must pad or resize your crops to a target resolution (e.g., 256x256 or 512x512) and convert them to the `.zip` format expected by the StyleGAN2-ADA training script.

### 2. Training on RTX A5000
*   **Repository:** Clone the official [StyleGAN2-ADA PyTorch repository](https://github.com/NVlabs/stylegan2-ada-pytorch).
*   **Transfer Learning:** Because 80 images is extremely small, you **must** use transfer learning. Initialize your training using a pre-trained network (e.g., `ffhq256` or `metfaces`).
*   **Expected Time:** Training at 256x256 on an RTX A5000 should take roughly 1.5 to 3 hours to converge adequately.
*   **Monitoring:** Monitor the FID (Fréchet Inception Distance) score and visually inspect the generated grids during training. Stop when the generated images look indistinguishable from real good labels.

### 3. Generation & Curation
*   **Sampling:** Use the trained network to generate a large batch (e.g., 5,000) of synthetic "good" label crops.
*   **Filtering:** Review the generated images. Discard any that have strange artifacts or Mode Collapse (where the network generates the exact same image repeatedly). 

---

## Stage 2: Defect Extraction and Injection (CutPaste)

**Status:** *Pending completion of Stage 1. Can be executed iteratively.*

**Objective:** Create diverse, realistic defective images by pasting real defect textures onto the synthetic clean backgrounds generated in Stage 1. This provides automatic, pixel-perfect ground truth labels.

### Phase 2A: Defect Extraction (Masking)
1.  Take your real defective crops (~20 images).
2.  Carefully annotate the exact defect (ink blob, smudge, scratch, missing print) using a tool like Label Studio or even Photoshop.
3.  Extract the defect region, creating a "patch" with an alpha channel (transparent background) representing just the anomaly.

### Phase 2B: Data Augmentation on Patches
To maximize the utility of your 20 patches, apply augmentations before pasting:
*   Random rotations (if applicable to the defect type).
*   Random scaling (making the blob slightly larger or smaller).
*   Slight color/brightness jittering.
*   Elastic deformations (stretching or morphing the shape slightly).

### Phase 2C: Seamless Blending (Poisson Editing)
1.  Randomly select one synthetic "good" background from Stage 1.
2.  Randomly select one augmented defect patch.
3.  Choose a random, valid location on the background to place the defect.
4.  **Crucial Step:** Do not just hard-paste the patch. Use **Poisson Image Editing** (available in OpenCV as `cv2.seamlessClone`). This blends the gradients of the defect patch into the background, matching local lighting and texture so the injection looks natural, not like a sticker.

### Phase 2D: Automatic Annotation Generation
Because you defined the exact coordinates and mask of the defect when pasting it onto the synthetic background, you automatically have perfect bounding box coordinates and pixel-level segmentation masks for every generated image. Save these programmatically in YOLO or COCO format.


Best Hybrid Approach for Generating Defective Images:
Use StyleGAN2-ADA + CutPaste / Defect Blending (Poisson / Copy-Paste Augmentation):

1.Use StyleGAN2-ADA to generate endless realistic GOOD background label variations (good text, clean colors).
2.Extract defect patches (ink blobs, smudges, scratches, tears) from your 20 real defective crops.
Blend / Paste those defect patches randomly (or using real defect masks) onto the StyleGAN2-generated clean labels.
This gives you thousands of realistic, perfectly labeled synthetic BAD images instantly.

This is the idea that i want to acheive this in stages 
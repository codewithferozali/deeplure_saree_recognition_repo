# Kaggle Execution Guide

1. **Upload**: Go to Kaggle, click "New Notebook", go to File -> Import Notebook and upload `deeplure_saree_recognition.ipynb`.
2. **Dataset**: Click "Add Data" -> Search for your dataset (or create a new dataset with the images) and attach it to the notebook. Update `DATA_ROOT` in Section 5 to the actual input path (e.g., `/kaggle/input/my-saree-dataset`).
3. **GPU**: In the notebook settings on Kaggle, set the Accelerator to **GPU T4 x2** or **P100**.
4. **Run**: Go to Run -> Run All.
5. **Expected Output**: The notebook will load images, perform metric learning training, and calculate Recall@K and verification metrics automatically. Checkpoints will be saved to `/kaggle/working/`.

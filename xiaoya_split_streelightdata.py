# import torch
# from mamba_ssm import Mamba

# batch, length, dim = 2, 64, 16
# x = torch.randn(batch, length, dim).to("cuda")
# model = Mamba(
#     # This module uses roughly 3 * expand * d_model^2 parameters
#     d_model=dim, # Model dimension d_model
#     d_state=16,  # SSM state expansion factor
#     d_conv=4,    # Local convolution width
#     expand=2,    # Block expansion factor
# ).to("cuda")
# y = model(x)
# assert y.shape == x.shape

# from causal_conv1d import causal_conv1d_fn

#split streelight_18k into train/val/test
# from sklearn.model_selection import train_test_split
# import pandas as pd

# # Read the data.txt file into a DataFrame
# df = pd.read_csv("/home/sci/mitra/Desktop/data_18k_streetlights/data_.txt", sep=" ", header=None, names=["filename", "label"])

# # Convert label to integer (just to be safe)
# df["label"] = df["label"].astype(int)

# # Optional: confirm it's working
# print(df.head())
# print(df['label'].value_counts())


# # First, split train (75%) and temp (25%)
# train_df, temp_df = train_test_split(
#     df,
#     stratify=df['label'],
#     test_size=0.25,
#     random_state=42
# )

# # Then split temp into val (10%) and test (15%)
# val_df, test_df = train_test_split(
#     temp_df,
#     stratify=temp_df['label'],
#     test_size=0.6,
#     random_state=42
# )

# # Check class distribution
# print("Train:", train_df['label'].value_counts())
# print("Val:", val_df['label'].value_counts())
# print("Test:", test_df['label'].value_counts())

# # import shutil
# # import os
# # from pathlib import Path

# # # Define your target root directory
# # output_root = '/home/collab/u1368791/Vim/data/streetlights'  # current directory

# # # Define split-to-folder mapping
# # split_map = {
# #     "train": train_df,
# #     "val": val_df,
# #     "test": test_df
# # }

# # for split_name, split_df in split_map.items():
# #     split_dir = os.path.join(output_root,split_name)
# #     Path(split_dir).mkdir(parents=True, exist_ok=True) 

# #     for src_path in split_df['filename']:
# #         src = Path(src_path)
# #         dst = os.path.join(split_dir, src.name)  # copy with same filename
# #         try:
# #             shutil.copy(src, dst)
# #         except Exception as e:
# #             print(f"❌ Failed to copy {src} → {dst}: {e}")

# import shutil
# from pathlib import Path
# import os 
# # Your root directory (current working directory)
# output_root = '/home/collab/u1368791/Vim/data/streetlights'

# # Map split names to their corresponding DataFrames
# split_map = {
#     "train": train_df,
#     "val": val_df,
#     "test": test_df
# }

# for split_name, df in split_map.items():
#     split_root = os.path.join(output_root, split_name)
#     for _, row in df.iterrows():
#         filename = Path(row["filename"]).name
#         label = str(row["label"])
        
#         src = os.path.join(split_root, filename)
#         label_dir = os.path.join(split_root, label)
#         Path(label_dir).mkdir(parents=True, exist_ok=True)
        
#         dst = os.path.join(label_dir, filename)
        
#         # Move file to label subdirectory
#         try:
#             shutil.move(str(src), str(dst))
#             print(f"✅ Succeeded to move {src} → {dst}")
#         except Exception as e:
#             print(f"❌ Failed to move {src} → {dst}: {e}")

from pathlib import Path

root = Path('/home/collab/u1368791/Vim/data/streetlights')

splits = ['train', 'val', 'test']
labels = ['0', '1']

for split in splits:
    print(f"\n📁 Split: {split}")
    for label in labels:
        folder = root / split / label
        count = len(list(folder.glob('*.jpg')))
        print(f"  Label {label}: {count} images")

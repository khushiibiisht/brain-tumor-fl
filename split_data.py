# split_data.py
# Splits the training dataset into N separate portions
# simulating N different hospitals each with their own data
# 
# Real world: Hospital A has its own MRI scanner and patients
#             Hospital B has different patients, never shared
# We simulate this by splitting our dataset into N parts

import os
import shutil
import random
from pathlib import Path


def split_dataset(
    source_dir,
    output_dir,
    num_clients=3,
    seed=42
):
    """
    Splits a dataset into num_clients portions.

    source_dir  — original Training folder
    output_dir  — where to save the split data
    num_clients — number of hospital clients (we use 3)
    seed        — random seed for reproducibility
                  (same seed = same split every time)
    """

    random.seed(seed)

    # Get all class names (glioma, meningioma, etc.)
    classes = [
        d for d in os.listdir(source_dir)
        if os.path.isdir(os.path.join(source_dir, d))
    ]
    print(f'Classes found: {classes}')
    print(f'Splitting into {num_clients} client portions...\n')

    # Create output folders for each client
    # structure: split_data/client_0/glioma/
    #                       client_0/meningioma/ etc.
    for i in range(num_clients):
        for cls in classes:
            Path(f'{output_dir}/client_{i}/{cls}').mkdir(
                parents=True, exist_ok=True
            )

    # Split each class separately
    # This ensures each client gets images from ALL classes
    # (important — we don't want client_0 to only see glioma)
    for cls in classes:
        cls_dir = os.path.join(source_dir, cls)

        # Get all image files in this class
        images = [
            f for f in os.listdir(cls_dir)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ]

        # Shuffle so split is random
        random.shuffle(images)

        # Split images evenly across clients
        splits = _split_list(images, num_clients)

        for i, client_images in enumerate(splits):
            dst_dir = f'{output_dir}/client_{i}/{cls}'
            for img in client_images:
                src = os.path.join(cls_dir, img)
                dst = os.path.join(dst_dir, img)
                shutil.copy2(src, dst)

            print(
                f'  client_{i} | {cls:12s} | '
                f'{len(client_images)} images'
            )
        print()

    print('Split complete!')
    print(f'Data saved to: {output_dir}')


def _split_list(lst, n):
    """Splits a list into n roughly equal parts."""
    k, m = divmod(len(lst), n)
    return [
        lst[i*k + min(i, m):(i+1)*k + min(i+1, m)]
        for i in range(n)
    ]


def verify_split(output_dir, num_clients=3):
    """
    Prints a summary of how many images each client has.
    Call this after splitting to verify it worked.
    """
    print('\n--- Split Verification ---')
    for i in range(num_clients):
        client_dir = f'{output_dir}/client_{i}'
        total = 0
        classes = sorted(os.listdir(client_dir))
        print(f'\nclient_{i}:')
        for cls in classes:
            cls_path = os.path.join(client_dir, cls)
            count = len(os.listdir(cls_path))
            total += count
            print(f'  {cls:12s}: {count} images')
        print(f'  {"TOTAL":12s}: {total} images')


# ─────────────────────────────────────────
# Run this file directly to perform the split
# python split_data.py
# ─────────────────────────────────────────
if __name__ == '__main__':
    split_dataset(
        source_dir='data/Training',
        output_dir='data/split',
        num_clients=3,
        seed=42
    )
    verify_split('data/split', num_clients=3)
import cv2
import numpy as np
from pathlib import Path
import shutil
from tqdm import tqdm


def apply_laplacian_sharpening(image, strength=1.5):
    laplacian = cv2.Laplacian(image, cv2.CV_64F)
    laplacian = np.uint8(np.absolute(laplacian))
    sharpened = cv2.addWeighted(image, 1.0, laplacian, strength, 0)
    return sharpened


def process_uwcd_dataset(input_dir, output_dir, strength=1.5):
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    print("=" * 70)
    print("UWCD Dataset Sharpening (Laplacian)")
    print("=" * 70)
    print(f"Input: {input_dir}")
    print(f"Output: {output_dir}")
    print(f"Strength: {strength}")
    print()
    
    if not input_path.exists():
        print(f"Error: Input path not found: {input_path}")
        return
    
    if output_path.exists():
        response = input(f"Output folder exists: {output_path}\nDelete and recreate? (y/n): ")
        if response.lower() == 'y':
            shutil.rmtree(output_path)
        else:
            print("Cancelled.")
            return
    
    output_path.mkdir(parents=True, exist_ok=True)
    
    total_processed = 0
    total_failed = 0
    
    has_split = (input_path / 'train').exists() and (input_path / 'val').exists()
    splits = ['train', 'val'] if has_split else ['.']
    
    print(f"Structure: {'Train/Val' if has_split else 'Single folder'}\n")
    
    for split in splits:
        if split == '.':
            split_input = input_path
            split_output = output_path
        else:
            split_input = input_path / split
            split_output = output_path / split
            
            if not split_input.exists():
                continue
            
            print(f"\n{'='*70}")
            print(f"{split.upper()}")
            print('='*70)
        
        category_dirs = [d for d in split_input.iterdir() if d.is_dir()]
        
        for category_dir in category_dirs:
            category_name = category_dir.name
            output_category = split_output / category_name
            output_category.mkdir(parents=True, exist_ok=True)
            
            images = (list(category_dir.glob('*.jpg')) + 
                     list(category_dir.glob('*.png')) + 
                     list(category_dir.glob('*.jpeg')))
            
            if not images:
                continue
            
            print(f"\n{category_name}: {len(images):,} images")
            
            failed_count = 0
            for img_path in tqdm(images, desc="Processing", unit="img"):
                try:
                    img = cv2.imread(str(img_path))
                    
                    if img is None:
                        failed_count += 1
                        continue
                    
                    sharpened = apply_laplacian_sharpening(img, strength=strength)
                    output_file = output_category / img_path.name
                    cv2.imwrite(str(output_file), sharpened)
                    
                    total_processed += 1
                    
                except Exception as e:
                    failed_count += 1
                    total_failed += 1
            
            if failed_count > 0:
                print(f"  Failed: {failed_count}")
    
    print("\n" + "=" * 70)
    print("Completed!")
    print("=" * 70)
    print(f"Processed: {total_processed:,} images")
    if total_failed > 0:
        print(f"Failed: {total_failed:,} images")
    print(f"\nOutput: {output_path.absolute()}")
    print()


def main():
    input_dir = '/mnt/sagemaker-nvme/uwcd_5class_split'
    output_dir = '/mnt/sagemaker-nvme/uwcd_5class_split_sharpened'
    
    strength = 1.5
    
    print(f"Parameters: strength={strength}\n")
    
    process_uwcd_dataset(input_dir, output_dir, strength)
    
    output_path = Path(output_dir)
    if output_path.exists():
        print("\nDataset structure:")
        print("-" * 70)
        
        has_split = (output_path / 'train').exists()
        
        if has_split:
            for split in ['train', 'val']:
                split_path = output_path / split
                if split_path.exists():
                    print(f"\n{split.upper()}:")
                    for category in sorted(split_path.iterdir()):
                        if category.is_dir():
                            count = len(list(category.glob('*.jpg')))
                            print(f"  {category.name:20s}: {count:6,d} images")
        else:
            for category in sorted(output_path.iterdir()):
                if category.is_dir():
                    count = len(list(category.glob('*.jpg')))
                    print(f"  {category.name:20s}: {count:6,d} images")
        
        print()


if __name__ == '__main__':
    main()


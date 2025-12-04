from ultralytics import YOLO
import torch
from pathlib import Path
import shutil
import os

print("=" * 60)
print("UWCD 5-Class Waste Classification Training")
print("Unified Waste Classification Dataset → 5 Classes")
print("=" * 60)

# GPU 확인
print(f"\nCUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"   GPU count: {torch.cuda.device_count()}")
    print(f"   GPU name: {torch.cuda.get_device_name(0)}")
    print(f"   Using GPU device: 0")
else:
    print("   Using CPU (Warning: Training will be slow)")

# 모델 크기 선택
model_size = 'n'  # n, s, m, l, x 중 선택
print(f"\nModel: yolov8{model_size}-cls.pt")

# 데이터셋 경로 (8-class 원본)
source_data_path = '/mnt/sagemaker-nvme/uwcd_dataset/content/unified_dataset'
# 5-class 변환 경로
target_data_path = '/mnt/sagemaker-nvme/uwcd_5class'

print(f"\nSource (8-class): {source_data_path}")
print(f"Target (5-class): {target_data_path}")
print()

# 클래스 매핑: 8-class → 5-class
class_mapping = {
    'plastic': 'plastic',
    'glass': 'glass',
    'metal': 'metal',
    'paper_cardboard': 'paper',
    'battery': 'trash',
    'organic_waste': 'trash',
    'textiles': 'trash',
    'trash': 'trash'
}

print("📊 Class Mapping (8 → 5):")
print("-" * 60)
for old_cls, new_cls in class_mapping.items():
    print(f"   {old_cls:20s} → {new_cls}")
print()

# 5-class 데이터셋 생성
print("🔄 Creating 5-class dataset...")
print("-" * 60)

if os.path.exists(target_data_path):
    print(f"⚠️  Target directory already exists: {target_data_path}")
    response = input("Delete and recreate? (y/n): ")
    if response.lower() == 'y':
        shutil.rmtree(target_data_path)
    else:
        print("Using existing dataset")

if not os.path.exists(target_data_path):
    # 5개 클래스 폴더 생성
    for cls in ['plastic', 'glass', 'metal', 'paper', 'trash']:
        os.makedirs(os.path.join(target_data_path, cls), exist_ok=True)
    
    # 이미지 복사 및 클래스 병합
    source_path = Path(source_data_path)
    target_path = Path(target_data_path)
    
    stats = {cls: 0 for cls in ['plastic', 'glass', 'metal', 'paper', 'trash']}
    
    for old_class_dir in source_path.iterdir():
        if old_class_dir.is_dir():
            old_class = old_class_dir.name
            new_class = class_mapping.get(old_class, 'trash')
            
            print(f"Processing {old_class:20s} → {new_class:10s}", end=" ")
            
            target_class_dir = target_path / new_class
            count = 0
            
            for img_file in old_class_dir.glob("*.jpg"):
                # 새 파일명: 원본 클래스명 포함하여 중복 방지
                new_filename = f"{old_class}_{img_file.name}"
                target_file = target_class_dir / new_filename
                shutil.copy2(img_file, target_file)
                count += 1
            
            stats[new_class] += count
            print(f"({count:6d} images)")
    
    print()
    print("📊 5-Class Dataset Statistics:")
    print("-" * 60)
    total = 0
    for cls, count in sorted(stats.items()):
        total += count
        print(f"   {cls:10s}: {count:6d} images ({count/total*100:5.1f}%)")
    print("-" * 60)
    print(f"   {'TOTAL':10s}: {total:6d} images")
    print()

# 모델 초기화
model = YOLO(f'yolov8{model_size}-cls.pt')

# 학습 설정
print("⚙️  Training Configuration:")
print("-" * 60)

config = {
    'data': target_data_path,
    'epochs': 50,
    'imgsz': 224,
    'batch': 256,
    'patience': 15,
    'cache': False,            # RAM 캐시 비활성화 (메모리 부족 방지)
    'device': 0,
    'workers': 2,              # 워커 수 감소 (8 → 2, shared memory 문제 방지)
    'amp': True,
    'close_mosaic': 0,
    'pretrained': True,
    'optimizer': 'AdamW',
    'lr0': 0.001,
    'weight_decay': 0.0005,
    'warmup_epochs': 5,
    'project': 'runs/classify',
    'name': 'uwcd_5class',
    'exist_ok': True,
    'verbose': True,
}

for key, value in config.items():
    print(f"{key:20s}: {value}")
print()

# 학습 시작
print("🚀 Training started...")
print("=" * 60)
print()

results = model.train(**config)

print("\n" + "=" * 60)
print("✅ Training completed!")
print("=" * 60)
print(f"\n📊 Results:")
print(f"   Best model: {results.save_dir}/weights/best.pt")
print(f"   Last model: {results.save_dir}/weights/last.pt")
print(f"   Results: {results.save_dir}")
print()

# 검증
print("🔍 Validation on test set...")
metrics = model.val()
print(f"\n   Top-1 Accuracy: {metrics.top1:.2%}")
print(f"   Top-5 Accuracy: {metrics.top5:.2%}")

print("\n" + "=" * 60)
print("💡 Next Steps:")
print("=" * 60)
print("1. Download best.pt model")
print("2. Test with recycle_classification_5class.py")
print("3. Compare with previous 5-class model")
print()

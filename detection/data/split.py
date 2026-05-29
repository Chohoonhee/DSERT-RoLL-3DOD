import os
import random

# 두 개의 디렉토리 경로 지정
dir1 = "./processed_data/Night"
dir2 = "./processed_data/Normal"

# 결과 저장 파일 이름
train_file = "train.txt"
val_file = "val.txt"

# 폴더 이름 수집
def collect_folders(*dirs):
    folders = []
    for d in dirs:
        for item in os.listdir(d):
            path = os.path.join(d, item)
            if os.path.isdir(path) and (item.startswith("2025_04") or item.startswith("2025_06")):
                # import pdb; pdb.set_trace()
                folders.append(d.split('/')[-1] + '/' + item)
    return folders

# 폴더 리스트 수집
all_folders = collect_folders(dir1, dir2)


# 랜덤 셔플
random.shuffle(all_folders)

# 7:3으로 나누기
split_idx = int(len(all_folders) * 0.7)
train_folders = all_folders[:split_idx]
val_folders = all_folders[split_idx:]

# 파일 저장
with open(train_file, "w") as f:
    for name in train_folders:
        f.write(name + "\n")

with open(val_file, "w") as f:
    for name in val_folders:
        f.write(name + "\n")

print(f"총 {len(all_folders)}개 폴더 중 {len(train_folders)}개는 train.txt, {len(val_folders)}개는 val.txt에 저장되었습니다.")

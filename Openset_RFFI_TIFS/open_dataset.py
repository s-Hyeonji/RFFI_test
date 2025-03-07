import h5py
import numpy as np

# HDF5 파일 열기
file_path = './dataset/Test/channel_problem/A.h5'  # 여기에 파일 경로를 입력하세요.
with h5py.File(file_path, "r") as f:
    # 데이터셋 내부 구조 확인
    print("Keys in the HDF5 file:", list(f.keys()))
    
    # 'data'와 'label' 불러오기
    data = f["data"][:]  # (N, 16384) 형태일 가능성이 큼
    labels = f["label"][:]
    cfo = f["CFO"][:]
    rss = f["RSS"][:]

# 데이터 형태 출력
print("Data shape:", data.shape)  # 예상: (패킷 수, 16384)
print("Label shape:", labels.shape)  # 예상: (패킷 수,)
print("CFO shape:", cfo.shape)  # 예상: (패킷 수,)
print("RSS shape:", rss.shape)  # 예상: (패킷 수,)


print("\nFirst 5 CFO values:", cfo[:5])
print("First 5 RSS values:", rss[:5])

# 첫 번째 샘플 확인
print("First sample data:", data[0])
print("First sample label:", labels[0])

# I/Q 분리
I_branch = data[:, :8192]
Q_branch = data[:, 8192:]

print("I-branch shape:", I_branch.shape)
print("Q-branch shape:", Q_branch.shape)

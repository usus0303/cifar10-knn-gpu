# -*- coding: utf-8 -*-
"""CIFAR-10 최종 Test 혼동행렬만 생성 (5-fold 재실행 없음).

사용법: 이 파일을 knn_cifar10_gpu.py, cifar-10-python.tar.gz와 같은 폴더에 두고
       py plot_confusion_only.py

이미 results_full_gpu/test_예측상세.csv가 있으면 GPU 계산 없이 바로 그립니다.
없으면 기존 KNN 코드의 함수로 최종 Test 10,000장만 GPU에서 다시 예측합니다.
"""

import csv  # 기존 예측 결과 CSV를 읽고 새 결과를 저장합니다.
from pathlib import Path  # VS Code의 현재 폴더와 관계없이 파일 경로를 찾습니다.
import numpy as np  # 혼동행렬의 숫자를 계산합니다.
import matplotlib.pyplot as plt  # 혼동행렬을 이미지로 그립니다.

# 1. 실험에서 사용한 경로와 클래스 이름을 설정합니다.
HERE = Path(__file__).resolve().parent  # 이 파이썬 파일이 있는 위치입니다.
OUT = HERE / "results_full_gpu"  # 기존 GPU KNN 실험의 결과 폴더입니다.
OUT.mkdir(exist_ok=True)  # 결과 폴더가 없다면 만듭니다.
PRED_CSV = OUT / "test_예측상세.csv"  # 이미지별 실제/예측 라벨을 저장할 파일입니다.
NAMES = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]  # CIFAR-10 클래스입니다.

# 2. 예측 결과 CSV가 이미 있는 경우에는 그대로 읽습니다.
if PRED_CSV.exists():  # 10,000장에 대한 이전 예측 결과가 있는지 확인합니다.
    with PRED_CSV.open("r", encoding="utf-8-sig", newline="") as f:  # CSV를 엽니다.
        rows = list(csv.DictReader(f))  # 각 행에서 실제 정답과 예측값을 읽습니다.
    y_true = np.array([int(row["true_label"]) for row in rows])  # 실제 클래스 번호 배열입니다.
    y_pred = np.array([int(row["pred_label"]) for row in rows])  # 예측 클래스 번호 배열입니다.
    print("저장된 예측 CSV를 불러왔습니다. (KNN 재실행 없음)")  # 데이터 재사용을 안내합니다.

# 3. 예측 CSV가 없으면 5-fold 없이 최종 테스트 예측만 한 번 수행합니다.
else:  # 이전 코드에서 이미지별 예측을 저장하지 않은 경우입니다.
    import torch  # GPU에 전체 학습 데이터를 올리기 위해 사용합니다.
    from knn_cifar10_gpu import load_cifar10, knn_predict  # 이전 GPU KNN 코드에 이미 있는 함수를 재사용합니다.
    if not torch.cuda.is_available():  # GPU 사용 가능 여부를 확인합니다.
        raise RuntimeError("NVIDIA CUDA GPU가 인식되지 않습니다. PyTorch CUDA 설치 상태를 확인하세요.")
    x_train, y_train, x_test, y_true = load_cifar10()  # 학습 50,000장과 Test 10,000장을 불러옵니다.
    train_gpu = torch.as_tensor(x_train, device="cuda", dtype=torch.float32) / 255.0  # 학습 픽셀을 GPU에 올립니다.
    labels_gpu = torch.as_tensor(y_train, device="cuda", dtype=torch.long)  # 학습 정답도 GPU에 올립니다.
    print("5-fold 생략: Manhattan 거리, k=1로 Test 10,000장만 예측합니다.")  # 수행 범위를 알립니다.
    y_pred = knn_predict(train_gpu, labels_gpu, x_test, "manhattan", [1])[1]  # 검증 결과에서 최적이었던 조합으로 예측합니다.
    with PRED_CSV.open("w", encoding="utf-8-sig", newline="") as f:  # 예측값을 다음에도 사용할 수 있도록 저장합니다.
        writer = csv.writer(f)  # CSV 작성 도구를 준비합니다.
        writer.writerow(["sample_index", "true_label", "pred_label"])  # CSV 열 이름을 기록합니다.
        writer.writerows((i, int(a), int(p)) for i, (a, p) in enumerate(zip(y_true, y_pred)))  # 이미지별 정답/예측을 기록합니다.
    print("테스트 예측 저장:", PRED_CSV)  # 저장 위치를 알려 줍니다.

# 4. 실제 정답과 예측 번호로 10×10 혼동행렬을 계산합니다.
if len(y_true) != 10000 or len(y_pred) != 10000:  # 실제 CIFAR-10 Test 이미지 10,000장이 맞는지 확인합니다.
    raise ValueError("테스트 예측이 정확히 10,000개가 아닙니다. CSV 내용을 확인하세요.")
cm = np.zeros((10, 10), dtype=int)  # 행=실제 클래스, 열=예측 클래스인 빈 행렬입니다.
np.add.at(cm, (y_true, y_pred), 1)  # 실제/예측 조합이 나타날 때마다 해당 칸에 1을 더합니다.
acc = np.trace(cm) / cm.sum()  # 대각선 합계를 전체 수로 나누어 Accuracy를 구합니다.
print(f"최종 Test Accuracy: {acc * 100:.2f}%")  # 기존 실행 결과의 38.59%와 비교할 수 있습니다.
np.savetxt(OUT / "confusion_counts.csv", cm, delimiter=",", fmt="%d")  # 원본 개수 혼동행렬도 CSV로 저장합니다.

# 5. 원본 개수(Counts)와 행별 정규화 백분율(Percent) 두 그림을 그립니다.
row_sums = cm.sum(axis=1, keepdims=True)  # 실제 클래스별 이미지 수를 계산합니다.
cm_percent = np.divide(cm * 100.0, row_sums, out=np.zeros_like(cm, dtype=float), where=row_sums != 0)  # 클래스별 예측 비율입니다.
for values, filename, title, is_percent in [  # 같은 혼동행렬을 두 방식으로 표시합니다.
    (cm, "confusion_counts.png", "CIFAR-10 KNN Confusion Matrix (Counts)", False),  # 실제 이미지 수입니다.
    (cm_percent, "confusion_percent.png", "CIFAR-10 KNN Confusion Matrix (%)", True),  # 행별 백분율입니다.
]:
    fig, ax = plt.subplots(figsize=(10, 8))  # 그래프 영역을 만듭니다.
    image = ax.imshow(values, cmap="Blues")  # 값이 클수록 진한 파란색으로 표시합니다.
    fig.colorbar(image, ax=ax)  # 오른쪽에 색상 범례를 표시합니다.
    ax.set_xticks(range(10), labels=NAMES, rotation=45, ha="right")  # 가로축에 예측 클래스 이름을 적습니다.
    ax.set_yticks(range(10), labels=NAMES)  # 세로축에 실제 클래스 이름을 적습니다.
    ax.set_xlabel("Predicted Label")  # 가로축은 KNN이 예측한 클래스입니다.
    ax.set_ylabel("True Label")  # 세로축은 실제 정답 클래스입니다.
    ax.set_title(title)  # 그래프 제목을 표시합니다.
    for i in range(10):  # 실제 클래스 10개를 반복합니다.
        for j in range(10):  # 예측 클래스 10개를 반복합니다.
            label = f"{values[i, j]:.1f}%" if is_percent else str(values[i, j])  # 수치를 % 또는 개수로 표시합니다.
            color = "white" if values[i, j] > values.max() / 2 else "black"  # 진한 칸은 흰 글자로 표시합니다.
            ax.text(j, i, label, ha="center", va="center", fontsize=8, color=color)  # 칸 중앙에 수치를 표시합니다.
    fig.tight_layout()  # 축 이름이 잘리지 않게 정렬합니다.
    fig.savefig(OUT / filename, dpi=180)  # PNG 이미지로 저장합니다.
    print("저장 완료:", OUT / filename)  # 생성된 파일 경로를 출력합니다.

plt.show()  # VS Code에서 두 그래프를 화면에 표시합니다.

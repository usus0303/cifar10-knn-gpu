# -*- coding: utf-8 -*-
"""CIFAR-10 전체 60,000장을 사용한 GPU KNN / 5-fold 교차검증 (한글 주석 버전)."""

import argparse  # 실행 옵션(--gpu-check, --self-test)을 읽습니다.
import csv  # 실험 결과를 CSV 파일로 저장합니다.
import json  # 재실행 시 설정이 같은지 확인합니다.
import pickle  # CIFAR-10 배치 파일의 데이터를 읽습니다.
import tarfile  # CIFAR-10 .tar.gz 압축파일을 바로 읽습니다.
import time  # 실험 진행 시간을 확인합니다.
from pathlib import Path  # Windows 폴더 경로를 안전하게 다룹니다.

import matplotlib  # 그래프를 파일로 저장하는 라이브러리입니다.
matplotlib.use("Agg")  # VS Code에서 그래프 창 없이 PNG 파일을 만듭니다.
import matplotlib.pyplot as plt  # 검증 결과를 그래프로 그립니다.
import numpy as np  # 이미지와 정답 데이터를 배열로 처리합니다.
import torch  # GPU(CUDA)에서 이미지 간 거리를 계산합니다.
import torch.nn.functional as F  # 코사인 정규화와 클래스별 다수결에 사용합니다.
from sklearn.model_selection import StratifiedKFold  # 클래스 비율을 유지하는 5-fold 분할입니다.

# ==================== 1. 실험 설정 ====================
BASE_DIR = Path(__file__).resolve().parent  # 이 코드가 있는 폴더를 기준으로 합니다.
DATA_PATH = BASE_DIR / "cifar-10-python.tar.gz"  # CIFAR-10 압축파일 경로입니다.
RESULT_DIR = BASE_DIR / "results_full_gpu"  # GPU 실험 결과를 저장할 폴더입니다.
K_VALUES = [1, 3, 5, 7, 9]  # 비교할 최근접 이웃의 개수입니다.
DISTANCES = ["euclidean", "manhattan", "cosine"]  # 비교할 세 가지 거리 방식입니다.
CLASS_COUNT = 10  # CIFAR-10 클래스는 10개입니다.
CLASS_NAMES = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]  # 혼동행렬의 10개 클래스 이름입니다.
FOLD_COUNT = 5  # 전체 학습 데이터를 5개의 Fold로 나눕니다.
SEED = 42  # 다시 실행해도 동일한 Fold가 되도록 설정합니다.
TRAIN_CHUNK = 4096  # 거리 계산 때 학습 이미지 4,096장씩 처리합니다.
QUERY_BATCH = {"euclidean": 128, "manhattan": 8, "cosine": 128}  # 거리별 GPU 검증 묶음 크기입니다.
CSV_HEADER = ["fold", "distance", "k", "accuracy", "precision", "recall", "f1"]  # 결과표 열 이름입니다.

# ==================== 2. CIFAR-10의 배치 파일 6개 불러오기 ====================
def read_batch(tar, filename):  # 압축파일 내부에서 배치 하나를 읽는 함수입니다.
    member = tar.extractfile(f"cifar-10-batches-py/{filename}")  # 원하는 배치 파일을 찾습니다.
    if member is None:  # 찾으려는 배치가 없는지 확인합니다.
        raise FileNotFoundError(f"압축파일 안에 {filename}이 없습니다.")  # 누락된 파일명을 알려 줍니다.
    with member as file:  # 열린 배치를 안전하게 닫을 수 있도록 처리합니다.
        return pickle.load(file, encoding="bytes")  # 픽셀 배열과 정답 라벨을 읽습니다.


def load_cifar10():  # 학습 5배치 + 테스트 1배치를 모두 읽는 함수입니다.
    train_images = []  # 학습 배치의 이미지들을 저장합니다.
    train_labels = []  # 학습 배치의 정답들을 저장합니다.
    with tarfile.open(DATA_PATH, "r:gz") as tar:  # 압축 해제 없이 tar.gz를 엽니다.
        for batch_no in range(1, 6):  # data_batch_1부터 data_batch_5까지 읽습니다.
            batch = read_batch(tar, f"data_batch_{batch_no}")  # 학습 배치를 하나 읽습니다.
            train_images.append(batch[b"data"])  # 이미지 10,000장을 목록에 추가합니다.
            train_labels.extend(batch[b"labels"])  # 각 이미지의 정답 10,000개를 추가합니다.
            print(f"data_batch_{batch_no}: {len(batch[b'data']):,}장 읽기 완료", flush=True)  # 불러온 장수를 출력합니다.
        test_batch = read_batch(tar, "test_batch")  # 별도 테스트 데이터 10,000장을 읽습니다.
        print(f"test_batch: {len(test_batch[b'data']):,}장 읽기 완료", flush=True)  # 테스트 배치 수를 출력합니다.
    x_train = np.concatenate(train_images, axis=0)  # 학습 배치 5개를 합쳐 50,000장으로 만듭니다.
    y_train = np.array(train_labels, dtype=np.int64)  # 학습 정답을 숫자 배열로 만듭니다.
    x_test = test_batch[b"data"]  # 테스트 이미지 10,000장을 가져옵니다.
    y_test = np.array(test_batch[b"labels"], dtype=np.int64)  # 테스트 정답을 가져옵니다.
    assert x_train.shape == (50000, 3072)  # 학습 이미지 50,000장과 픽셀 3,072개를 확인합니다.
    assert x_test.shape == (10000, 3072)  # 테스트 이미지 10,000장을 확인합니다.
    return x_train, y_train, x_test, y_test  # 이미지·정답 데이터를 반환합니다.


# ==================== 3. GPU에서 사용할 세 거리 계산 수식 ====================
# Euclidean(L2) = sqrt(Σ(x_i - y_i)^2)  # 픽셀별 차이를 제곱해 더한 뒤 제곱근을 구합니다.
def euclidean_distance(query, train):  # 두 이미지 묶음의 Euclidean 거리를 구합니다.
    query_sq = (query * query).sum(dim=1, keepdim=True)  # 검증 이미지의 제곱합을 구합니다.
    train_sq = (train * train).sum(dim=1).unsqueeze(0)  # 학습 이미지의 제곱합을 구합니다.
    squared = query_sq + train_sq - 2.0 * (query @ train.T)  # 전개한 거리 공식으로 빠르게 계산합니다.
    return torch.sqrt(torch.clamp(squared, min=0.0))  # 음수 오차를 없애고 제곱근을 취합니다.


# Manhattan(L1) = Σ|x_i - y_i|  # 픽셀 차이의 절댓값을 모두 더합니다.
def manhattan_distance(query, train):  # 두 이미지 묶음의 Manhattan 거리를 구합니다.
    return torch.cdist(query, train, p=1)  # GPU의 L1 거리 연산을 사용합니다.


# Cosine Distance = 1 - (x·y) / (||x|| ||y||)  # 이미지 벡터의 방향 차이를 측정합니다.
def cosine_distance(query, normalized_train):  # 정규화된 학습 데이터와 Cosine 거리를 구합니다.
    normalized_query = F.normalize(query, p=2, dim=1)  # 검증 이미지의 벡터 길이를 1로 만듭니다.
    return 1.0 - (normalized_query @ normalized_train.T)  # 1 - 코사인 유사도를 반환합니다.


# ==================== 4. GPU에서 직접 구현한 KNN ====================
@torch.no_grad()  # KNN은 신경망 학습이 아니므로 기울기를 계산하지 않습니다.
def knn_predict(train_gpu, train_labels_gpu, query_cpu, distance_name, k_values):  # 예측을 실행하는 함수입니다.
    max_k = max(k_values)  # 가장 큰 k까지만 이웃을 찾으면 다른 k에도 재사용할 수 있습니다.
    predictions = {k: np.empty(len(query_cpu), dtype=np.uint8) for k in k_values}  # k별 예측 결과를 준비합니다.
    batch_size = QUERY_BATCH[distance_name]  # 거리 방식에 알맞은 검증 이미지 묶음 크기를 사용합니다.
    total_batches = (len(query_cpu) + batch_size - 1) // batch_size  # 전체 반복 횟수를 계산합니다.
    started = time.perf_counter()  # 진행 시간을 측정합니다.
    if distance_name == "cosine":  # Cosine 거리를 사용할 때만 실행합니다.
        normalized_train = F.normalize(train_gpu, p=2, dim=1)  # 학습 이미지의 벡터 길이를 한 번만 정규화합니다.
    else:  # Euclidean이나 Manhattan에서는 정규화된 배열이 필요하지 않습니다.
        normalized_train = None  # 불필요한 GPU 메모리를 할당하지 않습니다.

    for batch_no, query_start in enumerate(range(0, len(query_cpu), batch_size), start=1):  # 검증 이미지를 나누어 반복합니다.
        query_end = min(query_start + batch_size, len(query_cpu))  # 현재 묶음의 마지막 위치를 구합니다.
        query = torch.as_tensor(query_cpu[query_start:query_end], device=train_gpu.device, dtype=torch.float32) / 255.0  # 검증 픽셀을 학습 텐서와 같은 장치로 옮깁니다.
        best_distances = None  # 현재까지 찾은 가장 가까운 이웃들의 거리를 저장합니다.
        best_indices = None  # 현재까지 찾은 이웃의 학습 데이터 번호를 저장합니다.

        for train_start in range(0, len(train_gpu), TRAIN_CHUNK):  # 전체 40,000장(테스트 시 50,000장)을 나누어 비교합니다.
            train_end = min(train_start + TRAIN_CHUNK, len(train_gpu))  # 현재 학습 묶음의 끝을 정합니다.
            train_part = train_gpu[train_start:train_end]  # 비교할 학습 이미지를 선택합니다.
            if distance_name == "euclidean":  # L2를 선택한 경우입니다.
                distances = euclidean_distance(query, train_part)  # Euclidean 거리 행렬을 구합니다.
            elif distance_name == "manhattan":  # L1을 선택한 경우입니다.
                distances = manhattan_distance(query, train_part)  # Manhattan 거리 행렬을 구합니다.
            elif distance_name == "cosine":  # Cosine을 선택한 경우입니다.
                distances = cosine_distance(query, normalized_train[train_start:train_end])  # Cosine 거리를 구합니다.
            else:  # 잘못된 거리 이름을 입력한 경우입니다.
                raise ValueError(f"알 수 없는 거리 방식: {distance_name}")  # 거리 이름을 확인하라는 오류입니다.

            nearest_dist, nearest_local = torch.topk(distances, min(max_k, train_end - train_start), dim=1, largest=False, sorted=True)  # 묶음에 이미지가 k개보다 적으면 있는 개수만 선택합니다.
            nearest_global = nearest_local + train_start  # 묶음 안의 번호를 전체 학습 데이터 번호로 바꿉니다.
            if best_distances is None:  # 첫 학습 묶음인 경우입니다.
                best_distances = nearest_dist  # 현재 묶음의 k개 이웃을 저장합니다.
                best_indices = nearest_global  # 해당 이웃들의 번호를 저장합니다.
            else:  # 이전 묶음에서도 이웃 후보를 찾은 경우입니다.
                all_distances = torch.cat([best_distances, nearest_dist], dim=1)  # 이전과 현재 묶음의 후보 거리를 합칩니다.
                all_indices = torch.cat([best_indices, nearest_global], dim=1)  # 후보 이미지 번호도 합칩니다.
                best_distances, chosen = torch.topk(all_distances, max_k, dim=1, largest=False, sorted=True)  # 합친 후보 중 가까운 k개를 다시 고릅니다.
                best_indices = all_indices.gather(1, chosen)  # 최종적으로 선택된 이웃 번호를 가져옵니다.

        nearest_labels = train_labels_gpu[best_indices]  # 최종 이웃의 정답 클래스를 가져옵니다.
        one_hot = F.one_hot(nearest_labels.long(), num_classes=CLASS_COUNT)  # 10개 클래스별 표 형태로 바꿉니다.
        votes = one_hot.cumsum(dim=1)  # 1번째부터 k번째까지 클래스별 득표수를 누적합니다.
        for k in k_values:  # k=1,3,5,7,9를 각각 확인합니다.
            pred = votes[:, k - 1, :].argmax(dim=1)  # 해당 k에서 가장 표를 많이 얻은 클래스를 선택합니다.
            predictions[k][query_start:query_end] = pred.cpu().numpy().astype(np.uint8)  # 예측을 CPU 결과 배열에 저장합니다.
        if batch_no % max(1, total_batches // 10) == 0 or batch_no == total_batches:  # 약 10%마다 진행률을 출력합니다.
            print(f"    {distance_name}: {query_end:,}/{len(query_cpu):,}장 ({time.perf_counter() - started:.0f}초)", flush=True)  # 처리량을 출력합니다.

    del normalized_train  # Cosine용 추가 GPU 텐서를 정리합니다.
    return predictions  # k 값별 예측 결과 전체를 반환합니다.


# ==================== 5. 네 평가 지표를 수식대로 계산 ====================
def evaluate(y_true, y_pred):  # Accuracy, Precision, Recall, F1을 직접 계산합니다.
    accuracy = float(np.mean(y_true == y_pred))  # 정확도 = 맞은 예측 수 / 전체 이미지 수입니다.
    precision_list, recall_list, f1_list = [], [], []  # 클래스별 평가 결과를 담습니다.
    for cls in range(CLASS_COUNT):  # 10개 클래스 각각을 평가합니다.
        tp = np.sum((y_true == cls) & (y_pred == cls))  # TP: 해당 클래스를 정확하게 예측한 수입니다.
        fp = np.sum((y_true != cls) & (y_pred == cls))  # FP: 다른 클래스를 해당 클래스로 잘못 예측한 수입니다.
        fn = np.sum((y_true == cls) & (y_pred != cls))  # FN: 해당 클래스를 다른 클래스로 잘못 예측한 수입니다.
        precision = tp / (tp + fp) if tp + fp > 0 else 0.0  # Precision = TP / (TP + FP)입니다.
        recall = tp / (tp + fn) if tp + fn > 0 else 0.0  # Recall = TP / (TP + FN)입니다.
        f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0  # F1 = 2PR / (P + R)입니다.
        precision_list.append(precision)  # 현재 클래스의 정밀도를 저장합니다.
        recall_list.append(recall)  # 현재 클래스의 재현율을 저장합니다.
        f1_list.append(f1)  # 현재 클래스의 F1 점수를 저장합니다.
    return [accuracy, float(np.mean(precision_list)), float(np.mean(recall_list)), float(np.mean(f1_list))]  # Macro 평균 지표를 반환합니다.


# ==================== 6. 결과 CSV 저장 및 이어서 실행 ====================
def save_csv(path, header, rows):  # CSV 파일을 저장하는 함수입니다.
    with path.open("w", newline="", encoding="utf-8-sig") as file:  # 엑셀에서 한글이 깨지지 않도록 엽니다.
        writer = csv.writer(file)  # CSV 작성기를 만듭니다.
        writer.writerow(header)  # 열 이름을 저장합니다.
        writer.writerows(rows)  # 결과를 모두 저장합니다.


def read_saved_rows(path):  # 이전에 성공적으로 계산한 Fold 결과를 읽습니다.
    if not path.exists():  # 저장된 CSV가 없으면 새 실험입니다.
        return []  # 빈 결과 목록을 반환합니다.
    with path.open("r", newline="", encoding="utf-8-sig") as file:  # 기존 CSV를 읽습니다.
        return [[int(r["fold"]), r["distance"], int(r["k"]), float(r["accuracy"]),  # Fold, 거리, k, Accuracy를 읽습니다.
                 float(r["precision"]), float(r["recall"]), float(r["f1"])]  # Precision, Recall, F1도 읽습니다.
                for r in csv.DictReader(file)]  # 파일에 있는 결과 행을 순회합니다.


# ==================== 7. 최종 테스트의 Confusion Matrix 그리기 ====================
def make_confusion_matrix(y_true, y_pred):  # 실제 정답과 예측으로 10×10 혼동행렬을 만듭니다.
    matrix = np.zeros((CLASS_COUNT, CLASS_COUNT), dtype=np.int64)  # 행=실제 클래스, 열=예측 클래스로 초기화합니다.
    for actual, predicted in zip(y_true, y_pred):  # 테스트 이미지의 정답과 예측을 한 쌍씩 읽습니다.
        matrix[int(actual), int(predicted)] += 1  # 해당 (실제, 예측) 칸의 이미지 수를 1 증가시킵니다.
    return matrix  # 전체 10개 클래스의 혼동행렬을 반환합니다.


def save_confusion_matrix(y_true, y_pred):  # 혼동행렬 CSV와 PNG 이미지를 저장합니다.
    matrix = make_confusion_matrix(y_true, y_pred)  # 실제 정답과 예측으로 혼동행렬을 계산합니다.
    totals = matrix.sum(axis=1, keepdims=True)  # 실제 클래스마다 이미지가 몇 장인지 계산합니다.
    percent = np.divide(matrix * 100.0, totals, out=np.zeros_like(matrix, dtype=float), where=totals != 0)  # 행별 백분율을 구합니다.
    title_header = ["Actual / Predicted"] + CLASS_NAMES  # CSV 파일의 첫 행에 예측 클래스 이름을 적습니다.
    count_rows = [[CLASS_NAMES[i]] + matrix[i].tolist() for i in range(CLASS_COUNT)]  # 실제 클래스별 예측 개수를 표로 만듭니다.
    percent_rows = [[CLASS_NAMES[i]] + [round(x, 2) for x in percent[i]] for i in range(CLASS_COUNT)]  # 백분율 표를 만듭니다.
    save_csv(RESULT_DIR / "test_confusion_matrix_counts.csv", title_header, count_rows)  # 실제 이미지 개수를 CSV에 저장합니다.
    save_csv(RESULT_DIR / "test_confusion_matrix_percent.csv", title_header, percent_rows)  # 클래스별 비율을 CSV에 저장합니다.

    for data, file_name, chart_title, format_text in [  # 이미지 2개를 각각 저장하기 위해 설정합니다.
        (matrix, "test_confusion_matrix_counts.png", "CIFAR-10 Test Confusion Matrix (Counts)", "count"),  # 실제 이미지 개수입니다.
        (percent, "test_confusion_matrix_percent.png", "CIFAR-10 Test Confusion Matrix (Row %)", "percent"),  # 행별 백분율입니다.
    ]:  # 두 종류의 혼동행렬을 차례로 그립니다.
        fig, ax = plt.subplots(figsize=(10, 8))  # 이미지와 그래프 영역을 만듭니다.
        image = ax.imshow(data, cmap="Blues")  # 숫자가 클수록 진한 파란색으로 표시합니다.
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)  # 오른쪽에 색상 범례를 추가합니다.
        ax.set_xticks(range(CLASS_COUNT), labels=CLASS_NAMES, rotation=45, ha="right")  # 가로축에 예측 클래스 이름을 씁니다.
        ax.set_yticks(range(CLASS_COUNT), labels=CLASS_NAMES)  # 세로축에 실제 클래스 이름을 씁니다.
        ax.set_xlabel("Predicted label")  # 가로축은 모델이 예측한 클래스입니다.
        ax.set_ylabel("True label")  # 세로축은 실제 정답 클래스입니다.
        ax.set_title(chart_title)  # 그래프의 제목을 표시합니다.
        threshold = float(data.max()) / 2.0  # 숫자가 잘 보이도록 글자 색을 정할 기준입니다.
        for row in range(CLASS_COUNT):  # 혼동행렬의 10개 행을 순회합니다.
            for col in range(CLASS_COUNT):  # 각 행의 10개 열을 순회합니다.
                number = f"{int(data[row, col])}" if format_text == "count" else f"{data[row, col]:.1f}%"  # 개수 또는 %로 표시합니다.
                color = "white" if data[row, col] > threshold else "black"  # 진한 배경에서는 흰 글자를 사용합니다.
                ax.text(col, row, number, ha="center", va="center", fontsize=8, color=color)  # 각 칸에 수치를 적습니다.
        fig.tight_layout()  # 클래스 이름이 잘리지 않도록 정렬합니다.
        fig.savefig(RESULT_DIR / file_name, dpi=180, bbox_inches="tight")  # 완성된 혼동행렬 이미지를 저장합니다.
        plt.close(fig)  # GPU 실험 메모리를 아끼기 위해 그래프를 닫습니다.

    print("혼동행렬 저장 완료: test_confusion_matrix_counts.png / test_confusion_matrix_percent.png", flush=True)  # 저장된 파일을 안내합니다.
    print(f"혼동행렬 정답 합계: {int(np.trace(matrix)):,} / {int(matrix.sum()):,}장", flush=True)  # 대각선 합계로 맞힌 이미지 수를 확인합니다.


def plot_saved_predictions():  # 이미 저장된 예측 CSV로 혼동행렬만 다시 그립니다.
    prediction_path = RESULT_DIR / "test_예측상세.csv"  # GPU 실험 종료 시 저장되는 예측 CSV 파일입니다.
    if not prediction_path.exists():  # 예측 CSV가 없다면 혼동행렬을 복원할 수 없습니다.
        raise FileNotFoundError(f"{prediction_path} 파일이 없습니다. 먼저 전체 GPU 실험을 완료하세요.")  # 안내 메시지를 보여줍니다.
    with prediction_path.open("r", newline="", encoding="utf-8-sig") as file:  # 예측 결과 CSV를 엽니다.
        rows = list(csv.DictReader(file))  # 전체 테스트 이미지의 예측값을 읽습니다.
    y_true = np.array([int(row["true_label"]) for row in rows])  # CSV에서 실제 정답만 추출합니다.
    y_pred = np.array([int(row["pred_label"]) for row in rows])  # CSV에서 모델 예측만 추출합니다.
    RESULT_DIR.mkdir(exist_ok=True)  # 저장할 폴더가 있는지 확인합니다.
    save_confusion_matrix(y_true, y_pred)  # GPU를 다시 실행하지 않고 그림만 생성합니다.


# ==================== 8. 작은 데이터로 코드 수식 테스트 ====================
def self_test():  # 전체 실험 전에 거리 계산과 투표를 빠르게 점검합니다.
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score  # 수식을 비교할 기준입니다.
    from sklearn.metrics import pairwise_distances  # 거리 계산을 비교할 기준입니다.
    rng = np.random.default_rng(123)  # 재현 가능한 예시 데이터를 만듭니다.
    train = rng.random((31, 24), dtype=np.float32)  # 24차원 학습 벡터 31개를 만듭니다.
    query = rng.random((11, 24), dtype=np.float32)  # 24차원 검증 벡터 11개를 만듭니다.
    y_train = rng.integers(0, 10, size=31)  # 학습 정답을 만듭니다.
    y_query = rng.integers(0, 10, size=11)  # 검증 정답을 만듭니다.
    device = "cuda" if torch.cuda.is_available() else "cpu"  # 작은 테스트에서는 CPU도 허용합니다.
    x = torch.tensor(train, device=device)  # 학습 벡터를 텐서로 바꿉니다.
    q = torch.tensor(query, device=device)  # 검증 벡터를 텐서로 바꿉니다.
    calculated = {"euclidean": euclidean_distance(q, x).cpu().numpy(),  # 직접 구현한 L2 결과입니다.
                  "manhattan": manhattan_distance(q, x).cpu().numpy(),  # 직접 구현한 L1 결과입니다.
                  "cosine": cosine_distance(q, F.normalize(x, dim=1)).cpu().numpy()}  # 직접 구현한 Cosine 결과입니다.
    for metric, dist in calculated.items():  # 세 거리 방식 모두 검사합니다.
        reference = pairwise_distances(query, train, metric=metric)  # 기준 라이브러리 결과를 구합니다.
        np.testing.assert_allclose(dist, reference, rtol=0.001, atol=0.001)  # 두 거리 행렬이 일치하는지 검사합니다.
        for k in [1, 3, 5, 7, 9]:  # 다섯 k 값에서 다수결을 검사합니다.
            indices = np.argsort(dist, axis=1)[:, :k]  # 가장 가까운 k개의 번호를 얻습니다.
            prediction = np.array([np.bincount(y_train[row], minlength=10).argmax() for row in indices])  # 다수결 예측을 구합니다.
            scores = evaluate(y_query, prediction)  # 직접 구현한 네 평가 지표를 계산합니다.
            expected = [accuracy_score(y_query, prediction),  # 기준 정확도를 계산합니다.
                        precision_score(y_query, prediction, labels=list(range(10)), average="macro", zero_division=0),  # 기준 정밀도를 계산합니다.
                        recall_score(y_query, prediction, labels=list(range(10)), average="macro", zero_division=0),  # 기준 재현율을 계산합니다.
                        f1_score(y_query, prediction, labels=list(range(10)), average="macro", zero_division=0)]  # 기준 F1을 계산합니다.
            np.testing.assert_allclose(scores, expected)  # 평가 수식의 결과가 일치하는지 검사합니다.
        print(f"검사 통과: {metric} 거리와 4개 평가 지표", flush=True)  # 검증 완료를 출력합니다.
    # GPU용 예측 함수 전체도 작은 픽셀 데이터로 점검합니다.
    global TRAIN_CHUNK, QUERY_BATCH  # 테스트에서는 묶음을 작게 줄여 전체 반복 경로를 점검합니다.
    old_chunk, old_batches = TRAIN_CHUNK, QUERY_BATCH.copy()  # 기존 전체 실험 설정을 저장합니다.
    TRAIN_CHUNK = 13  # 작은 테스트에서 학습 데이터를 세 묶음으로 나눕니다.
    QUERY_BATCH = {"euclidean": 4, "manhattan": 4, "cosine": 4}  # 검증 데이터도 여러 묶음으로 나눕니다.
    small_train = rng.integers(0, 256, size=(31, 24), dtype=np.uint8)  # 실제 이미지와 같은 uint8 픽셀 예시입니다.
    small_query = rng.integers(0, 256, size=(11, 24), dtype=np.uint8)  # 작은 검증 이미지 예시입니다.
    train_gpu = torch.as_tensor(small_train, device=device, dtype=torch.float32) / 255.0  # 테스트용 학습 텐서입니다.
    labels_gpu = torch.as_tensor(y_train, device=device, dtype=torch.long)  # 테스트용 정답 텐서입니다.
    for metric in DISTANCES:  # 세 거리 방식 각각 전체 예측 경로를 검사합니다.
        outputs = knn_predict(train_gpu, labels_gpu, small_query, metric, K_VALUES)  # 직접 구현한 KNN 예측을 구합니다.
        ref_distance = pairwise_distances(small_query / 255.0, small_train / 255.0, metric=metric)  # 기준 거리표입니다.
        for k in K_VALUES:  # 모든 k에서 예측을 비교합니다.
            ref_ids = np.argsort(ref_distance, axis=1)[:, :k]  # 기준 거리표에서 k개 이웃을 찾습니다.
            ref_prediction = np.array([np.bincount(y_train[row], minlength=10).argmax() for row in ref_ids])  # 기준 다수결입니다.
            np.testing.assert_array_equal(outputs[k], ref_prediction)  # 직접 구현한 예측과 기준을 비교합니다.
        print(f"KNN 이웃 선택·다수결 검사 통과: {metric}", flush=True)  # 전체 예측 절차 검증 결과입니다.
    TRAIN_CHUNK, QUERY_BATCH = old_chunk, old_batches  # 원래 전체 실험 설정으로 돌려놓습니다.
    print(f"작은 예제 검증 완료 (장치: {device}). 전체 CIFAR-10 GPU 실험은 실행하지 않았습니다.", flush=True)  # 테스트 범위를 안내합니다.


# ==================== 9. 전체 50,000장으로 5-fold 교차검증 ====================
def main():  # 전체 GPU 실험을 실행합니다.
    if not torch.cuda.is_available():  # NVIDIA CUDA를 사용할 수 있는지 확인합니다.
        raise RuntimeError("CUDA GPU가 감지되지 않았습니다. 'py knn_cifar10_gpu.py --gpu-check'와 'nvidia-smi'를 확인하세요. CPU로 자동 전환하지 않습니다.")  # CPU 실행을 방지합니다.
    if not DATA_PATH.exists():  # CIFAR-10 압축파일이 같은 폴더에 있는지 확인합니다.
        raise FileNotFoundError(f"{DATA_PATH} 파일을 코드와 같은 폴더에 넣어주세요.")  # 경로 오류를 알려줍니다.
    RESULT_DIR.mkdir(exist_ok=True)  # 결과를 저장할 폴더를 생성합니다.
    config_path = RESULT_DIR / "설정.json"  # 실험 설정 기록 파일입니다.
    current_config = {"folds": FOLD_COUNT, "seed": SEED, "k": K_VALUES, "distances": DISTANCES,  # Fold와 거리 설정을 기록합니다.
                      "image_shape": [32, 32, 3], "train_count": 50000, "test_count": 10000, "code_version": 1}  # 이미지 크기와 수를 기록합니다.
    if config_path.exists():  # 이전 결과가 있다면 설정을 확인합니다.
        if json.loads(config_path.read_text(encoding="utf-8")) != current_config:  # 설정이 다른 결과가 섞이지 않게 합니다.
            raise RuntimeError("기존 실험 설정과 다릅니다. results_full_gpu 폴더를 다른 이름으로 옮긴 후 실행하세요.")  # 재실행 방법을 알려줍니다.
    else:  # 실험을 처음 시작하는 경우입니다.
        config_path.write_text(json.dumps(current_config, ensure_ascii=False, indent=2), encoding="utf-8")  # 설정 파일을 작성합니다.

    print("GPU:", torch.cuda.get_device_name(0), flush=True)  # 사용하는 NVIDIA GPU 이름을 출력합니다.
    print("학습 50,000장 / 테스트 10,000장 / 원본 32×32 RGB 사용", flush=True)  # 사용 데이터 크기를 출력합니다.
    print("※ Euclidean·Cosine은 GPU 행렬곱, Manhattan은 GPU L1 연산입니다.", flush=True)  # 계산 방식을 안내합니다.
    print("※ 특히 Manhattan은 전체 50,000장 기준으로 GPU에서도 오래 걸릴 수 있습니다.", flush=True)  # 예상 부담을 알립니다.
    x_train, y_train, x_test, y_test = load_cifar10()  # CIFAR-10 배치 6개를 읽어 들입니다.
    cv_path = RESULT_DIR / "cv_fold별_결과.csv"  # Fold별 4개 지표를 저장할 파일입니다.
    all_rows = read_saved_rows(cv_path)  # 이미 계산한 결과가 있으면 가져옵니다.
    kfold = StratifiedKFold(n_splits=FOLD_COUNT, shuffle=True, random_state=SEED)  # 학습 50,000장을 균형 있게 5분할합니다.

    for fold, (train_idx, val_idx) in enumerate(kfold.split(x_train, y_train), start=1):  # 5개의 Train/Validation 분할을 순서대로 사용합니다.
        todo = [dist for dist in DISTANCES if not all(any(r[0] == fold and r[1] == dist and r[2] == k for r in all_rows) for k in K_VALUES)]  # 아직 안 끝난 거리 방식만 고릅니다.
        if not todo:  # 이 Fold가 이미 완료된 경우입니다.
            print(f"Fold {fold}/5: 저장된 결과가 있어 건너뜁니다.", flush=True)  # 중복 계산하지 않습니다.
            continue  # 다음 Fold로 넘어갑니다.
        print(f"\n========== Fold {fold}/5 : Train {len(train_idx):,}장 / Validation {len(val_idx):,}장 ==========", flush=True)  # 이번 검증의 규모입니다.
        train_gpu = torch.as_tensor(x_train[train_idx], device="cuda", dtype=torch.float32) / 255.0  # 학습 40,000장 전체를 GPU에 적재합니다.
        label_gpu = torch.as_tensor(y_train[train_idx], device="cuda", dtype=torch.long)  # 학습 정답도 GPU에 적재합니다.
        val_cpu = x_train[val_idx]  # 검증 10,000장은 작은 묶음으로 GPU에 전송합니다.
        val_labels = y_train[val_idx]  # 검증 정답을 가져옵니다.
        for dist in todo:  # 아직 결과가 없는 거리 방식만 계산합니다.
            print(f"  [{dist}] 시작", flush=True)  # 계산 중인 거리 방식입니다.
            predictions = knn_predict(train_gpu, label_gpu, val_cpu, dist, K_VALUES)  # 다섯 k의 예측을 한 번에 구합니다.
            all_rows = [r for r in all_rows if not (r[0] == fold and r[1] == dist)]  # 중복될 수 있는 기존 결과를 제거합니다.
            for k in K_VALUES:  # 다섯 k의 성능을 하나씩 출력합니다.
                scores = evaluate(val_labels, predictions[k])  # Accuracy, Precision, Recall, F1을 구합니다.
                all_rows.append([fold, dist, k] + scores)  # 현재 Fold 결과를 추가합니다.
                print(f"    Fold {fold} / {dist:9s} / k={k} : Acc={scores[0]*100:.2f}% P={scores[1]*100:.2f}% R={scores[2]*100:.2f}% F1={scores[3]*100:.2f}%", flush=True)  # 네 지표를 보여줍니다.
            all_rows.sort(key=lambda r: (r[0], DISTANCES.index(r[1]), r[2]))  # Fold, 거리, k 순으로 정렬합니다.
            save_csv(cv_path, CSV_HEADER, all_rows)  # 거리 계산 하나가 끝날 때마다 결과를 저장합니다.
            print(f"  저장 완료: {cv_path.name}", flush=True)  # 저장 사실을 출력합니다.
        del train_gpu, label_gpu, val_cpu  # 이번 Fold에서 사용한 GPU 텐서를 제거합니다.
        torch.cuda.empty_cache()  # 다음 Fold를 위한 캐시 메모리를 정리합니다.

    summary = []  # 15개 조합별 5-fold 평균과 표준편차를 담습니다.
    for dist in DISTANCES:  # 세 거리 측정 방식을 순서대로 확인합니다.
        for k in K_VALUES:  # 각 거리별 다섯 k를 확인합니다.
            values = np.array([r[3:] for r in all_rows if r[1] == dist and r[2] == k], dtype=np.float64)  # 해당 조합의 다섯 Fold 결과를 모읍니다.
            if len(values) != FOLD_COUNT:  # 다섯 결과가 모두 있는지 검사합니다.
                raise RuntimeError(f"누락된 결과가 있습니다: {dist}, k={k}, 완료 {len(values)}/5")  # 미완료를 알려줍니다.
            means = values.mean(axis=0)  # 다섯 Fold의 평균 점수를 구합니다.
            stds = values.std(axis=0)  # 다섯 Fold의 표준편차를 구합니다.
            summary.append([dist, k] + means.tolist() + stds.tolist())  # 조합별 평균과 표준편차를 저장합니다.
    save_csv(RESULT_DIR / "cv_평균_표준편차.csv", ["distance", "k", "accuracy_mean", "precision_mean", "recall_mean", "f1_mean",  # 평균 열 이름입니다.
                                                   "accuracy_std", "precision_std", "recall_std", "f1_std"], summary)  # 표준편차 열도 함께 저장합니다.
    print("\n========== 5-fold 평균 결과 ==========", flush=True)  # 다섯 검증의 평균을 출력합니다.
    for row in summary:  # 15개 조합을 한 줄씩 확인합니다.
        print(f"{row[0]:9s} k={row[1]} Acc={row[2]*100:.2f}% P={row[3]*100:.2f}% R={row[4]*100:.2f}% F1={row[5]*100:.2f}%", flush=True)  # 네 점수의 평균입니다.
    best = max(summary, key=lambda r: r[5])  # 평균 Macro F1이 가장 높은 조합을 선택합니다.
    print(f"\n최적 조합: {best[0]}, k={best[1]} / 5-fold 평균 F1={best[5]*100:.2f}%", flush=True)  # 최적 설정을 출력합니다.

    print("\n========== 전체 학습 50,000장 -> 별도 Test 10,000장 ==========", flush=True)  # 독립된 테스트 단계입니다.
    full_train_gpu = torch.as_tensor(x_train, device="cuda", dtype=torch.float32) / 255.0  # 학습 50,000장을 GPU에 모두 올립니다.
    full_label_gpu = torch.as_tensor(y_train, device="cuda", dtype=torch.long)  # 학습 정답도 GPU에 올립니다.
    pred = knn_predict(full_train_gpu, full_label_gpu, x_test, best[0], [best[1]])[best[1]]  # 최적 조합으로 테스트 이미지를 예측합니다.
    scores = evaluate(y_test, pred)  # 최종 네 평가 점수를 계산합니다.
    save_csv(RESULT_DIR / "test_예측상세.csv", ["sample_index", "true_label", "pred_label"],  # 혼동행렬을 GPU 없이 다시 그릴 수 있도록 예측을 보존합니다.
             [[i, int(y_test[i]), int(pred[i])] for i in range(len(y_test))])  # 10,000장 각각의 실제 정답과 예측을 저장합니다.
    save_confusion_matrix(y_test, pred)  # 테스트 이미지 10,000장으로 혼동행렬 2개를 생성합니다.
    save_csv(RESULT_DIR / "test_최종결과.csv", ["distance", "k", "accuracy", "precision", "recall", "f1"], [[best[0], best[1]] + scores])  # 최종 점수를 저장합니다.
    for name, score in zip(["Accuracy", "Precision", "Recall", "F1-score"], scores):  # 지표를 하나씩 확인합니다.
        print(f"  {name}: {score*100:.2f}%", flush=True)  # 최종 테스트 점수를 출력합니다.

    for dist in DISTANCES:  # 그래프에 거리 방식마다 선을 하나씩 그립니다.
        rows = [r for r in summary if r[0] == dist]  # 현재 거리의 k별 성능을 고릅니다.
        plt.plot(K_VALUES, [r[5] * 100 for r in rows], marker="o", label=dist)  # k와 평균 F1의 관계를 표시합니다.
    plt.xlabel("k (Number of neighbors)")  # 가로축 이름입니다.
    plt.ylabel("Macro F1 (%)")  # 세로축 이름입니다.
    plt.title("Full CIFAR-10 / 5-fold CV / GPU KNN")  # 그래프 제목입니다.
    plt.xticks(K_VALUES)  # 비교한 k 값을 눈금에 표시합니다.
    plt.legend()  # 세 거리 방식의 이름을 표시합니다.
    plt.grid(alpha=0.3)  # 비교하기 쉽게 격자를 그립니다.
    plt.tight_layout()  # 제목과 축이 잘리지 않게 배치합니다.
    plt.savefig(RESULT_DIR / "cv_F1_비교그래프.png", dpi=160)  # 그래프를 파일로 저장합니다.
    plt.close()  # 그래프를 닫습니다.
    print(f"\n실험 완료. 결과 저장 폴더: {RESULT_DIR}", flush=True)  # 저장 위치를 출력합니다.


if __name__ == "__main__":  # Python 파일을 직접 실행한 경우에만 아래를 수행합니다.
    parser = argparse.ArgumentParser(description="CIFAR-10 전체 데이터 GPU KNN")  # 실행 옵션을 설명합니다.
    parser.add_argument("--gpu-check", action="store_true", help="CUDA GPU 정보만 확인")  # GPU 점검 옵션입니다.
    parser.add_argument("--self-test", action="store_true", help="작은 예제로 거리·평가 수식만 테스트")  # 빠른 테스트 옵션입니다.
    parser.add_argument("--plot-only", action="store_true", help="저장된 테스트 예측 CSV로 혼동행렬만 다시 그리기")  # GPU 없이 그림만 다시 만듭니다.
    args = parser.parse_args()  # 입력한 실행 옵션을 읽습니다.
    if args.gpu_check:  # GPU 점검을 선택한 경우입니다.
        print("PyTorch 버전:", torch.__version__)  # 설치된 PyTorch 버전을 출력합니다.
        print("CUDA 빌드 버전:", torch.version.cuda)  # PyTorch에 포함된 CUDA 버전을 출력합니다.
        print("CUDA 사용 가능:", torch.cuda.is_available())  # NVIDIA GPU를 사용할 수 있는지 출력합니다.
        if torch.cuda.is_available():  # GPU가 정상 인식된 경우입니다.
            print("GPU 이름:", torch.cuda.get_device_name(0))  # GPU 이름을 출력합니다.
            free, total = torch.cuda.mem_get_info()  # 현재 GPU 메모리를 확인합니다.
            print(f"GPU 메모리: 남음 {free/1024**3:.2f}GB / 전체 {total/1024**3:.2f}GB")  # VRAM 용량을 출력합니다.
    elif args.self_test:  # 코드 수식을 점검하도록 요청한 경우입니다.
        self_test()  # 작은 데이터로 거리와 평가 지표를 검사합니다.
    elif args.plot_only:  # 이미 예측한 결과로 혼동행렬만 다시 만들고 싶은 경우입니다.
        plot_saved_predictions()  # GPU 없이 CSV에서 실제 정답과 예측을 읽습니다.
    else:  # 별도 옵션이 없으면 전체 실험을 실행합니다.
        main()  # CUDA에서 전체 5-fold + 테스트를 수행합니다.

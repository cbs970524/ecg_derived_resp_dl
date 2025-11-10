import os
from glob import glob

import numpy as np
import pandas as pd
from scipy import signal


def load_bidmc_data():
    # loading of bidmc
    # 💡 상수 사용으로 가독성 및 유지보수성 개선
    DATA_ROOT = "data/bidmc-ppg-and-respiration-dataset-1.0.0"
    FILE_PATTERN = "*Signals.csv"

    try:
        # **: 모든 하위 디렉토리를 재귀적으로 탐색 (이전 단계에서 수정한 부분)
        all_csv_files = glob(
            os.path.join(DATA_ROOT, "**", FILE_PATTERN), recursive=True
        )
        print(f"✅ 로드된 CSV 파일 수: {len(all_csv_files)}")
    except Exception as e:
        print(f"❌ 데이터 경로 탐색 오류 발생: {e}")
        all_csv_files = []

    patients = []
    data = {}
    no_errors = 0

    if not all_csv_files:
        print(
            f"❌⚠️ 오류: '{DATA_ROOT}' 경로에서 '{FILE_PATTERN}' 파일을 찾을 수 없습니다. 경로와 파일 구조를 확인하세요."
        )
        return data, patients

    for file in all_csv_files:
        try:
            df = pd.read_csv(file)
            X4 = df[" II"].values  # ECG (II Lead)
            Y = df[" RESP"].values  # Respiration

            # 🚨 수정된 부분: 환자 ID를 안전하게 추출하는 로직
            file_name_with_ext = os.path.basename(file)  # 예: 'bidmc_45_Signals.csv'
            file_name_no_ext = file_name_with_ext.split(".")[
                0
            ]  # 예: 'bidmc_45_Signals'

            # 'bidmc_45_Signals'를 '_'로 분리하고 두 번째 요소('45')를 가져옴
            patient_id_str = file_name_no_ext.split("_")[1]  # 예: '45'
            patient = int(patient_id_str)

            patients.append(patient)
            data[patient] = [X4, Y]

        except Exception as e:
            # ⚠️ 경고: 어떤 파일에서 어떤 오류가 났는지 명확하게 출력
            print(f"⚠️ 경고: 파일 처리 중 오류 발생 ({file}): {e}")
            no_errors += 1

    print(f"✅ 최종 로드된 환자 수: {len(patients)} (오류 건수: {no_errors})")
    return data, patients


def sliding_window(
    data,
    window_size,
    downsampled_window_size,
    overlap,
    train_patients,
    validation_patients,
    test_patients,
):
    windows_ecg_train = []
    windows_resp_train = []

    for train_patient in train_patients:

        N = len(data[train_patient][0])
        max_step = int(N // (window_size * overlap))
        for step in range(max_step):
            ecg = data[train_patient][0][
                step * int(window_size * overlap) : step * int(window_size * overlap)
                + window_size
            ]
            resp = data[train_patient][1][
                step * int(window_size * overlap) : step * int(window_size * overlap)
                + window_size
            ]

            if ecg.min() < ecg.max():
                normalized_ecg = (ecg - ecg.min()) / (ecg.max() - ecg.min()) - 0.5
                # zero_centered_ecg = ecg - np.mean(ecg)
                # normalized_ecg = zero_centered_ecg / np.std(zero_centered_ecg)
                resampled_ecg = signal.resample(normalized_ecg, downsampled_window_size)
                if resp.min() < resp.max():
                    normalized_resp = (resp - resp.min()) / (resp.max() - resp.min())
                    # zero_centered_resp = resp - np.mean(resp)
                    # normalized_resp = zero_centered_resp / np.std(zero_centered_resp)
                    resampled_resp = signal.resample(
                        normalized_resp, downsampled_window_size
                    )
                    windows_ecg_train.append(np.float32(resampled_ecg))
                    windows_resp_train.append(np.float32(resampled_resp))

    windows_ecg_validation = []
    windows_resp_validation = []

    for validation_patient in validation_patients:
        N = len(data[validation_patient][0])
        max_step = int(N // (window_size * overlap))
        for step in range(max_step):
            ecg = data[validation_patient][0][
                step * int(window_size * overlap) : step * int(window_size * overlap)
                + window_size
            ]
            resp = data[validation_patient][1][
                step * int(window_size * overlap) : step * int(window_size * overlap)
                + window_size
            ]

            if ecg.min() < ecg.max():
                normalized_ecg = (ecg - ecg.min()) / (ecg.max() - ecg.min()) - 0.5
                # zero_centered_ecg = ecg - np.mean(ecg)
                # normalized_ecg = zero_centered_ecg / np.std(zero_centered_ecg)
                resampled_ecg = signal.resample(normalized_ecg, downsampled_window_size)
                if resp.min() < resp.max():
                    normalized_resp = (resp - resp.min()) / (resp.max() - resp.min())
                    # zero_centered_resp = resp - np.mean(resp)
                    # normalized_resp = zero_centered_resp / np.std(zero_centered_resp)
                    resampled_resp = signal.resample(
                        normalized_resp, downsampled_window_size
                    )
                    windows_ecg_validation.append(np.float32(resampled_ecg))
                    windows_resp_validation.append(np.float32(resampled_resp))

    windows_ecg_test = []
    windows_resp_test = []

    for test_patient in test_patients:
        N = len(data[test_patient][0])
        max_step = int(N // (window_size * overlap))
        for step in range(max_step):
            ecg = data[test_patient][0][
                step * int(window_size * overlap) : step * int(window_size * overlap)
                + window_size
            ]
            resp = data[test_patient][1][
                step * int(window_size * overlap) : step * int(window_size * overlap)
                + window_size
            ]

            if ecg.min() < ecg.max():
                normalized_ecg = (ecg - ecg.min()) / (ecg.max() - ecg.min()) - 0.5
                # zero_centered_ecg = ecg - np.mean(ecg)
                # normalized_ecg = zero_centered_ecg / np.std(zero_centered_ecg)
                resampled_ecg = signal.resample(normalized_ecg, downsampled_window_size)
                if resp.min() < resp.max():
                    normalized_resp = (resp - resp.min()) / (resp.max() - resp.min())
                    # zero_centered_resp = resp - np.mean(resp)
                    # normalized_resp = zero_centered_resp / np.std(zero_centered_resp)
                    resampled_resp = signal.resample(
                        normalized_resp, downsampled_window_size
                    )
                    windows_ecg_test.append(np.float32(resampled_ecg))
                    windows_resp_test.append(np.float32(resampled_resp))

    windows_ecg_train = np.stack(windows_ecg_train, axis=0)
    windows_resp_train = np.stack(windows_resp_train, axis=0)
    windows_ecg_validation = np.stack(windows_ecg_validation, axis=0)
    windows_resp_validation = np.stack(windows_resp_validation, axis=0)
    windows_ecg_test = np.stack(windows_ecg_test, axis=0)
    windows_resp_test = np.stack(windows_resp_test, axis=0)

    windows_ecg_train = windows_ecg_train[:, :, np.newaxis]
    windows_resp_train = windows_resp_train[:, :, np.newaxis]
    windows_ecg_validation = windows_ecg_validation[:, :, np.newaxis]
    windows_resp_validation = windows_resp_validation[:, :, np.newaxis]
    windows_ecg_test = windows_ecg_test[:, :, np.newaxis]
    windows_resp_test = windows_resp_test[:, :, np.newaxis]

    return (
        windows_ecg_train,
        windows_resp_train,
        windows_ecg_validation,
        windows_resp_validation,
        windows_ecg_test,
        windows_resp_test,
    )

# 모델 학습 파라미터

이 문서는 `AI_Learn` 저장소에서 YOLO 기반 헬멧/사람/조끼 PPE 객체 탐지 모델을 학습할 때 사용한 주요 파라미터를 정리한 문서입니다.

기준 파일은 `notebooks/train_hiyoung_ppe_colab.ipynb`입니다. 해당 노트북의 본 학습 셀에 남아 있는 실행 명령을 기준으로 가장 최근 학습 파라미터를 정리했습니다.

## 학습 목적

- 작업 목적: 헬멧, 사람, 안전조끼를 탐지하는 PPE 객체 탐지 모델 학습
- 사용 모델 계열: Ultralytics YOLO
- 최종 사용 목적: 외부 프로젝트의 AI Worker에서 `.pt` 가중치 파일을 로드하여 실시간 안전 모니터링에 사용

## 클래스 구성

학습 데이터셋의 최종 클래스 순서는 아래 순서를 기준으로 사용했습니다.

```yaml
0: helmet
1: person
2: vest
```

원본 SH17 데이터셋에서는 필요한 클래스만 추려서 아래처럼 다시 매핑했습니다.

```text
Person source id 0       -> person -> 1
Helmet source id 10      -> helmet -> 0
Safety-vest source id 16 -> vest   -> 2
```

중요: 원본 데이터셋의 클래스 id를 그대로 사용하지 않고, 최종 YOLO 라벨 기준에 맞춰 `helmet=0`, `person=1`, `vest=2`로 재매핑했습니다.

## 데이터셋 YAML

Colab 노트북에서는 `/content/AI_Learn/datasets/hiyoung_ppe_content.yaml` 파일을 생성해서 사용했습니다.

```yaml
path: /content/AI_Learn/datasets/hiyoung_ppe

train: images/train
val: images/val

names:
  0: helmet
  1: person
  2: vest
```

## 최근 본 학습 파라미터

가장 최근 본 학습은 `yolo26n.pt`를 사용했습니다.

| 항목 | 값 | 설명 |
|---|---:|---|
| Base model | `/content/AI_Learn/yolo26n.pt` | Google Drive에서 복사한 26n 모델 가중치 |
| 원본 모델 위치 | `/content/drive/MyDrive/AI_Learn/yolo26n.pt` | Colab 실행 전 Drive에 둔 모델 파일 |
| Dataset YAML | `datasets/hiyoung_ppe_content.yaml` | Colab 런타임 내부 기준 데이터셋 설정 파일 |
| Epochs | `60` | 학습 epoch 수 |
| Image size | `640` | 입력 이미지 크기 |
| Project | `/content/AI_Learn/runs` | 학습 결과 저장 위치 |
| Run name | `helmet_person_vest_yolo26n_640_e60` | 26n 학습 실행 이름 |

노트북 학습 셀에는 `batch`, `device`, `patience` 옵션을 명시하지 않았습니다. 따라서 해당 값들은 Ultralytics YOLO의 기본 동작을 따릅니다.

## 실행 명령

`notebooks/train_hiyoung_ppe_colab.ipynb`의 본 학습 셀 기준 실행 명령은 아래와 같습니다.

```bash
cd /content/AI_Learn

cp /content/drive/MyDrive/AI_Learn/yolo26n.pt /content/AI_Learn/yolo26n.pt

python scripts/train_yolo.py \
  --model /content/AI_Learn/yolo26n.pt \
  --epochs 60 \
  --imgsz 640 \
  --data datasets/hiyoung_ppe_content.yaml \
  --project /content/AI_Learn/runs \
  --name helmet_person_vest_yolo26n_640_e60
```

## 검증 명령

학습 후 아래 명령으로 최종 `best.pt`를 검증했습니다.

```bash
python scripts/validate_model.py \
  --weights /content/AI_Learn/runs/helmet_person_vest_yolo26n_640_e60/weights/best.pt
```

노트북에서는 이어서 아래 결과 이미지를 확인했습니다.

```text
/content/AI_Learn/runs/helmet_person_vest_yolo26n_640_e60/val_batch0_labels.jpg
/content/AI_Learn/runs/helmet_person_vest_yolo26n_640_e60/val_batch0_pred.jpg
/content/AI_Learn/runs/helmet_person_vest_yolo26n_640_e60/confusion_matrix.png
/content/AI_Learn/runs/helmet_person_vest_yolo26n_640_e60/results.png
```

## 학습 결과물

학습 완료 후 생성되는 기본 가중치 파일은 아래 위치에 저장됩니다.

```text
/content/AI_Learn/runs/helmet_person_vest_yolo26n_640_e60/weights/best.pt
```

최종 검증이 끝난 가중치는 Google Drive의 `weights/` 폴더로 복사했습니다.

```bash
mkdir -p /content/drive/MyDrive/AI_Learn/weights

cp /content/AI_Learn/runs/helmet_person_vest_yolo26n_640_e60/weights/best.pt \
  /content/drive/MyDrive/AI_Learn/weights/hiyoung_helmet_person_vest_yolo26n_640_e60.pt
```

최종 파일명은 아래와 같습니다.

```text
/content/drive/MyDrive/AI_Learn/weights/hiyoung_helmet_person_vest_yolo26n_640_e60.pt
```

## 이전 테스트 학습 참고

본 학습 전에 데이터셋 구조, 라벨 매핑, 클래스 이름이 정상적으로 동작하는지 확인하기 위해 테스트 학습을 먼저 진행했습니다.

| 항목 | 값 |
|---|---|
| 테스트 모델 | `yolo11s.pt` |
| 테스트 Epoch | `10` |
| 테스트 Image size | `640` |
| 테스트 Dataset YAML | `datasets/hiyoung_ppe_content.yaml` |
| 테스트 Run name | `test_fixed_helmet_person_vest_yolo11s` |

테스트 학습 명령은 아래와 같습니다.

```bash
python scripts/train_yolo.py \
  --model yolo11s.pt \
  --epochs 10 \
  --imgsz 640 \
  --data datasets/hiyoung_ppe_content.yaml \
  --name test_fixed_helmet_person_vest_yolo11s
```

## Git 관리 비고

- `runs/` 폴더의 중간 학습 산출물은 Git에 올리지 않습니다.
- `raw_datasets/`, `datasets/hiyoung_ppe/`, `downloads/`, `runs/`, `logs/`는 Git 제외 대상입니다.
- 최종 공유 또는 배포용 `.pt` 파일만 `weights/` 폴더에 복사해서 관리합니다.
- 파일 크기가 큰 `.pt` 가중치는 Git LFS 사용을 권장합니다.

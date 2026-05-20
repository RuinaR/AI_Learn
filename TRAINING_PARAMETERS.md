# 모델 학습 파라미터

이 문서는 `AI_Learn` 저장소에서 YOLO 기반 헬멧/사람/조끼 PPE 객체 탐지 모델을 학습할 때 사용한 주요 파라미터를 정리한 문서입니다.

중요: `scripts/train_yolo.py`에는 이전 기준의 기본값으로 `yolo11m.pt`가 남아 있을 수 있습니다. 가장 최근에 실제 학습하고 Git에 올린 모델은 26n 모델 기준으로 정리합니다.

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

Colab 및 Google Drive 기준 학습에서는 아래 YAML 경로를 사용했습니다.

```yaml
path: /content/drive/MyDrive/AI_Learn/datasets/hiyoung_ppe
train: images/train
val: images/val
test: images/test

names:
  0: helmet
  1: person
  2: vest
```

## 최근 본 학습 파라미터

가장 최근에 실제 학습한 모델은 26n 모델입니다.

| 항목 | 값 | 설명 |
|---|---:|---|
| Base model | `yolo26n.pt` | 최근 학습에 사용한 26n 계열 모델 |
| Dataset YAML | `/content/drive/MyDrive/AI_Learn/datasets/hiyoung_ppe.yaml` | Colab + Google Drive 기준 데이터셋 설정 파일 |
| Epochs | `120` | 학습 epoch 수 |
| Image size | `960` | 입력 이미지 크기 |
| Batch size | `-1` | Ultralytics YOLO의 자동 batch 설정 사용 |
| Device | `0` | 첫 번째 GPU 사용 |
| Patience | `30` | Early stopping 기준 epoch 수 |
| Project | `/content/drive/MyDrive/AI_Learn/runs` | 학습 결과 저장 위치 |
| Run name | `helmet_person_vest_yolo26n_960` | 26n 학습 실행 이름 |

## 실행 명령 예시

```bash
python scripts/train_yolo.py \
  --model yolo26n.pt \
  --data /content/drive/MyDrive/AI_Learn/datasets/hiyoung_ppe.yaml \
  --epochs 120 \
  --imgsz 960 \
  --batch -1 \
  --device 0 \
  --patience 30 \
  --project /content/drive/MyDrive/AI_Learn/runs \
  --name helmet_person_vest_yolo26n_960
```

## 이전 테스트/기본값 참고

본 학습 전에 데이터셋 구조, 라벨 매핑, 클래스 이름이 정상적으로 동작하는지 확인하기 위해 테스트 학습을 먼저 진행하는 흐름을 사용했습니다.

| 항목 | 값 |
|---|---|
| 테스트 모델 | `yolo11s` |
| 테스트 Epoch | `10` |
| 목적 | 데이터셋 구조, 클래스 순서, 라벨 매핑 검증 |
| 검증 기준 | `best.pt`의 클래스 이름이 `helmet`, `person`, `vest` 순서인지 확인 |

현재 `scripts/train_yolo.py`의 기본값은 이전 학습 기준인 `yolo11m.pt`, `helmet_person_vest_yolo11m_960`으로 남아 있을 수 있습니다. 최신 실험 내용을 재현하려면 위의 26n 실행 명령처럼 `--model`, `--name`을 명시해서 실행합니다.

## 학습 전 확인 절차

학습 전에 아래 순서로 데이터셋과 라벨 상태를 확인했습니다.

```bash
python scripts/convert_sh17_to_hiyoung_ppe.py --overwrite
python scripts/check_yolo_dataset.py
python scripts/visualize_yolo_labels.py --split val --count 12
```

라벨 시각화 결과에서 헬멧, 사람, 안전조끼가 각각 `helmet`, `person`, `vest`로 올바르게 표시되는지 확인한 뒤 학습을 진행했습니다.

## 학습 결과물

학습 완료 후 생성되는 기본 가중치 파일은 아래 위치에 저장됩니다.

```text
/content/drive/MyDrive/AI_Learn/runs/helmet_person_vest_yolo26n_960/weights/best.pt
```

최종 검증이 끝난 가중치는 외부 프로젝트에서 사용하기 쉽도록 아래와 같은 이름으로 복사해서 관리합니다.

```text
weights/hiyoung_helmet_person_vest_yolo26n.pt
```

## Git 관리 비고

- `runs/` 폴더의 중간 학습 산출물은 Git에 올리지 않습니다.
- `raw_datasets/`, `datasets/hiyoung_ppe/`, `downloads/`, `runs/`, `logs/`는 Git 제외 대상입니다.
- 최종 공유 또는 배포용 `.pt` 파일만 `weights/` 폴더에 복사해서 관리합니다.
- 파일 크기가 큰 `.pt` 가중치는 Git LFS 사용을 권장합니다.

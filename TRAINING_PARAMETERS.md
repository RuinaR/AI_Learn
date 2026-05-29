# 모델 학습 파라미터

이 문서는 현재 저장소 기준의 권장 학습 설정을 정리합니다. 목표는 로컬 GPU에서 `yolo26x`로 `helmet`, `person` 2클래스만 최대한 정확하게 학습하는 것입니다.

## 클래스 구성

```yaml
0: helmet
1: person
```

원본 SH17 매핑은 아래를 사용합니다.

```text
Helmet source id 10 -> helmet -> 0
Person source id 0  -> person -> 1
```

`Safety-vest`와 기타 클래스는 모두 제거합니다.

## 데이터셋 YAML

로컬 기본 YAML:

```yaml
path: C:/Users/AISW_203_114/Desktop/AI_Learn/datasets/hiyoung_ppe
train: images/train
val: images/val
test: images/test

names:
  0: helmet
  1: person
```

## 권장 본 학습 설정

| 항목 | 값 |
|---|---:|
| Base model | `yolo26x.pt` |
| Dataset YAML | `datasets/hiyoung_ppe_local.yaml` |
| Epochs | `120` |
| Image size | `960` |
| Batch | `0.70` |
| Device | `0` |
| Patience | `25` |
| Cache | `ram` |
| Optimizer | `auto` |
| Close mosaic | `10` |
| Mosaic | `0.9` |
| MixUp | `0.2` |
| Copy-paste | `0.2` |
| Degrees | `0.0` |
| Translate | `0.2` |
| Scale | `0.85` |
| Horizontal flip | `0.3` |
| HSV-H | `0.013` |
| HSV-S | `0.35` |
| HSV-V | `0.2` |
| Weight decay | `0.00027` |
| Warmup epochs | `3.0` |
| Box loss | `9.83` |
| Cls loss | `0.65` |
| Cls weight power | `0.25` |
| DFL | `0.96` |
| Erasing | `0.1` |
| Project | `runs` |
| Run name | `helmet_person_yolo26x_960_rtx3080` |
| Engine export | `enabled` |
| Engine batch | `1` |
| Engine workspace | `4.0 GiB` |

## 권장 실행 명령

```bash
py -3.12 scripts/convert_sh17_to_hiyoung_ppe.py --overwrite
py -3.12 scripts/check_yolo_dataset.py
py -3.12 scripts/visualize_yolo_labels.py --split val --count 12

py -3.12 scripts/train_yolo.py \
  --model yolo26x.pt \
  --data datasets/hiyoung_ppe_local.yaml \
  --device 0 \
  --epochs 120 \
  --imgsz 960 \
  --batch 0.70 \
  --name helmet_person_yolo26x_960_rtx3080 \
  --cos-lr
```

이 명령은 학습 후 `best.pt` 기준으로 TensorRT `.engine` export까지 같이 수행합니다.

한 번에 실행하려면 루트의 `train_engine_rtx3080.cmd`를 사용해도 됩니다.

중간 종료 후 재개하려면 루트의 `resume_train_engine_rtx3080.cmd`를 사용하면 됩니다.

## 반영 근거

- YOLO26 X pretraining recipe는 `MuSGD`, 낮은 `lr0`, 높은 `mosaic/scale`, `close_mosaic=10`, `degrees=0` 성향을 사용합니다.
- YOLO fine-tuning 가이드는 커스텀 데이터에선 기본값부터 시작하고, 작은 물체면 `imgsz=1280`, 클래스 불균형이면 `cls_pw`를 조정하라고 권장합니다.
- 현재 데이터는 자연 이미지 기반이라 COCO와 도메인이 크게 다르지 않지만, `helmet`이 소수 클래스이고 작은 박스로 등장할 가능성이 커서 `imgsz`, `cls_pw`, `mosaic/scale` 쪽을 보강했습니다.
- 로컬 머신은 `NVIDIA GeForce RTX 3080 10GB`이고 현재 데스크톱 프로세스가 약 `1.47GB` VRAM을 사용 중이라, 기본 프리셋은 `960 / batch 0.70`으로 맞췄습니다.

## 자동 튜닝 명령

```bash
py -3.12 scripts/tune_yolo.py \
  --model yolo26x.pt \
  --data datasets/hiyoung_ppe_local.yaml \
  --device 0 \
  --iterations 30
```

## RTX 3080 10GB 권장 단계

1. 1차 안정 프리셋: `--imgsz 960 --batch 0.70`
2. 2차 정확도 프리셋: 다른 GPU 점유 앱을 줄인 뒤 `--imgsz 1280 --batch 0.55`
3. OOM 시: `--batch 0.60` 또는 `--batch 0.50`

## GPU 메모리 부족 시 대안

메모리가 부족하면 아래 순서로 하나씩 낮추는 것을 권장합니다.

1. `--imgsz 960`
2. `--batch 4`
3. `--batch 2`
4. 필요하면 `--model yolo26l.pt`

정확도 우선이라면 가능하면 `yolo26x`를 유지하고, 먼저 `imgsz`와 `batch`를 조정하는 편이 좋습니다.

## 검증 명령

```bash
py -3.12 scripts/validate_model.py \
  --weights runs/helmet_person_yolo26x_960_rtx3080/weights/best.pt
```

정상 상태에서는 클래스명이 `helmet`, `person`만 확인되어야 합니다.

## TensorRT 엔진 관련 옵션

기본값:

- `--engine-batch 1`
- `--engine-workspace 4.0`
- FP16 엔진 export 사용

예시:

```bash
python scripts/train_yolo.py \
  --device 0 \
  --imgsz 960 \
  --batch 0.70 \
  --engine-batch 1 \
  --engine-workspace 4
```

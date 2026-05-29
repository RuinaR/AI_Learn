# AI_Learn

이 저장소는 로컬 또는 Colab에서 YOLO 기반 PPE 탐지 모델을 학습하기 위한 독립 작업공간입니다. 이번 기준은 `helmet`, `person` 2클래스만 사용하며, 로컬 GPU에서 `yolo26x`를 최대한 정확도 위주로 학습하는 흐름에 맞춰 정리되어 있습니다.

## 현재 클래스 규칙

외부 프로젝트 연동을 고려해 클래스 순서는 아래처럼 고정합니다.

```yaml
0: helmet
1: person
```

중요: `person`과 `helmet`만 학습하더라도 class id 순서는 `helmet=0`, `person=1`을 유지합니다.

## SH17 원본 매핑

원본 SH17 데이터셋에서는 아래 2개 클래스만 사용합니다.

```text
Helmet source id 10 -> helmet -> 0
Person source id 0  -> person -> 1
```

`Safety-vest`를 포함한 나머지 클래스는 모두 제외합니다.

## 주요 파일

- [scripts/convert_sh17_to_hiyoung_ppe.py](/C:/Users/AISW_203_114/Desktop/AI_Learn/scripts/convert_sh17_to_hiyoung_ppe.py): SH17를 2클래스 YOLO 데이터셋으로 변환
- [scripts/check_yolo_dataset.py](/C:/Users/AISW_203_114/Desktop/AI_Learn/scripts/check_yolo_dataset.py): 데이터셋 구조와 라벨 검사
- [scripts/visualize_yolo_labels.py](/C:/Users/AISW_203_114/Desktop/AI_Learn/scripts/visualize_yolo_labels.py): 샘플 라벨 시각화
- [scripts/train_yolo.py](/C:/Users/AISW_203_114/Desktop/AI_Learn/scripts/train_yolo.py): 로컬 GPU/Colab 학습
- [scripts/validate_model.py](/C:/Users/AISW_203_114/Desktop/AI_Learn/scripts/validate_model.py): 학습된 가중치 클래스명 검증
- [scripts/export_weight_to_project.py](/C:/Users/AISW_203_114/Desktop/AI_Learn/scripts/export_weight_to_project.py): 외부 프로젝트로 `.pt` 복사

## 데이터셋 YAML

- `datasets/hiyoung_ppe_local.yaml`: 로컬 Windows 기준
- `datasets/hiyoung_ppe.yaml`: Colab/Drive 기준
- `datasets/hiyoung_ppe_colab.yaml`: Colab에서 별도 지정할 때 사용

모든 YAML은 현재 `helmet`, `person` 2클래스만 선언합니다.

## 권장 실행 순서

1. SH17 원본에서 2클래스 데이터셋 재생성

```bash
py -3.12 scripts/convert_sh17_to_hiyoung_ppe.py --overwrite
```

2. 데이터셋 검사

```bash
py -3.12 scripts/check_yolo_dataset.py
```

3. 라벨 시각화

```bash
py -3.12 scripts/visualize_yolo_labels.py --split val --count 12
```

4. 로컬 GPU로 `yolo26x` 학습

```bash
py -3.12 scripts/train_yolo.py ^
  --model yolo26x.pt ^
  --data datasets/hiyoung_ppe_local.yaml ^
  --device 0 ^
  --epochs 120 ^
  --imgsz 960 ^
  --batch 0.70 ^
  --name helmet_person_yolo26x_960_rtx3080 ^
  --cos-lr
```

기본 동작으로 학습 완료 후 `runs/.../weights/best.pt`에서 TensorRT `.engine`도 함께 export합니다.

기본값 자체가 정확도 우선 설정입니다. 공식 YOLO26 문서의 X 모델 recipe와 fine-tuning 가이드를 참고해, 현재 저장소에는 아래 방향으로 반영했습니다.

- 모델 기본값: `yolo26x.pt`
- 이미지 크기 기본값: `960`
- epoch 기본값: `120`
- `patience=25`
- `close_mosaic=10`
- `mosaic=0.9`
- `mixup=0.2`
- `copy_paste=0.2`
- `scale=0.85`
- `translate=0.2`
- `degrees=0.0`
- `fliplr=0.3`
- `hsv_h=0.013`, `hsv_s=0.35`, `hsv_v=0.2`
- `box=9.83`, `cls=0.65`, `dfl=0.96`
- `cls_pw=0.25`
- `weight_decay=0.00027`
- `cache=ram`
- `amp=True`

이 값들은 COCO용 YOLO26x 원본 recipe보다 약간 완만합니다. 이유는 현재 데이터셋이 약 `7.6k`장 규모라서, pretraining 수준의 매우 강한 설정을 그대로 복제하기보다 fine-tuning 안정성과 일반화 성능을 조금 더 우선했기 때문입니다.

현재 로컬 GPU 기준 권장 프리셋은 `RTX 3080 10GB`에 맞춘 값입니다.

- 기본 학습 프리셋: `imgsz=960`, `batch=0.70`
- 이유: Windows 데스크톱 프로세스가 이미 약 `1.5GB` VRAM을 사용 중이라 `yolo26x + 1280` 기본 진입은 OOM 위험이 큽니다.
- 정확도 우선 2차 시도: 다른 앱을 줄인 뒤 `--imgsz 1280 --batch 0.55`

GPU 메모리가 부족하면 `--imgsz 960` 또는 `--batch 4`처럼 낮춰서 시작하면 됩니다.

`batch`는 `-1` 외에도 `0.70` 같은 GPU 메모리 활용 비율을 받을 수 있게 바꿔 두었습니다.

현재 로컬에서 학습과 `.engine` export 의존성이 깔린 인터프리터는 `Python 3.12`입니다. 기본 `python`은 3.14일 수 있으니, 로컬에선 `py -3.12 ...` 형태로 실행하는 것을 권장합니다.

바로 실행용 배치 파일:

```bash
train_engine_rtx3080.cmd
```

중간 종료 후 재개용 배치 파일:

```bash
resume_train_engine_rtx3080.cmd
```

사전 환경 점검만 하고 싶으면:

```bash
py -3.12 scripts/check_train_export_env.py
```

## 추가 튜닝

더 밀어붙이고 싶으면 [scripts/tune_yolo.py](/C:/Users/AISW_203_114/Desktop/AI_Learn/scripts/tune_yolo.py:1) 로 좁은 탐색 범위의 자동 튜닝을 돌릴 수 있습니다.

```bash
py -3.12 scripts/tune_yolo.py ^
  --model yolo26x.pt ^
  --data datasets/hiyoung_ppe_local.yaml ^
  --device 0 ^
  --iterations 30
```

## 학습 후 검증

```bash
py -3.12 scripts/validate_model.py ^
  --weights runs/helmet_person_yolo26x_960_rtx3080/weights/best.pt
```

정상이라면 모델 클래스는 `helmet`, `person`만 보여야 합니다.

생성되는 주요 산출물:

- `runs/helmet_person_yolo26x_960_rtx3080/weights/best.pt`
- `runs/helmet_person_yolo26x_960_rtx3080/weights/best.engine` 또는 export 결과 경로

## 외부 프로젝트로 복사

```bash
py -3.12 scripts/export_weight_to_project.py ^
  --source runs/helmet_person_yolo26x_960_rtx3080/weights/best.pt ^
  --target-dir C:\path\to\external\project\weights
```

기본 출력 파일명은 `hiyoung_helmet_person_yolo26x.pt`입니다.

TensorRT export를 끄고 싶으면 아래 옵션을 추가하면 됩니다.

```bash
--skip-engine-export
```

엔진 export가 실패하면 보통 TensorRT 설치 또는 Ultralytics export 의존성 문제입니다. 이 경우 `.pt` 학습 결과는 남고, `.engine` 단계에서 오류가 발생합니다.

## 참고

- 기존 `datasets/hiyoung_ppe/` 안에 3클래스 라벨이 남아 있으면 그대로 학습하면 안 됩니다.
- 반드시 `convert_sh17_to_hiyoung_ppe.py --overwrite`로 다시 생성한 뒤 검사하고 학습하세요.
- `runs/`와 데이터셋 폴더는 Git 제외 대상입니다. 최종 배포용 `.pt`만 `weights/`에 따로 보관하는 방식을 권장합니다.

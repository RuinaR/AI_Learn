# AI_Learn

`AI_Learn` 폴더는 실제 서비스 레포가 아닌, YOLO 기반 헬멧/사람 객체 탐지 모델을 학습하기 위한 독립 작업공간입니다. 이곳에서 데이터셋 점검, 학습, 검증, 가중치 정리를 수행한 뒤, 완성된 `.pt` 파일만 다른 프로젝트에 복사해서 사용합니다.

외부 프로젝트 `hiyoung_team_github/safety_monitor_workspace`는 Python 기반 AI Worker에서 YOLO `.pt` 모델을 읽어 사용한다고 가정합니다. 이 작업공간에서는 외부 프로젝트 파일을 읽거나 수정하지 않고, 학습에 필요한 파일만 현재 폴더 안에서 관리합니다.

## 중요한 클래스 이름 규칙

외부 프로젝트의 안전모 미착용 판단 로직은 클래스 이름 문자열에 의존하므로, 학습 데이터셋과 최종 모델의 클래스 이름은 반드시 아래 순서를 유지해야 합니다.

```yaml
0: helmet
1: person
```

클래스 이름을 바꾸거나 순서를 바꾸면 외부 프로젝트에서 오동작할 수 있습니다.

## 권장 폴더 구조

```text
AI_Learn/
├─ README.md
├─ .gitignore
├─ datasets/
│  ├─ hiyoung_ppe.yaml
│  └─ hiyoung_ppe/
│     ├─ images/
│     │  ├─ train/
│     │  ├─ val/
│     │  └─ test/
│     └─ labels/
│        ├─ train/
│        ├─ val/
│        └─ test/
├─ scripts/
├─ notebooks/
├─ runs/
└─ weights/
```

YOLO 라벨 파일은 이미지와 같은 파일명으로 `.txt` 형식을 사용해야 하며, 각 줄은 아래 형식을 따라야 합니다.

```text
class_id x_center y_center width height
```

모든 bbox 값은 정규화된 값이라서 `0~1` 범위여야 합니다.

## Colab + Google Drive 기준 사용 순서

1. 이 폴더를 Google Drive의 `MyDrive/AI_Learn` 위치에 맞춰 두고 데이터셋을 `datasets/hiyoung_ppe/` 아래에 배치합니다.
2. `datasets/hiyoung_ppe.yaml`의 클래스 이름이 `helmet`, `person` 순서인지 다시 확인합니다.
3. 필요하면 Colab 또는 로컬에서 `scripts/check_yolo_dataset.py`로 데이터셋 구조와 라벨 형식을 검사합니다.
4. Colab에서 `scripts/train_yolo.py` 또는 `notebooks/train_hiyoung_ppe_colab.ipynb`를 사용해 학습합니다.
5. 학습 완료 후 생성된 `best.pt`를 `scripts/validate_model.py`로 불러와 클래스 이름과 예측 결과를 점검합니다.
6. 최종 확인이 끝난 가중치를 외부 프로젝트의 `safety_ai_monitor/models/weights/` 폴더로 복사합니다.
7. 외부 프로젝트에서는 `config.py`의 `MODEL_PATH`를 복사한 `.pt` 파일명으로 맞춰 사용하면 됩니다.

## 외부 프로젝트 반영 방법

학습이 끝나면 보통 `runs/.../weights/best.pt`가 생성됩니다. 이 파일을 외부 프로젝트의 아래 폴더로 복사합니다.

```text
safety_ai_monitor/models/weights/
```

복사 후 외부 프로젝트의 `config.py`에서 `MODEL_PATH`를 새 파일명으로 지정하면 됩니다. 이 작업공간에는 외부 프로젝트 파일이 없으므로, 실제 반영은 해당 프로젝트에서 별도로 수행해야 합니다.

## Git 관리 주의사항

다음 파일과 결과물은 Git에 올리지 않는 것을 권장합니다.

- 원본 데이터셋 이미지와 라벨
- `runs/` 아래 학습 결과물
- `.pt`, `.onnx` 같은 모델 가중치 파일

이 저장소의 `.gitignore`에는 위 항목이 반영되어 있습니다.


# AI_Learn

`AI_Learn` 폴더는 실제 서비스 레포가 아닌, YOLO 기반 헬멧/사람/조끼 PPE 객체 탐지 모델을 학습하기 위한 독립 작업공간입니다. 이곳에서 데이터셋 점검, 학습, 검증, 가중치 정리를 수행한 뒤, 완성된 `.pt` 파일만 다른 프로젝트에 복사해서 사용합니다. 학습 완료 후 최종 배포용 가중치는 `AI_Learn/weights/` 폴더에 따로 보관합니다.

외부 프로젝트 `hiyoung_team_github/safety_monitor_workspace`는 Python 기반 AI Worker에서 YOLO `.pt` 모델을 읽어 사용한다고 가정합니다. 이 작업공간에서는 외부 프로젝트 파일을 읽거나 수정하지 않고, 학습에 필요한 파일만 현재 폴더 안에서 관리합니다.

## 중요한 클래스 이름 규칙

외부 프로젝트의 안전모 미착용 판단 로직은 클래스 이름 문자열에 의존하므로, 학습 데이터셋과 최종 모델의 클래스 이름은 반드시 아래 순서를 유지해야 합니다.

```yaml
0: helmet
1: person
2: vest
```

클래스 이름을 바꾸거나 순서를 바꾸면 외부 프로젝트에서 오동작할 수 있습니다.

## SH17 원본 클래스 매핑

이번 학습에서는 SH17 원본 데이터셋의 클래스 전체를 그대로 쓰지 않고, 아래 3개 클래스만 사용합니다.

```text
Person -> person
Helmet -> helmet
Safety-vest -> vest
```

실제 SH17 원본 클래스 목록 텍스트 파일은 [raw_datasets/sh17/새 텍스트 문서.txt](</C:/Users/AISW_203_114/Desktop/AI_Learn/raw_datasets/sh17/새 텍스트 문서.txt>) 에 있지만, 이 순서가 실제 YOLO 라벨 id와 일치하지 않을 수 있습니다.

따라서 이 작업공간에서는 [scripts/visualize_sh17_source_ids.py](</C:/Users/AISW_203_114/Desktop/AI_Learn/scripts/visualize_sh17_source_ids.py>) 로 source id preview를 직접 확인해 실제 id를 확정했습니다.

```text
Person = 0
Helmet = 10
Safety-vest = 16
```

중요: 원본 데이터셋의 class id를 그대로 사용하면 안 됩니다. 최종 YOLO 라벨은 반드시 아래 기준으로 다시 매핑해야 합니다.

```text
helmet = 0
person = 1
vest = 2
```

즉, 원본 `Safety-vest` 클래스는 최종 라벨에서 `vest`라는 이름과 `class id 2`로 저장되어야 합니다.

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

## YAML 파일 구분

이 작업공간에는 같은 데이터셋 구조를 서로 다른 실행 환경에서 쉽게 쓰기 위해 YAML 파일을 분리해 두었습니다.

- `datasets/hiyoung_ppe.yaml`: 기존 Colab 기준 YAML
- `datasets/hiyoung_ppe_colab.yaml`: Colab + Google Drive 기준 YAML
- `datasets/hiyoung_ppe_local.yaml`: 로컬 Windows 기준 YAML

기본 경로는 아래와 같습니다.

```text
hiyoung_ppe_colab.yaml -> /content/drive/MyDrive/AI_Learn/datasets/hiyoung_ppe
hiyoung_ppe_local.yaml -> C:/Users/AISW_203_114/Desktop/AI_Learn/datasets/hiyoung_ppe
```

로컬에서 데이터셋 검사를 할 때는 기본값으로 `datasets/hiyoung_ppe_local.yaml`이 사용됩니다. Colab에서 실행할 때는 `--yaml datasets/hiyoung_ppe_colab.yaml` 또는 기존 `datasets/hiyoung_ppe.yaml`을 사용하면 됩니다.

## Colab + Google Drive 기준 사용 순서

1. 이 폴더를 Google Drive의 `MyDrive/AI_Learn` 위치에 맞춰 두고 데이터셋을 `datasets/hiyoung_ppe/` 아래에 배치합니다.
2. `datasets/hiyoung_ppe.yaml`의 클래스 이름이 `helmet`, `person`, `vest` 순서인지 다시 확인합니다.
3. 로컬 Windows에서는 `python scripts/check_yolo_dataset.py`로 데이터셋 구조와 라벨 형식을 검사합니다.
4. Colab에서는 `python scripts/check_yolo_dataset.py --yaml datasets/hiyoung_ppe_colab.yaml` 또는 노트북 안의 명령으로 같은 검사를 수행합니다.
5. 검사 결과에서 클래스가 `helmet`, `person`, `vest` 3개로 인식되는지 확인합니다.
6. Colab에서 `scripts/train_yolo.py` 또는 `notebooks/train_hiyoung_ppe_colab.ipynb`를 사용해 학습합니다.
7. 학습 완료 후 생성된 `best.pt`를 `scripts/validate_model.py`로 불러와 클래스 이름과 예측 결과를 점검합니다.
8. 최종 확인이 끝난 가중치를 외부 프로젝트의 `safety_ai_monitor/models/weights/` 폴더로 복사합니다.
9. 외부 프로젝트에서는 `config.py`의 `MODEL_PATH`를 `hiyoung_helmet_person_vest_yolo11m.pt` 같은 파일명으로 맞춰 사용하면 됩니다.

## VSCode Colab 확장 실행 순서

학습은 로컬 Windows 터미널에서 실행하지 않고, VSCode의 Colab 확장을 통해 Colab 원격 런타임에서 진행합니다.

1. VSCode에서 `notebooks/train_hiyoung_ppe_colab.ipynb`를 엽니다.
2. Colab 확장으로 런타임에 연결하고 GPU 런타임을 선택합니다.
3. Google Drive mount 인증을 진행합니다.
4. 모든 경로는 로컬 `Desktop`이 아니라 `/content/drive/MyDrive/AI_Learn` 기준으로 실행합니다.
5. 먼저 데이터셋 검사 셀을 실행해 `datasets/hiyoung_ppe_colab.yaml` 기준 구조를 확인합니다.
6. `yolo11s`, `10 epoch` 테스트 학습 셀을 먼저 실행합니다.
7. 테스트 모델 검증 셀로 `best.pt`의 클래스 이름이 `helmet`, `person`, `vest`인지 확인합니다.
8. 테스트가 정상적으로 끝나면 `yolo11m` 본 학습 셀을 실행합니다.
9. 마지막으로 최종 `best.pt`를 `weights/hiyoung_helmet_person_vest_yolo11m.pt`로 복사합니다.

## SH17 변환 순서

SH17 원본 데이터셋은 `raw_datasets/sh17/` 아래에 두고, 원본 클래스 중 `Person`, `Helmet`, `Safety-vest`만 추려 최종 학습용 YOLO 데이터셋으로 변환합니다.

원본 클래스는 아래 기준으로 다시 매핑해야 합니다.

```text
Person (source id 0) -> person -> 1
Helmet (source id 10) -> helmet -> 0
Safety-vest (source id 16) -> vest -> 2
```

중요: 원본 class id를 그대로 쓰면 안 됩니다. 최종 YOLO 라벨 txt에는 반드시 `helmet=0`, `person=1`, `vest=2`가 저장되어야 하며, 그 외 클래스는 모두 제외합니다.

기존 `datasets/hiyoung_ppe/`가 이미 있다면, 잘못 변환된 라벨이 섞이지 않도록 삭제 후 다시 생성해야 합니다. 변환 스크립트는 `--overwrite` 옵션으로 이를 처리합니다.

```bash
python scripts/convert_sh17_to_hiyoung_ppe.py --overwrite
```

기본 경로는 아래를 사용합니다.

- `--src-root raw_datasets/sh17`
- `--dst-root datasets/hiyoung_ppe`
- `--train-list raw_datasets/sh17/train_files.txt`
- `--val-list raw_datasets/sh17/val_files.txt`

빈 라벨이 된 이미지도 유지하려면 아래 옵션을 추가합니다.

```bash
python scripts/convert_sh17_to_hiyoung_ppe.py --overwrite --keep-empty
```

학습 전에는 반드시 구조 검사와 라벨 시각화를 먼저 확인합니다.

1. `python scripts/check_yolo_dataset.py`
2. `python scripts/visualize_yolo_labels.py --split val --count 12`
3. `datasets/hiyoung_ppe_previews/` 아래 결과 이미지를 열어서 `helmet`, `person`, `vest` 라벨명이 실제 물체와 맞는지 확인합니다.
4. 그 다음에만 학습을 시작합니다.

중요: 학습 전 라벨 시각화 확인은 필수입니다. `val_batch_labels.jpg` 등에서 헬멧이 `vest`로 보였다면 변환 매핑이 잘못된 상태이므로, 학습을 진행하면 안 됩니다.

현재 생성된 테스트 가중치 `runs/test_helmet_person_vest_yolo11s/weights/best.pt` 는 잘못된 라벨로 학습되었으므로 사용하지 않습니다. 필요하면 올바른 라벨로 데이터셋을 다시 만든 뒤 새로 학습해야 합니다.

이번 수정 후 다시 실행할 명령은 아래와 같습니다.

```bash
python scripts/convert_sh17_to_hiyoung_ppe.py --overwrite
python scripts/check_yolo_dataset.py
python scripts/visualize_yolo_labels.py --split val --count 12
```

특정 클래스만 따로 확인하고 싶으면 `--class-id` 와 `--output-dir` 를 함께 사용할 수 있습니다. 예를 들어 vest만 따로 확인하려면 아래처럼 실행합니다.

```bash
python scripts/visualize_yolo_labels.py --split train --count 12 --class-id 2 --output-dir datasets/hiyoung_ppe_previews_vest
```

라벨 글씨가 작게 보여 preview 확인이 어렵다면, 큰 글씨 옵션으로 다시 렌더링할 수 있습니다. 아래 예시는 글자 크기와 선 두께를 키우고, 라벨 배경 박스를 함께 그려서 더 잘 보이게 만드는 명령입니다.

```bash
python scripts/visualize_yolo_labels.py --split val --count 12 --class-id 0 --font-scale 1.5 --font-thickness 4 --box-thickness 4 --output-dir datasets/preview_helmet_big
```

## 외부 프로젝트 반영 방법

학습이 끝나면 보통 `runs/.../weights/best.pt`가 생성됩니다. 이 파일을 외부 프로젝트의 아래 폴더로 복사합니다.

```text
safety_ai_monitor/models/weights/
```

복사 후 외부 프로젝트의 `config.py`에서 `MODEL_PATH`를 예를 들어 `hiyoung_helmet_person_vest_yolo11m.pt`로 지정하면 됩니다. 이 작업공간에는 외부 프로젝트 파일이 없으므로, 실제 반영은 해당 프로젝트에서 별도로 수행해야 합니다.

## Git 관리 주의사항

이 작업공간은 학습 실험 중간 산출물과 원본 데이터를 Git에서 제외하고, 최종 배포용 `.pt`만 선택적으로 포함하는 정책을 사용합니다.

- 원본 데이터셋과 변환 데이터셋은 Git에 올리지 않습니다.
- `datasets/hiyoung_ppe/`, `raw_datasets/`, `downloads/` 같은 데이터 폴더는 Git 제외 대상입니다.
- Colab 학습 결과인 `runs/`와 로그 폴더 `logs/`는 Git에 올리지 않습니다.
- 따라서 `runs/**/weights/best.pt`, `runs/**/weights/last.pt`도 Git에 포함되지 않습니다.
- 최종 배포 또는 공유용 `.pt` 파일만 `weights/` 폴더에 복사해서 Git에 포함합니다.
- 현재 정책상 `weights/*.pt`는 포함되지만 `*.onnx`, `*.engine`, `*.pth`는 제외합니다.

권장 사용 방식은 아래와 같습니다.

1. Colab 학습이 끝나면 `runs/.../weights/best.pt`를 확인합니다.
2. 최종본만 `weights/hiyoung_helmet_person_vest_yolo11m.pt`로 복사합니다.
3. 커밋할 때는 `weights/` 아래 최종 `.pt` 파일만 포함합니다.

파일 크기가 50MB 이상이면 일반 Git 대신 Git LFS 사용을 권장합니다. 예시는 아래와 같습니다.

```bash
git lfs install
git lfs track "weights/*.pt"
git add .gitattributes
```

이 저장소의 `.gitignore`에는 위 정책이 반영되어 있습니다.

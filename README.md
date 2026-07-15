# Machine Learning for Seakeeping Control in Waves

This repository contains the research outputs developed by **Sungshin Women's University** as a joint research institution under the project supervised by **Seoul National University**, supported by the **Lloyd's Register Foundation**.

본 저장소는 **Lloyd Register Foundation**의 지원을 받고 **서울대학교**가 주관하며, 공동연구기관인 **성신여자대학교**가 수행한 **"Machine Learning for Seakeeping Control in Waves"** 과제의 연구 결과물을 포함하고 있습니다.

---

## Project Overview (과제 개요)

- **Project Title (과제명):** Machine Learning for Seakeeping Control in Waves (파랑 중 내항 조종을 위한 기계학습 연구)
- **Supported by (지원 기관):** Lloyd's Register Foundation
- **Lead Institution (주관 기관):** Seoul National University (서울대학교)
- **Joint Research Institution (공동연구 기관):** Sungshin Women's University (성신여자대학교, School of AI Convergence)
- **Principal Investigator at SSWU (성신여대 공동연구 책임자):** Prof. Yangjun Ahn (안양준 교수)

---

## Research Scope & Key Contents (주요 연구 내용)

본 연구는 파랑 환경 하에서 선박의 안전한 운항 및 제어(Seakeeping Control)를 위해 인공지능 기법을 적용한 선박 운동 시계열 예측과 예측의 신뢰도 확보를 위한 불확실성 정량화(Uncertainty Quantification)를 다룹니다.

### 1. Theoretical Background (배경 이론)
- 파랑 중 선박 운동(Ship Motion in Waves)에 대한 수조 모형 시험 및 수치 시뮬레이션 데이터 분석 기반 구축.
- 주파수 영역 및 시간 영역에서의 비선형 선박 운동 시계열 데이터(6자유도 운동 등) 분석 기법 정의.

### 2. Deep Learning Models for Ship Motion Prediction (선박 운동 시계열 예측 AI 모델)
- 복잡한 선박 운동 시계열 데이터의 장단기 예측 성능을 확보하기 위해 최신 딥러닝 아키텍처 구현 및 비교 분석.
- 주요 모델 구성:
  - **Time-CNN**: 다변량 시계열의 시공간적 특징을 포착하기 위한 2D Convolutional Neural Network 기반 아키텍처.
  - **TimesNet**: 시계열의 다중 주기성(Multi-periodicity) 분석을 위해 1차원 시계열을 2차원 공간으로 변환하여 특징을 학습하는 구조.
  - **TCN (Temporal Convolutional Network)**: 인과성(Causality)을 유지하면서 시계열 패턴을 추출하는 확장 합성곱(Dilated Convolution) 기반 모델.

### 3. Uncertainty Quantification (UQ, 불확실성 정량화)
- 안전 보장 및 신뢰도 높은 제어를 위해 AI 모델 예측값의 우연적 불확실성(Aleatoric Uncertainty) 및 인식적 불확실성(Epistemic Uncertainty) 정량화 기법 설계.
- Monte Carlo Dropout 및 Bayesian Neural Network를 활용한 예측 구간(Prediction Interval, Confidence Band) 산출 연구 탑재.

### 4. Data Sufficiency Analysis (해석 필요 데이터양 실험)
- 파랑 시뮬레이션 해석 및 예측 모델 학습 시, 요구되는 적정 수준의 데이터양과 성능 간의 Trade-off 분석 실험.
- 제한된 관측 데이터 환경에서 신뢰할 수 있는 예측 성능을 확보하기 위한 최소 데이터 요건 검토.

---

## Repository Structure (디렉토리 구조)

```text
lrf-sswu/
├── README.md
├── ComrisonOfModels.ipynb   # 모델 비교·분석 노트북
├── dataset/                 # 해상 조건별 선박 운동 시계열 데이터
│   ├── 000_Tm_*_HS_*_V_*.csv|.out
│   ├── ...
│   └── Wave_180_Tm_*_HS_*_V_*.csv|.out
│       # 파일명 규칙:
│       #   heading(deg)_Tm_(s)_HS_(m)_V_(knots)
│       #   heading: 0–180° (15° 간격)
│       #   Tm: modal period, HS: significant wave height, V: ship speed
└── dataset_grid/            # 그리드 조건 기반 시계열 데이터
    ├── 090_Tm_*_HS_*.csv|.out
    └── Wave_*_Tm_*_HS_*.csv|.out
```

### File / Folder Description

| Path | Description |
|------|-------------|
| `ComrisonOfModels.ipynb` | Time-CNN, TimesNet, TCN 등 선박 운동 예측 모델 비교 및 분석 노트북 |
| `dataset/` | 해상상태(heading, Tm, Hs, V)별 6자유도 운동 시계열 데이터 (CSV / OUT) |
| `dataset_grid/` | 격자형으로 구성된 해석 조건의 시계열 데이터 (CSV / OUT) |

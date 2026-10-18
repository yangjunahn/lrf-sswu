"""
Causal ModernTCN 기반 6-DOF 시계열 예측 + Optuna 하이퍼파라미터 최적화
- ID(2Hz 시뮬레이션) 데이터 로딩 → 조건 단위 train/val/test 분할
- z-score 정규화 (train 통계만 사용)
- 1단계: Optuna로 모델/학습 하이퍼파라미터 탐색 (val MSE 최소화, MedianPruner 적용)
- 2단계: 최적 파라미터로 전체 재학습 (조기 종료) 후 테스트 평가 및 시각화

"""

import json
import random
from pathlib import Path

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler

# ============================================================
# 0. Device & Seed
# ============================================================
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

BASE_SEED = 42
set_seed(BASE_SEED)

# ============================================================
# 1. Configuration
# ============================================================
ID_DATA_DIR = Path("csv_dataset")

IN_LEN = 128
OUT_LEN = 128
STRIDE = 16
ID_HZ = 2.0
DT_SEC = 1.0 / ID_HZ

TRAIN_RATIO, VAL_RATIO, TEST_RATIO = 0.70, 0.15, 0.15

EXP_NAME = f"CausalModernTCN_optuna_in{IN_LEN}_out{OUT_LEN}_stride{STRIDE}"
RESULT_DIR = Path("experiments0904") / EXP_NAME
MODEL_DIR = RESULT_DIR / "models"
PLOT_DIR = RESULT_DIR / "plots"
for d in (MODEL_DIR, PLOT_DIR):
    d.mkdir(parents=True, exist_ok=True)

MODEL_SAVE_PATH = MODEL_DIR / "best_model.pt"
SCALER_SAVE_PATH = RESULT_DIR / "scaler.json"
HISTORY_SAVE_PATH = RESULT_DIR / "history.json"
BEST_PARAMS_PATH = RESULT_DIR / "best_params.json"
STUDY_DB_PATH = RESULT_DIR / "optuna_study.db"
TRIALS_CSV_PATH = RESULT_DIR / "optuna_trials.csv"

# Optuna 설정
OPTUNA_CONFIG = {
    "n_trials": 40,            # 탐색 trial 수
    "timeout_sec": None,       # 전체 탐색 시간 제한(초), None이면 제한 없음
    "trial_max_epochs": 40,    # trial당 최대 epoch (탐색 단계는 짧게)
    "trial_patience": 10,       # trial 내 조기 종료 patience
    "pruner_warmup_epochs": 8, # 이 epoch 이후부터 pruning 판단
    "n_startup_trials": 8,     # 초기 랜덤 탐색 trial 수
    "num_workers": 4,
    # 중단(Ctrl+C) 처리
    "finish_partial_trial": True,   # 중단 시 진행 중 trial을 현재 best val로 COMPLETE 처리
    "train_after_interrupt": False, # True면 중단 후에도 지금까지의 best로 2단계 재학습 진행
    "skip_search": False,           # True면 탐색 생략, DB의 best_params로 바로 2단계 진행
    "verbose_epoch": True,          # trial 내 epoch별 손실 출력
    "verbose_params": True,         # trial 시작 시 샘플링된 하이퍼파라미터 출력
}

# 최종 재학습 설정
FINAL_TRAIN_CONFIG = {
    "max_epochs": 300,
    "patience": 10,
    "num_workers": 4,
}

dof_cols = ["x1", "x2", "x3", "x4", "x5", "x6"]
dof_names = ["Surge (X)", "Sway (Y)", "Heave (Z)", "Roll", "Pitch", "Yaw"]
dof_units = ["m", "m", "m", "deg", "deg", "deg"]

# ============================================================
# 2. Data Loading & Preprocessing
# ============================================================
def load_id_data(data_dir):
    file_paths = sorted([p for p in data_dir.glob("*.csv") if not p.name.startswith("Wave_")])
    if not file_paths:
        raise RuntimeError(f"{data_dir}에서 CSV 파일을 찾지 못했습니다.")

    all_arrays, kept_names = [], []
    for path in tqdm(file_paths, desc="Loading ID CSVs"):
        try:
            df = pd.read_csv(path, usecols=dof_cols)
        except ValueError:
            df = pd.read_csv(path)
            df = df.iloc[:, -6:]
        arr = df.to_numpy(dtype=np.float32)
        if len(arr) >= IN_LEN + OUT_LEN and np.isfinite(arr).all():
            all_arrays.append(arr)
            kept_names.append(path.name)
    print(f" - 유효 조건(파일) 수: {len(all_arrays)}")
    return all_arrays, kept_names


def make_window_meta(condition_indices, all_arrays):
    meta = []
    for file_idx in condition_indices:
        n = len(all_arrays[file_idx])
        max_start = n - IN_LEN - OUT_LEN
        for start_idx in range(0, max_start + 1, STRIDE):
            meta.append((file_idx, start_idx))
    return np.asarray(meta, dtype=np.int64) if meta else np.empty((0, 2), dtype=np.int64)


class TimeWindowDataset(Dataset):
    def __init__(self, raw_arrays, meta, data_mean, data_std):
        self.raw_arrays = raw_arrays
        self.meta = meta
        self.data_mean = data_mean
        self.data_std = data_std

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, idx):
        file_idx, start_idx = self.meta[idx]
        e_in = start_idx + IN_LEN
        e_out = e_in + OUT_LEN
        raw_arr = self.raw_arrays[file_idx]
        x_scaled = (raw_arr[start_idx:e_in] - self.data_mean) / self.data_std
        y_scaled = (raw_arr[e_in:e_out] - self.data_mean) / self.data_std
        return torch.from_numpy(x_scaled).float(), torch.from_numpy(y_scaled).float()


def make_loaders(train_ds, val_ds, test_ds, batch_size, num_workers):
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader, test_loader

# ============================================================
# 3. Causal ModernTCN 모델 정의
# ============================================================
class PermuteToChannelsFirst(nn.Module):
    def forward(self, x):
        return x.transpose(1, 2)


class PermuteToTimeFirst(nn.Module):
    def forward(self, x):
        return x.transpose(1, 2)


class ChannelLayerNorm(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.norm = nn.LayerNorm(channels)

    def forward(self, x):
        return self.norm(x.transpose(1, 2)).transpose(1, 2)


class CausalConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, groups=1, bias=True):
        super().__init__()
        if kernel_size < 1:
            raise ValueError("kernel_size는 1 이상이어야 합니다.")
        self.left_padding = kernel_size - 1
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size,
                              padding=0, groups=groups, bias=bias)

    def forward(self, x):
        x = F.pad(x, (self.left_padding, 0))
        return self.conv(x)


class ModernTCNBlock(nn.Module):
    def __init__(self, channels, kernel_size=31, expansion=2.0, dropout=0.1):
        super().__init__()
        hidden_dim = int(channels * expansion)
        self.dwconv = CausalConv1d(channels, channels, kernel_size, groups=channels)
        self.norm = ChannelLayerNorm(channels)
        self.pwffn = nn.Sequential(
            nn.Conv1d(channels, hidden_dim, kernel_size=1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(hidden_dim, channels, kernel_size=1),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        residual = x
        x = self.dwconv(x)
        x = self.norm(x)
        x = self.pwffn(x)
        return x + residual


class ModernTCNBackbone(nn.Module):
    def __init__(self, in_dof=6, channels=128, num_blocks=6, kernel_size=31,
                 expansion=2.0, dropout=0.1, stem_kernel_size=7):
        super().__init__()
        self.input_proj = nn.Sequential(
            PermuteToChannelsFirst(),
            CausalConv1d(in_dof, channels, stem_kernel_size),
        )
        self.blocks = nn.ModuleList([
            ModernTCNBlock(channels, kernel_size, expansion, dropout)
            for _ in range(num_blocks)
        ])
        self.out_permute = PermuteToTimeFirst()

    def forward(self, x):
        x = self.input_proj(x)
        for block in self.blocks:
            x = block(x)
        return self.out_permute(x)


class ModernTCNRegressor(nn.Module):
    def __init__(self, in_dof=6, out_dof=6, in_len=128, out_len=64, channels=128,
                 num_blocks=6, kernel_size=31, expansion=2.0, dropout=0.1,
                 stem_kernel_size=7, head_dropout=0.1):
        super().__init__()
        self.backbone = ModernTCNBackbone(in_dof, channels, num_blocks, kernel_size,
                                          expansion, dropout, stem_kernel_size)
        self.temporal_projection = nn.Sequential(
            nn.Linear(in_len, out_len),
            nn.GELU(),
            nn.Dropout(head_dropout),
        )
        self.output_projection = nn.Linear(channels, out_dof)

    def forward(self, src):
        x = self.backbone(src)            # (B, IN_LEN, C)
        x = x.transpose(1, 2)             # (B, C, IN_LEN)
        x = self.temporal_projection(x)   # (B, C, OUT_LEN)
        x = x.transpose(1, 2)             # (B, OUT_LEN, C)
        return self.output_projection(x)  # (B, OUT_LEN, DOF)


def create_model(params):
    return ModernTCNRegressor(
        in_dof=6, out_dof=6, in_len=IN_LEN, out_len=OUT_LEN,
        channels=params["channels"],
        num_blocks=params["num_blocks"],
        kernel_size=params["kernel_size"],
        expansion=params["expansion"],
        dropout=params["dropout"],
        stem_kernel_size=params["stem_kernel_size"],
        head_dropout=params["head_dropout"],
    )


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# ============================================================
# 4. 학습/평가 루프
# ============================================================
def train_epoch(model, loader, optimizer, criterion, show_bar=True):
    model.train()
    epoch_loss = 0.0
    for src, trg in tqdm(loader, desc="Train", leave=False, disable=not show_bar):
        src, trg = src.to(device), trg.to(device)
        optimizer.zero_grad()
        pred = model(src)
        loss = criterion(pred, trg)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        epoch_loss += loss.item()
    return epoch_loss / len(loader)


@torch.no_grad()
def evaluate_epoch(model, loader, criterion, show_bar=True):
    model.eval()
    epoch_loss, epoch_mae = 0.0, 0.0
    for src, trg in tqdm(loader, desc="Eval", leave=False, disable=not show_bar):
        src, trg = src.to(device), trg.to(device)
        pred = model(src)
        epoch_loss += criterion(pred, trg).item()
        epoch_mae += F.l1_loss(pred, trg).item()
    return epoch_loss / len(loader), epoch_mae / len(loader)


def build_optimizer_and_scheduler(model, params):
    optimizer = optim.AdamW(model.parameters(),
                            lr=params["learning_rate"],
                            weight_decay=params["weight_decay"])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3)
    return optimizer, scheduler

# ============================================================
# 5. Optuna 탐색 공간 및 objective
# ============================================================
def sample_params(trial):
    """탐색 공간 정의. 반환 dict는 create_model / optimizer 설정에 그대로 사용."""
    return {
        # 모델 구조
        "channels": trial.suggest_categorical("channels", [64, 96, 128, 192, 256]),
        "num_blocks": trial.suggest_int("num_blocks", 2, 8),
        "kernel_size": trial.suggest_categorical("kernel_size", [7, 15, 23, 31, 39, 47, 55, 63]),
        "expansion": trial.suggest_categorical("expansion", [1.0, 2.0, 3.0, 4.0]),
        "dropout": trial.suggest_float("dropout", 0.0, 0.3, step=0.05),
        "stem_kernel_size": trial.suggest_categorical("stem_kernel_size", [3, 5, 7, 11, 15, 21]),
        "head_dropout": trial.suggest_float("head_dropout", 0.0, 0.3, step=0.05),
        # 학습
        "learning_rate": trial.suggest_float("learning_rate", 1e-4, 5e-3, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 1e-5, 1e-2, log=True),
        "batch_size": trial.suggest_categorical("batch_size", [32, 64, 128]),
    }


def make_objective(train_ds, val_ds, test_ds):
    def objective(trial):
        set_seed(BASE_SEED)
        params = sample_params(trial)

        train_loader, val_loader, _ = make_loaders(
            train_ds, val_ds, test_ds, params["batch_size"], OPTUNA_CONFIG["num_workers"])

        model = create_model(params).to(device)
        n_params = count_parameters(model)
        trial.set_user_attr("n_params", n_params)
        criterion = nn.MSELoss()
        optimizer, scheduler = build_optimizer_and_scheduler(model, params)

        if OPTUNA_CONFIG["verbose_params"]:
            pstr = ", ".join(f"{k}={v}" for k, v in params.items())
            print(f"\n[Trial {trial.number}] 시작 | params({n_params:,}): {pstr}")

        best_val = float("inf")
        patience_counter = 0
        try:
            for epoch in range(1, OPTUNA_CONFIG["trial_max_epochs"] + 1):
                train_loss = train_epoch(model, train_loader, optimizer, criterion, show_bar=False)
                val_loss, _ = evaluate_epoch(model, val_loader, criterion, show_bar=False)
                scheduler.step(val_loss)

                if not np.isfinite(val_loss):
                    raise optuna.TrialPruned()

                # 중간 결과를 DB에 기록 (report는 intermediate_values, user_attr은 진행 상황)
                trial.report(val_loss, step=epoch)
                if val_loss < best_val:
                    best_val = val_loss
                    patience_counter = 0
                    improved = True
                else:
                    patience_counter += 1
                    improved = False
                trial.set_user_attr("epochs_done", epoch)
                trial.set_user_attr("best_val_so_far", float(best_val))

                will_prune = trial.should_prune()
                if OPTUNA_CONFIG["verbose_epoch"]:
                    cur_lr = optimizer.param_groups[0]["lr"]
                    tag = " *" if improved else ""
                    tag += " [prune]" if will_prune else ""
                    print(f"  [Trial {trial.number}] epoch {epoch:3d}/"
                          f"{OPTUNA_CONFIG['trial_max_epochs']} | "
                          f"train {train_loss:.5f} | val {val_loss:.5f} | "
                          f"best {best_val:.5f} | lr {cur_lr:.2e}{tag}")

                if will_prune:
                    raise optuna.TrialPruned()
                if patience_counter >= OPTUNA_CONFIG["trial_patience"]:
                    if OPTUNA_CONFIG["verbose_epoch"]:
                        print(f"  [Trial {trial.number}] trial 내 조기 종료 "
                              f"(patience {OPTUNA_CONFIG['trial_patience']})")
                    break
        except KeyboardInterrupt:
            # 진행 중이던 trial: 현재까지의 best val로 COMPLETE 처리 후 study 중지
            trial.set_user_attr("interrupted", True)
            if OPTUNA_CONFIG["finish_partial_trial"] and np.isfinite(best_val):
                print(f"\n[중단] trial #{trial.number}: "
                      f"{trial.user_attrs.get('epochs_done', 0)} epoch까지의 best val "
                      f"{best_val:.5f}로 기록 후 탐색 종료")
                _cleanup(model, optimizer, scheduler)
                trial.study.stop()
                return best_val
            # 완료된 epoch이 없으면 그대로 중단 (이 trial은 DB에 FAIL로 남음)
            _cleanup(model, optimizer, scheduler)
            raise

        _cleanup(model, optimizer, scheduler)
        if OPTUNA_CONFIG["verbose_epoch"]:
            print(f"[Trial {trial.number}] 완료 | best val {best_val:.5f} "
                  f"({trial.user_attrs.get('epochs_done', 0)} epoch)")
        return best_val
    return objective


def _cleanup(*objs):
    for o in objs:
        del o
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def save_study_results(study):
    """trial CSV와 best_params.json을 갱신. 완료된 trial이 없으면 best는 생략."""
    study.trials_dataframe().to_csv(TRIALS_CSV_PATH, index=False)
    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    payload = {
        "n_trials_total": len(study.trials),
        "n_trials_complete": len(completed),
    }
    if completed:
        payload.update({
            "best_value": study.best_value,
            "best_trial": study.best_trial.number,
            "best_params": study.best_params,
        })
    with open(BEST_PARAMS_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def _save_callback(study, trial):
    """매 trial 종료 시 호출: 중단되어도 직전 trial까지 파일에 남도록 함."""
    save_study_results(study)


def run_study(train_ds, val_ds, test_ds):
    # epoch 로그를 켜면 Optuna 기본 trial 로그와 겹치므로 경고 수준으로 낮춤
    if OPTUNA_CONFIG["verbose_epoch"]:
        optuna.logging.set_verbosity(optuna.logging.WARNING)

    sampler = TPESampler(seed=BASE_SEED,
                         n_startup_trials=OPTUNA_CONFIG["n_startup_trials"])
    pruner = MedianPruner(n_startup_trials=OPTUNA_CONFIG["n_startup_trials"],
                          n_warmup_steps=OPTUNA_CONFIG["pruner_warmup_epochs"])
    study = optuna.create_study(
        study_name=EXP_NAME,
        direction="minimize",
        sampler=sampler,
        pruner=pruner,
        storage=f"sqlite:///{STUDY_DB_PATH}",
        load_if_exists=True,   # 중단 후 재실행 시 이어서 탐색
    )
    # 재개 시 남은 trial 수만 실행 (진행 중이던 RUNNING 상태 trial은 제외)
    n_done = sum(t.state != optuna.trial.TrialState.RUNNING for t in study.trials)
    n_remaining = max(0, OPTUNA_CONFIG["n_trials"] - n_done)
    if n_done > 0:
        print(f" - 기존 study 재개: {n_done}개 trial 기록 있음, {n_remaining}개 추가 실행")

    interrupted = False
    if n_remaining > 0:
        try:
            study.optimize(
                make_objective(train_ds, val_ds, test_ds),
                n_trials=n_remaining,
                timeout=OPTUNA_CONFIG["timeout_sec"],
                gc_after_trial=True,
                show_progress_bar=not OPTUNA_CONFIG["verbose_epoch"],
                callbacks=[_save_callback],
            )
        except KeyboardInterrupt:
            interrupted = True
            print("\n[중단] 사용자 요청으로 탐색을 중단합니다. 직전 trial까지의 기록은 저장되어 있습니다.")

    # study.stop()으로 끝난 경우도 중단으로 간주
    if any(t.user_attrs.get("interrupted") for t in study.trials):
        interrupted = True

    save_study_results(study)

    n_pruned = sum(t.state == optuna.trial.TrialState.PRUNED for t in study.trials)
    n_complete = sum(t.state == optuna.trial.TrialState.COMPLETE for t in study.trials)
    print(f"\nOptuna 탐색 {'중단' if interrupted else '완료'}: "
          f"complete {n_complete} / pruned {n_pruned} / total {len(study.trials)}")
    if n_complete > 0:
        print(f" - best trial #{study.best_trial.number}, val MSE {study.best_value:.5f}")
        for k, v in study.best_params.items():
            print(f"     {k:18s}: {v}")
    print(f" - 저장 위치: {STUDY_DB_PATH}, {TRIALS_CSV_PATH}, {BEST_PARAMS_PATH}")
    return study, interrupted


def load_study_only():
    """탐색 생략 시 기존 DB에서 study만 로드."""
    if not STUDY_DB_PATH.exists():
        raise RuntimeError(f"skip_search=True이지만 study DB가 없습니다: {STUDY_DB_PATH}")
    return optuna.load_study(study_name=EXP_NAME, storage=f"sqlite:///{STUDY_DB_PATH}")


def plot_study(study, save_dir):
    """optimization history와 param importance를 matplotlib으로 저장."""
    try:
        fig = optuna.visualization.matplotlib.plot_optimization_history(study).figure
        fig.tight_layout()
        fig.savefig(save_dir / "optuna_history.png", dpi=200)
        plt.close(fig)

        fig = optuna.visualization.matplotlib.plot_param_importances(study).figure
        fig.tight_layout()
        fig.savefig(save_dir / "optuna_param_importance.png", dpi=200)
        plt.close(fig)
        print(f" Optuna 시각화 저장: {save_dir}")
    except Exception as e:
        print(f" Optuna 시각화 생략: {e}")

# ============================================================
# 6. 시각화 유틸리티
# ============================================================
def plot_training_curves(history, save_path):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(history["train_loss"], label="Train MSE")
    ax.plot(history["val_loss"], label="Val MSE")
    best_ep = int(np.argmin(history["val_loss"]))
    ax.axvline(best_ep, color="red", ls="--", lw=0.8, label=f"Best epoch ({best_ep + 1})")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE (z-score)")
    ax.set_title("Training / Validation Loss")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)
    print(f" 학습 곡선 저장: {save_path}")


def plot_test_predictions(model, dataset, id_file_names, data_mean, data_std,
                          n_samples, save_dir):
    model.eval()
    n_total = len(dataset)
    if n_total == 0:
        print(" 테스트 윈도우가 없어 시각화를 건너뜁니다.")
        return
    sample_ids = np.linspace(0, n_total - 1, min(n_samples, n_total), dtype=int)

    t_in = np.arange(-IN_LEN, 0) * DT_SEC
    t_out = np.arange(OUT_LEN) * DT_SEC

    for s_idx in sample_ids:
        src, trg = dataset[s_idx]
        with torch.no_grad():
            pred = model(src.unsqueeze(0).to(device)).squeeze(0).cpu().numpy()
        src, trg = src.numpy(), trg.numpy()

        src_phys = src * data_std + data_mean
        trg_phys = trg * data_std + data_mean
        pred_phys = pred * data_std + data_mean

        # 이 윈도우의 원본 파일과 시작 위치
        file_idx, start_idx = int(dataset.meta[s_idx][0]), int(dataset.meta[s_idx][1])
        src_name = Path(id_file_names[file_idx]).stem

        fig, axes = plt.subplots(3, 2, figsize=(16, 12))
        for i in range(6):
            ax = axes[i // 2, i % 2]
            ax.plot(t_in, src_phys[:, i], ls="--", alpha=0.6, color="gray", label="Input (History)")
            ax.plot(t_out, trg_phys[:, i], lw=1.5, color="green", label="Truth")
            ax.plot(t_out, pred_phys[:, i], lw=1.5, color="red", label="Pred")
            ax.axvline(0, color="k", ls=":", alpha=0.5)
            ax.set_ylabel(f"{dof_names[i]} ({dof_units[i]})")
            ax.grid(True, alpha=0.3)

            # x축 눈금/라벨은 마지막 행(아래쪽 두 서브플롯)에만
            if i // 2 == 2:
                ax.set_xlabel("Time (s)")
            else:
                ax.tick_params(labelbottom=False)

            if i == 0:
                ax.legend(loc="upper left", fontsize=9)

        fig.tight_layout()
        save_path = save_dir / f"test_pred_{src_name}_start{start_idx}.png"
        fig.savefig(save_path, dpi=500)
        plt.close(fig)
        print(f" 예측 시각화 저장: {save_path}")

# ============================================================
# 7. Main
# ============================================================
if __name__ == "__main__":
    # ---------- 데이터 준비 ----------
    print("데이터 로딩 중...")
    all_arrays, id_file_names = load_id_data(ID_DATA_DIR)
    num_conditions = len(all_arrays)

    condition_indices = np.arange(num_conditions)
    np.random.shuffle(condition_indices)
    n_test = max(1, int(round(num_conditions * TEST_RATIO)))
    n_val = max(1, int(round(num_conditions * VAL_RATIO)))
    test_idx = condition_indices[:n_test]
    val_idx = condition_indices[n_test:n_test + n_val]
    train_idx = condition_indices[n_test + n_val:]
    print(f" - 분할: train {len(train_idx)} / val {len(val_idx)} / test {len(test_idx)} 조건")

    train_concat = np.concatenate([all_arrays[i] for i in train_idx], axis=0)
    data_mean = train_concat.mean(axis=0).astype(np.float32)
    data_std = train_concat.std(axis=0).astype(np.float32)
    data_std[data_std == 0] = 1.0

    with open(SCALER_SAVE_PATH, "w", encoding="utf-8") as f:
        json.dump({
            "dof_cols": dof_cols,
            "mean": data_mean.tolist(),
            "std": data_std.tolist(),
            "train_files": [id_file_names[i] for i in train_idx],
            "val_files": [id_file_names[i] for i in val_idx],
            "test_files": [id_file_names[i] for i in test_idx],
        }, f, indent=2, ensure_ascii=False)
    print(f" 스케일러/분할 정보 저장: {SCALER_SAVE_PATH}")

    train_meta = make_window_meta(train_idx, all_arrays)
    val_meta = make_window_meta(val_idx, all_arrays)
    test_meta = make_window_meta(test_idx, all_arrays)
    print(f" - 윈도우: train {len(train_meta)} / val {len(val_meta)} / test {len(test_meta)}")

    train_ds = TimeWindowDataset(all_arrays, train_meta, data_mean, data_std)
    val_ds = TimeWindowDataset(all_arrays, val_meta, data_mean, data_std)
    test_ds = TimeWindowDataset(all_arrays, test_meta, data_mean, data_std)

    # ---------- 1단계: Optuna 탐색 ----------
    print("\n[1단계] Optuna 하이퍼파라미터 탐색")
    if OPTUNA_CONFIG["skip_search"]:
        study, interrupted = load_study_only(), False
        print(" - 탐색 생략, 기존 study의 best_params 사용")
    else:
        study, interrupted = run_study(train_ds, val_ds, test_ds)

    n_complete = sum(t.state == optuna.trial.TrialState.COMPLETE for t in study.trials)
    if n_complete == 0:
        raise SystemExit("완료된 trial이 없어 2단계를 진행할 수 없습니다.")
    plot_study(study, PLOT_DIR)

    if interrupted and not OPTUNA_CONFIG["train_after_interrupt"]:
        print("\n탐색이 중단되어 2단계 재학습을 생략합니다. "
              "같은 스크립트를 다시 실행하면 탐색이 이어지고, "
              "skip_search=True로 실행하면 현재 best로 바로 재학습합니다.")
        raise SystemExit(0)
    best_params = study.best_params

    # ---------- 2단계: 최적 파라미터로 재학습 ----------
    print("\n[2단계] 최적 파라미터로 전체 재학습")
    set_seed(BASE_SEED)
    train_loader, val_loader, test_loader = make_loaders(
        train_ds, val_ds, test_ds, best_params["batch_size"], FINAL_TRAIN_CONFIG["num_workers"])

    model = create_model(best_params).to(device)
    print(f"모델 파라미터 수: {count_parameters(model):,}")
    criterion = nn.MSELoss()
    optimizer, scheduler = build_optimizer_and_scheduler(model, best_params)

    history = {"train_loss": [], "val_loss": [], "val_mae": [], "lr": []}
    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(1, FINAL_TRAIN_CONFIG["max_epochs"] + 1):
        train_loss = train_epoch(model, train_loader, optimizer, criterion)
        val_loss, val_mae = evaluate_epoch(model, val_loader, criterion)
        scheduler.step(val_loss)
        cur_lr = optimizer.param_groups[0]["lr"]

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_mae"].append(val_mae)
        history["lr"].append(cur_lr)

        improved = val_loss < best_val_loss
        if improved:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
        else:
            patience_counter += 1

        marker = " (best 저장)" if improved else ""
        print(f"[Epoch {epoch:3d}/{FINAL_TRAIN_CONFIG['max_epochs']}] "
              f"train MSE {train_loss:.5f} | val MSE {val_loss:.5f} | "
              f"val MAE {val_mae:.5f} | lr {cur_lr:.2e}{marker}")

        if patience_counter >= FINAL_TRAIN_CONFIG["patience"]:
            print(f"조기 종료: {FINAL_TRAIN_CONFIG['patience']} epoch 동안 val 개선 없음")
            break

    with open(HISTORY_SAVE_PATH, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    plot_training_curves(history, PLOT_DIR / "training_curves.png")

    # ---------- 테스트 평가 ----------
    print("\n테스트 평가 (best 가중치 로드)")
    model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=device))
    test_loss, test_mae = evaluate_epoch(model, test_loader, criterion)
    print(f" - Test MSE (z-score): {test_loss:.5f}")
    print(f" - Test MAE (z-score): {test_mae:.5f}")

    model.eval()
    sq_err_sum = np.zeros(6, dtype=np.float64)
    n_elems = 0
    with torch.no_grad():
        for src, trg in test_loader:
            pred = model(src.to(device)).cpu().numpy()
            trg = trg.numpy()
            err_phys = (pred - trg) * data_std
            sq_err_sum += (err_phys ** 2).sum(axis=(0, 1))
            n_elems += pred.shape[0] * pred.shape[1]
    rmse_per_dof = np.sqrt(sq_err_sum / n_elems)
    print(" - DOF별 Test RMSE (물리 단위):")
    for name, unit, r in zip(dof_names, dof_units, rmse_per_dof):
        print(f"     {name:12s}: {r:.4f} {unit}")

    # 테스트 결과도 best_params.json에 병합
    with open(BEST_PARAMS_PATH, "r", encoding="utf-8") as f:
        summary = json.load(f)
    summary.update({
        "final_best_val_mse": best_val_loss,
        "test_mse_zscore": test_loss,
        "test_mae_zscore": test_mae,
        "test_rmse_per_dof": {n: float(r) for n, r in zip(dof_names, rmse_per_dof)},
    })
    with open(BEST_PARAMS_PATH, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # ---------- 예측 시각화 ----------
    print("\n테스트 예측 시각화")
    plot_test_predictions(model, test_ds, id_file_names, data_mean, data_std,
                          n_samples=4, save_dir=PLOT_DIR)

    print(f"\n완료. 결과 폴더: {RESULT_DIR}")
    print(f" - 모델 가중치   : {MODEL_SAVE_PATH}")
    print(f" - 최적 파라미터 : {BEST_PARAMS_PATH}")
    print(f" - Optuna DB     : {STUDY_DB_PATH}")
    print(f" - trial 로그    : {TRIALS_CSV_PATH}")
    print(f" - 그래프        : {PLOT_DIR}")

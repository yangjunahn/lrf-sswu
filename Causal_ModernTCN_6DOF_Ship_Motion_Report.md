# Analysis Report on Causal ModernTCN-Based 6-DOF Ship Motion Time-Series Prediction and Hyperparameter Optimization Code

## 1. Research Objective and Implementation Overview

This code conducts experiments to predict future six-degree-of-freedom (6-DOF) ship motions using historical 6-DOF ship-motion time-series data as input. The target motion components are Surge, Sway, Heave, Roll, Pitch, and Yaw. The task is formulated as a multivariate multi-step forecasting problem in which all six variables are jointly used as inputs and jointly predicted as outputs.

The model extracts features from historical ship motions using causal one-dimensional convolutions, depthwise convolutions, channel normalization, and residual connections. Subsequently, a linear transformation along the temporal dimension generates the future prediction horizon, and an output linear layer produces the predicted 6-DOF motion values. In the code, this model is referred to as the Causal ModernTCN.

The experiment is conducted in two stages. In the first stage, Optuna is used to explore the model architecture and training configurations. In the second stage, the model is initialized and retrained using the optimal configuration identified through the hyperparameter search. The final test performance is then evaluated using the model weights corresponding to the lowest validation loss.

## 2. Prediction Problem and Main Experimental Settings

### 2.1 Definition of Inputs and Outputs

Let \(L_{in}\) denote the input sequence length, \(L_{out}\) denote the output sequence length, and \(D\) denote the number of degrees of freedom. The model input and output are defined as follows:

\[
X \in \mathbb{R}^{B \times L_{in} \times D}, \qquad
\hat{Y} \in \mathbb{R}^{B \times L_{out} \times D}
\]

where \(B\) denotes the batch size. In this experiment, \(L_{in}=128\), \(L_{out}=128\), and \(D=6\). The sampling frequency of the dataset is 2 Hz, corresponding to a sampling interval of 0.5 s. Therefore, both the input and output sequences consist of 128 samples corresponding to 64 s.

### 2.2 Fixed Experimental Settings

| Setting | Value | Description |
|---|---|---|
| Data path | `csv_dataset` | Relative path used to read the CSV files |
| Sampling frequency | 2 Hz | Sampling frequency of the data |
| Input length | 128 samples | Historical 64 s |
| Output length | 128 samples | Future 64 s |
| Window stride | 16 samples | 8 s interval |
| Number of input/output variables | 6 each | 6-IN → 6-OUT |
| Validation/test ratio | 15% each | Rounded based on the number of files |
| Training ratio | Remaining approximately 70% | Files remaining after excluding validation and test files |
| Default seed | 42 | Applied to Python, NumPy, and PyTorch |
| Computing device | CUDA preferred; CPU otherwise | CUDA device is `cuda:0` |
| Optimization objective | Minimization of validation MSE | Evaluated in the normalized space |

### 2.3 Variable Configuration

| Original Column Name | Motion Component | Unit Used for Visualization and Evaluation |
|---|---|---|
| x1 | Surge | m |
| x2 | Sway | m |
| x3 | Heave | m |
| x4 | Roll | deg |
| x5 | Pitch | deg |
| x6 | Yaw | deg |

## 3. Data Loading and Preprocessing

### 3.1 CSV File Selection and Validity Check

The `load_id_data()` function reads CSV files located directly under the data directory after sorting them by filename. Files whose names begin with `Wave_` are excluded; therefore, the wave files themselves are not included as model inputs. Sea-state or operating-condition information contained in the filenames is not extracted or used as input features.

### 3.2 Condition-Level Data Splitting

The code treats each valid file as one experimental condition. The file indices are randomly shuffled and then assigned to the test, validation, and training sets, in that order. Sliding windows are generated separately within each subset after the file-level split. Consequently, overlapping windows extracted from the same file cannot be simultaneously assigned to the training and test sets. This procedure prevents time-series overlap leakage that could occur if windows were generated first and then randomly divided into different subsets.

### 3.3 Z-Score Normalization Based on Training Statistics

For each degree of freedom, the mean and standard deviation are calculated from an array constructed by concatenating all rows from the training files. The normalization statistics are calculated exclusively from the training data and are subsequently applied without modification to the validation and test datasets. This procedure prevents distributional information from the validation and test datasets from entering the estimation of the scaling parameters. It also enables translational and rotational motion components with different numerical scales to be jointly learned in a normalized feature space. The scaler parameters and the lists of files assigned to each subset are stored in `scaler.json`.

### 3.4 Sliding-Window Generation

The `make_window_meta()` function stores `(file index, starting row index)` pairs rather than actual copies of the data. Starting indices advance by 16 rows, such as 0, 16, 32, and so forth.

### 3.5 Dataset and DataLoader

`TimeWindowDataset` extracts only the requested window from the original array, normalizes it, and returns the resulting data as tensors. Because all windows are not copied and stored in advance, memory duplication associated with overlapping windows is reduced. However, the complete arrays of the original files and the concatenated training array used to calculate the normalization statistics remain in memory.

| Dataset | Shuffle | Final Incomplete Batch | Purpose |
|---|---|---|---|
| Training | Applied | Excluded (`drop_last=True`) | Weight update |
| Validation | Not applied | Retained | Selection of optimal configuration and checkpoint |
| Test | Not applied | Retained | Final performance evaluation |

All data loaders use `pin_memory=True`, and the default number of workers is 4.

## 4. Model Architecture

### 4.1 Overall Tensor Transformations

| Stage | Operation | Output Shape |
|---|---|---|
| Input | Normalized historical motion | \((B, 128, 6)\) |
| Axis transformation | Transpose temporal and channel axes | \((B, 6, 128)\) |
| Input projection | Causal convolution, \(6 \rightarrow C\) | \((B, C, 128)\) |
| Feature extraction | \(N\) ModernTCN blocks | \((B, C, 128)\) |
| Backbone output | Axis transformation | \((B, 128, C)\) |
| Preparation for temporal projection | Axis transformation | \((B, C, 128)\) |
| Temporal projection | `Linear(128, 128)`, GELU, Dropout | \((B, C, 128)\) |
| Preparation for output projection | Axis transformation | \((B, 128, C)\) |
| Output projection | `Linear(C, 6)` | \((B, 128, 6)\) |

Here, \(C\) denotes the number of hidden channels, and \(N\) denotes the number of blocks.

### 4.2 Causal Convolution

For a kernel size \(k\), `CausalConv1d` adds \(k-1\) zeros to the left side of the input and applies no padding to the right side. The convolution itself uses no internal padding, while the stride and dilation retain their default values of 1. Consequently, the feature at output time step \(t\) depends only on the input at time step \(t\) and preceding time steps. The temporal length of the input and output remains unchanged.

However, the earliest time steps of the input sequence are influenced by zero padding rather than actual historical observations. Causality is imposed during the convolutional feature-extraction process. The final prediction head jointly uses the entire observed input interval. Because only historical information already available at the start of forecasting is used, this operation does not constitute leakage of future ground-truth information.

### 4.3 Input Projection Layer

The input projection transforms the six motion components into \(C\) hidden channels. Because a standard convolution is used, this layer extracts temporal information while simultaneously integrating information across the input degrees of freedom. Consequently, the channels processed by subsequent depthwise convolutions are learned hidden representations rather than original channels that correspond one-to-one with the individual degrees of freedom.

### 4.4 ModernTCN Block

Each block performs the following operations.

First, the depthwise causal convolution is configured with `groups=channels`, such that the temporal pattern of each hidden channel is processed independently. No inter-channel mixing occurs during this operation.

Second, `ChannelLayerNorm` transposes the array into the shape \((B,T,C)\) and performs normalization along the channel dimension at each time step of each individual sample. Because the statistics are not calculated jointly across the entire temporal dimension or across different samples within the batch, this operation does not introduce information from subsequent time steps into the current time step.

Third, the pointwise feed-forward network uses a convolution with a kernel size of 1 to increase the number of channels from \(C\) to

\[
H = C \cdot \mathrm{expansion}.
\]

GELU and Dropout are then applied, after which the number of channels is reduced from \(H\) back to \(C\), followed by another Dropout operation. This component mixes information across channels at the same time step.

Finally, a residual connection is employed by adding the block input to the transformed output.

### 4.5 Temporal Projection and Direct Multi-Step Forecasting

The backbone output is passed to a linear layer operating along the temporal dimension. The same temporal transformation matrix is applied independently to each hidden channel, mapping the 128 input time steps to 128 output time steps. GELU and Dropout are subsequently applied, after which the channel representation at each output time step is transformed into the six degrees of freedom. Thus, all 128 future time steps are predicted simultaneously through a single forward pass.

## 5. Training Objective and Optimization

### 5.1 Training Procedure

The loss function is the mean squared error (MSE) between the normalized predictions and the corresponding ground-truth values.

The `train_epoch()` function sequentially performs the following operations: setting the model to training mode, transferring the inputs and targets to the computing device, resetting the gradients, generating predictions, calculating the MSE, performing backpropagation, applying gradient clipping, and updating the model parameters.

The maximum norm for gradient clipping is 1.0. This operation does not independently truncate every gradient value to be less than or equal to 1. Instead, when the norm of the gradients across all model parameters exceeds the specified threshold, the gradients are proportionally rescaled.

AdamW is used as the optimization algorithm, while the learning rate and weight decay are selected by Optuna. The learning-rate scheduler is `ReduceLROnPlateau(mode="min", factor=0.5, patience=3)`. Accordingly, the learning rate is reduced by half when improvement in the validation loss stagnates for a specified period.

### 5.2 Validation and Early Stopping

The `evaluate_epoch()` function uses `model.eval()` and `torch.no_grad()` to calculate MSE and MAE without updating the model parameters. An improvement is identified when the validation loss is lower than the previous minimum value. Training is terminated early when no improvement is observed for 10 consecutive epochs.

The same criterion is applied during both the hyperparameter-search stage and the final training stage, although the maximum numbers of epochs are 40 and 300, respectively. The training loss is calculated while Dropout is active and the model weights continue to change. In contrast, the validation loss is calculated after training for the corresponding epoch using fixed model weights with Dropout disabled.

## 6. Optuna Hyperparameter Search

### 6.1 Search Space

| Parameter | Search Range | Description |
|---|---|---|
| `channels` | 64, 96, 128, 192, 256 | Hidden representation dimension |
| `num_blocks` | Integers from 2 to 8 | Network depth |
| `kernel_size` | 7, 15, 23, 31, 39, 47, 55, 63 | Temporal kernel size of each block |
| `expansion` | 1.0, 2.0, 3.0, 4.0 | Channel expansion ratio |
| `dropout` | 0.00–0.30, step 0.05 | Block Dropout |
| `stem_kernel_size` | 3, 5, 7, 11, 15, 21 | Input projection kernel size |
| `head_dropout` | 0.00–0.30, step 0.05 | Temporal projection head Dropout |
| `learning_rate` | 0.0001–0.005, logarithmic search | Learning rate |
| `weight_decay` | 0.00001–0.01, logarithmic search | AdamW regularization strength |
| `batch_size` | 32, 64, 128 | Training batch size |

### 6.2 Procedure for Each Trial

For each trial, the random seed is reset to 42, the hyperparameters are selected, and a new model and corresponding data loaders are created. The model is trained for up to 40 epochs, and the validation MSE is recorded at every epoch.

The objective value of each trial is not the loss from the final epoch; rather, it is the minimum validation MSE observed over all epochs completed during that trial. The number of model parameters, the number of completed epochs, and the lowest validation MSE achieved up to that point are stored as additional trial information.

### 6.3 Sampler and Pruner

The sampler is `TPESampler`, with the number of startup trials set to 8. After the initial exploration, the search is configured to proceed using the TPE approach, which utilizes the accumulated evaluation results.

Pruning is performed using `MedianPruner`. The number of startup trials is set to 8, and the number of warm-up steps is also set to 8. In the code, the epoch number is reported as the step. Once the initial performance-recording and warm-up requirements are satisfied, trials considered unlikely to produce competitive results can be terminated early based on their intermediate performance.

Early stopping is based on stagnation of the loss within the same trial, whereas pruning uses intermediate evaluation information from the overall study. In the code, the pruning decision is performed before the within-trial early-stopping decision.

### 6.4 Search Budget and Resumption

The total number of trials is set to 40. If an existing SQLite study is found, the study is resumed. The code subtracts the number of trials whose states are not `RUNNING` from 40 and executes the corresponding number of additional trials. This count includes not only `COMPLETE` trials but also trials in states such as `PRUNED` and `FAIL`.

Resumption using the SQLite database preserves the existing search results and allows additional trials to be conducted. However, the model weights or optimizer states from intermediate epochs are not stored for the purpose of resuming the training process of an individual trial.

## 7. Final Retraining Using the Optimal Configuration

After completion of the Optuna search, if at least one trial is in the `COMPLETE` state, `study.best_params`, corresponding to the hyperparameters of the completed trial with the lowest validation MSE, is selected.

The default random seed of 42 is then set again, and a new model is created using the selected hyperparameters. The newly created model is trained from scratch for up to 300 epochs. Therefore, the final retraining procedure trains the model with the optimal architecture from the beginning. The existing training set is used to update the model weights, while the existing validation set is used for early stopping and selection of the optimal model weights.

Whenever the validation MSE improves, the model `state_dict` is stored in `models/best_model.pt`. Finally, the training MSE, validation MSE, validation MAE, and learning-rate history are stored in `history.json`.

## 8. Test Evaluation and Interpretation of Metrics

After completion of the final retraining process, the weights of the model corresponding to the lowest validation MSE are loaded, and the model is evaluated on the test dataset.

MSE and MAE are calculated over all six degrees of freedom and the entire prediction horizon using the normalized data. Consequently, these values do not have physical units.

Subsequently, the prediction errors are multiplied by the degree-of-freedom-specific standard deviations calculated from the training data to restore them to their original physical scales, and the RMSE is calculated separately for each degree of freedom. The RMSE values for Surge, Sway, and Heave are expressed in meters (m), whereas those for Roll, Pitch, and Yaw are expressed in degrees (deg).

## 9. Conclusion

### 9.1 Optimal Hyperparameters

| Parameter | Optimal Value |
|---|---:|
| Trial number | 38 |
| Validation MSE | 0.428539 |
| Batch size | 64 |
| Channels | 128 |
| Number of blocks | 4 |
| Kernel size | 55 |
| Expansion ratio | 2.0 |
| Dropout | 0.0 |
| Head dropout | 0.1 |
| Stem kernel size | 3 |
| Learning rate | \(4.61 \times 10^{-4}\) |
| Weight decay | \(2.61 \times 10^{-5}\) |

### 9.2 Comparison of the Top-Performing Trials

| Rank | Trial | Validation MSE | Channels | Blocks | Kernel | Expansion | Dropout |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 38 | 0.428539 | 128 | 4 | 55 | 2.0 | 0.00 |
| 2 | 21 | 0.430326 | 128 | 4 | 39 | 2.0 | 0.00 |
| 3 | 31 | 0.435885 | 128 | 3 | 47 | 3.0 | 0.00 |
| 4 | 32 | 0.437116 | 128 | 3 | 47 | 3.0 | 0.00 |
| 5 | 28 | 0.437514 | 128 | 3 | 47 | 3.0 | 0.00 |
| 6 | 33 | 0.439990 | 192 | 3 | 47 | 3.0 | 0.00 |
| 7 | 10 | 0.441250 | 128 | 8 | 63 | 3.0 | 0.30 |
| 8 | 11 | 0.442056 | 128 | 8 | 63 | 3.0 | 0.30 |
| 9 | 5 | 0.442903 | 128 | 4 | 39 | 2.0 | 0.05 |
| 10 | 12 | 0.443298 | 128 | 8 | 63 | 3.0 | 0.30 |

All of the top five trials used 128 hidden channels. Completed trials using 64 channels exhibited validation MSE values of 0.49 or higher, while models using 256 channels also recorded validation MSE values of 0.459 or higher. Within the present hyperparameter-search results, 128 channels provided an appropriate balance between model representational capacity and training stability.

The top five trials used either three or four blocks, with kernel sizes of 39, 47, or 55. These results indicate that relatively large convolutional kernels are advantageous for extracting long-term temporal patterns from time-series data sampled at 2 Hz. The optimal Trial 38 employed a kernel size of 55 and four blocks. In addition, the backbone dropout was zero for all of the top six trials. The results indicate that models with lower dropout ratios had a higher probability of achieving superior performance.

The model corresponding to the optimal Trial 38 contained 313,094 trainable parameters. In contrast, Trial 14 contained approximately 3.31 million parameters, but its validation MSE was 0.459013, which was higher than that of the optimal model. Although Trial 14 used approximately 10.6 times more parameters than the optimal model, its validation performance did not improve. Trial 21, which achieved performance similar to that of Trial 38, contained 314,118 parameters. Furthermore, Trials 28, 31, and 32, each containing approximately 330,000 parameters, were also among the top-performing trials. Accordingly, for the present dataset, selecting an appropriate temporal kernel size and number of blocks with a model size of approximately 310,000–340,000 parameters was more effective than simply increasing the overall model size.

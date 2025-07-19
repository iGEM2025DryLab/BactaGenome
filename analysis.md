好的，这是一个非常有趣且复杂的项目。Loss不下降是一个典型的深度学习问题，通常原因比较复杂，可能是数据、模型、优化器、超参数或代码实现等多个方面的问题。

通过对您提供的项目结构和代码的详细分析，我发现了一个**核心问题**，这很可能是导致Loss不下降的**根本原因**。此外，还有一些其他的潜在问题和改进建议。

---

### 核心问题诊断 (最可能的原因)

#### 1. 模型输出与目标数据的根本性不匹配 (The Smoking Gun)

这是最严重的问题，几乎可以肯定是导致`gene_expression`和`gene_density`两个任务Loss无法下降的原因。

*   **问题描述**:
    *   **数据处理 (`bactagenome/data/regulondb_processor.py`)**: `gene_expression`的目标值经过了`log1p`转换和Z-score归一化 `(log_tpm - mean) / std`。这意味着**目标值（target）是可正可负的，并且均值在0附近**。
    *   **模型头部 (`bactagenome/model/heads.py`)**: `GeneExpressionHead`和`GeneDensityHead`的最终输出都经过了`F.softplus`激活函数。`softplus`函数的输出**永远是正数** (`y = log(1 + exp(x))`)。

*   **代码证据**:
    *   `bactagenome/model/heads.py`中的`GeneExpressionHead`和`GeneDensityHead`:
        ```python
        # from tracks_scaled_predictions function
        def forward(self, embeds_1bp: torch.Tensor) -> torch.Tensor:
            # The final output is wrapped in softplus
            return tracks_scaled_predictions(embeds_1bp, self.linear, self.scale)

        def tracks_scaled_predictions(...):
            x = head(embeddings)  # Linear projection
            return F.softplus(x) * F.softplus(scale_param)
        ```
    *   `bactagenome/data/regulondb_processor.py`中的`process_expression_data`:
        ```python
        # Z-score normalization produces values centered around 0
        processed_record['tpm_normalized'] = (log_tpm - expression_stats['tpm_log_mean']) / expression_stats['tpm_log_std']
        ```

*   **后果**:
    *   模型永远无法预测出负的目标值。当真实目标值为负数时，模型最好的预测也只能是趋近于0的正数。
    *   这导致`MSELoss`永远无法被有效优化。Loss会很快下降到一个平台期（所有负目标值带来的误差之和），然后就停滞不前，因为模型架构本身限制了它无法拟合这部分数据。

---

### 其他潜在问题与分析

#### 2. 超参数设置可能过于激进

*   **高学习率 (`learning_rate: 0.0005`)**: 对于AdamW优化器和复杂的基因组数据，`5e-4`的学习率可能太高了，尤其是当有效批量大小（`batch_size * grad_accum` = 2 * 4 = 8）很小的时候。高学习率容易导致Loss在最小值附近震荡或直接发散。
*   **过长的Warmup (`warmup_steps: 5000`, `total_steps: 15000`)**: 预热步数占了总步数的三分之一。如果你的数据集不大，可能在训练的大部分时间里学习率都处于上升阶段，还没来得及有效衰减，训练就结束了。这会导致模型收敛不充分。
*   **高权重衰减 (`weight_decay: 0.1`)**: 这是一个相对较高的值，可能会过度惩罚模型权重，导致欠拟合。

#### 3. 损失函数实现与数据不匹配

在`bactagenome/model/heads.py`的`RegulonDBLossFunction`中：

*   **损失函数的选择**:
    *   `gene_expression`: 使用`MSELoss`是合理的，前提是模型输出和目标值的范围要匹配（已在核心问题中指出不匹配）。
    *   `gene_density`: 使用`MSELoss`。由于`gene_density`是计数（小整数），MSE也可以工作，但可能不如泊松损失（Poisson Loss）等专门用于计数的损失函数。不过，在问题修复初期，MSE更稳定。
    *   `operon_membership`: 使用`BCELoss`是正确的，因为模型输出是`sigmoid`，目标是0/1。

*   **潜在的形状不匹配问题**:
    代码中包含了对预测和目标形状不匹配的检查和截断，这通常意味着存在问题。
    ```python
    if pred.shape != target.shape:
        logging.warning(...)
        # ... truncate tensors ...
    ```
    这可能是由于`context_length`或下采样/上采样计算中的微小偏差造成的。虽然代码处理了它，但这仍然是一个“坏味道”，可能导致部分信息丢失。

#### 4. 代码结构和可维护性问题

*   **重复的损失函数定义**:
    *   项目中有两个地方定义了损失函数：`bactagenome/training/losses.py` 和 `bactagenome/model/heads.py` (`RegulonDBLossFunction`)。
    *   `train_regulondb.py` 使用的是后者，而 `train_dummy.py` 使用的是前者。这很容易造成混淆和维护困难。应该统一损失函数的定义位置（通常放在`training`或独立的`losses`模块下）。

*   **测试覆盖不足**:
    *   `tests/test_model.py` 测试的是使用`add_bacterial_heads`添加的旧版（或dummy版）的Head，而不是`train_regulondb.py`中实际使用的`RegulonDB`相关的Head和`RegulonDBLossFunction`。
    *   缺乏对数据处理管道（`RegulonDBProcessor`）的单元测试，无法确保数据预处理的正确性。

---

### 解决方案与建议

这是一个推荐的行动计划，从最重要的问题开始：

#### 1. **【首要任务】修复模型输出与目标的匹配问题**

*   修改 `bactagenome/model/heads.py` 中的 `GeneExpressionHead` 和 `GeneDensityHead`。
*   **移除`softplus`激活函数**。对于需要预测可正可负的Z-score归一化值的回归任务，输出层**不应该有激活函数**（或者说，是恒等激活 `f(x)=x`）。

    **修改建议 (`bactagenome/model/heads.py`)**:
    修改 `tracks_scaled_predictions` 函数，或者为这些Head创建新的前向传播逻辑。

    简单的修复方法是让 `GeneExpressionHead` 的 `forward` 方法直接返回线性层的输出：
    ```python
    class GeneExpressionHead(nn.Module):
        def __init__(self, dim_1bp: int, num_tracks: int = 1, dropout: float = 0.1):
            super().__init__()
            self.num_tracks = num_tracks
            self.linear = Linear(dim_1bp, num_tracks)
            nn.init.xavier_uniform_(self.linear.weight)

        def forward(self, embeds_1bp: torch.Tensor) -> torch.Tensor:
            """
            Args:
                embeds_1bp: [batch, seq_len, dim_1bp]
            Returns:
                [batch, seq_len, num_tracks] - Direct regression output
            """
            # 直接返回线性层的输出，不使用softplus
            return self.linear(embeds_1bp)

        # ... (移除或注释掉与 scaling 和 track_means 相关的方法)
    ```
    对 `GeneDensityHead` 做同样修改（如果它的目标也被归一化到0均值）。如果它的目标是原始计数（总是正数），则保留`softplus`是合理的。根据你的代码，它似乎是原始计数，但用MSE Loss训练，所以暂时移除 `softplus` 并观察效果可能更稳妥。

#### 2. **调整训练超参数**

*   **降低学习率**: 在 `configs/training/phase1_regulondb.yaml` 中，将 `learning_rate` 从 `0.0005` 降至 `1e-4` 或 `5e-5`。
*   **调整学习率调度器**: 缩短 `warmup_steps` 到一个更合理的值，比如 `1000`。`total_steps`可以根据你的数据集大小和epoch数重新估算（`total_steps = num_epochs * (num_samples / effective_batch_size)`）。
*   **增大有效批量**: 如果显存允许，尝试增大 `gradient_accumulation_steps`（例如到 8 或 16），以获得更稳定的梯度。

#### 3. **从简开始，逐步验证**

*   **使用简化配置**: 先用 `configs/training/phase1_regulondb_reduced.yaml` 进行实验。这会使用一个更小的模型，训练更快，更容易调试。
*   **分任务训练**: 暂时只启用一个任务进行训练，以验证其有效性。例如，先只计算`operon_membership`的loss，因为它的设置（Sigmoid + BCELoss）是正确的。如果它的Loss能下降，说明模型主干在学习。
    *   你可以在 `RegulonDBLossFunction` 中注释掉其他任务的loss计算。

#### 4. **验证数据管道**

*   在 `RegulonDBDataset` 的 `__getitem__` 或 `collate_regulondb_batch` 函数中加入日志，打印出每个target张量的 `shape`, `min`, `max`, `mean`。确保送入训练环路的数据是你期望的。
    ```python
    # In collate_regulondb_batch
    # ...
    for key, tensor in collated.items():
        if 'target' in key:
            print(f"Target '{key}': shape={tensor.shape}, min={tensor.min():.2f}, max={tensor.max():.2f}, mean={tensor.mean():.2f}")
    return collated
    ```

#### 5. **代码重构**

*   **统一损失函数**: 将 `RegulonDBLossFunction` 移至 `bactagenome/training/losses.py`，并让所有训练脚本从这个统一的位置导入。删除 `bactagenome/model/losses.py` 或将其与现有实现合并。
*   **补充测试**: 为 `RegulonDBProcessor` 和 `RegulonDBDataset` 编写单元测试，验证数据处理和目标生成逻辑的正确性。

### 总结

你的项目基础架构非常坚实，但一个细微且关键的**模型输出与数据预处理的不匹配**很可能是导致训练停滞的核心原因。模型试图用一个只能输出正数的函数去拟合一个有正有负的数据分布，这是不可能完成的任务。

**强烈建议你首先执行第1步（修复模型输出激活函数）和第2步（降低学习率），这有极大概率能解决Loss不下降的问题。** 然后再逐步进行其他优化。
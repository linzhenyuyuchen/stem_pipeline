# 可直接生成训练 config 的上采样策略

## 最终决策

使用“Markdown 建议权重 + 原始数据集 token + 目标总 token”计算每个 config 数据集的实际倍数，再把倍数拆成若干完整副本和最多一个小数副本。历史 config 不做下采样；输出 config 中不保留原名称，所有副本统一命名为 `原数据集名_1`、`原数据集名_2`……。

配套脚本：[apply_upsampling_to_config.py](./apply_upsampling_to_config.py)。脚本会同时生成修改后的 JSON config 和一份逐数据集审计 Markdown。

## 三个输入

### 1. 采样前 config

示例：

```text
/home/ma-user/work/others/w00911759/stem_caption_v3_szc_3.json
```

要求：

- JSON 顶层是 `{数据集名称: 数据集配置}`。
- 每个数据集必须包含正整数 `sample_num`。
- `sample_num_all`、路径和其他字段原样复制；只有小数副本的 `sample_num` 会修改。

### 2. 数据集建议权重 Markdown

示例：

```text
/home/ma-user/work/others/w00911759/evaluation_analysis/comparison_pangu_vs_qwen/new_data_upsampling_recommendations.md
```

要求表格至少包含：

```text
排名 | 数据集 | 上采样权重
```

`排名`作为唯一稳定 ID。不能只依赖数据集名称：当前示例 Markdown 有 175 行，但只有 144 个唯一名称，31 个名称重复出现。

### 3. token 预算与映射 JSON

推荐格式：

```json
{
  "target_tokens": 3000000000,
  "datasets": [
    {
      "config_key": "dataset_a",
      "total_tokens": 500000000,
      "weight_rank": 12,
      "weight_dataset": "geometry3k_0308"
    },
    {
      "config_key": "dataset_b",
      "total_tokens": 800000000,
      "weight_rank": 18,
      "weight_dataset": "doubao_physics_zh_0328"
    }
  ]
}
```

字段定义：

| 字段 | 必填 | 含义 |
|---|---|---|
| `target_tokens` | 是 | 输出 config 希望达到的总 token |
| `config_key` | 是 | 输入 config 顶层的准确数据集键 |
| `total_tokens` | 是 | 该 config 数据集在原始 `sample_num` 下的总 token |
| `weight_rank` | 条件必填 | 权重 Markdown 中的排名；名称不一致或重名时必须填写 |
| `weight_dataset` | 否 | 对 `weight_rank` 对应数据集名称做交叉校验，建议填写 |

如果 config 名称与 Markdown 数据集名称完全一致且该名称唯一，可以省略 `weight_rank`。生产使用仍建议显式填写 `weight_rank`，避免 Markdown 更新或重名造成错配。

也支持 CSV，字段为 `config_key,total_tokens,weight_rank,weight_dataset`，但目标总 token 需通过命令行 `--target-tokens` 提供。

## token 预算如何转换为实际倍数

设：

- 数据集原始 token 为 `tᵢ`。
- Markdown 建议权重为 `wᵢ`，且 `wᵢ ≥ 1`。
- 原始总 token 为 `B = Σtᵢ`。
- 完整采用建议权重后的总 token 为 `F = Σ(tᵢ × wᵢ)`。
- 目标总 token 为 `T`。

默认要求：

```text
B ≤ T ≤ F
```

权重强度：

```text
α = (T - B) / (F - B)
```

每个数据集实际倍数：

```text
mᵢ = 1 + α × (wᵢ - 1)
```

这个公式具备三个性质：

- `T = B` 时，所有数据集都是 `1×`。
- `T = F` 时，实际倍数等于 Markdown 建议权重。
- 目标位于两者之间时，只缩放“超过 1× 的增量”，不产生下采样，并保持建议权重的优先级方向。

如果 `T > F`，脚本默认报错，避免无意中超过分析建议；确认需要外推时添加 `--allow-extrapolation`。如果 `T < B`，由于任务限定为只上采样，脚本始终报错。

## config 复制规则

对每个实际倍数 `m = N + f`：

- `N = floor(m)`，生成 `N` 个完整副本。
- `f = m - N`。
- 若 `f = 0`，不生成额外副本。
- 若 `f > 0`，再生成一个小数副本，其 `sample_num = round_half_up(原 sample_num × f)`。
- 小数副本只修改 `sample_num`，包括 `sample_num_all` 在内的其他字段保持原值。

示例：原数据集 `dataset_a` 的 `sample_num=1000`。

实际倍数为 `3×`：

```json
{
  "dataset_a_1": {"sample_num": 1000},
  "dataset_a_2": {"sample_num": 1000},
  "dataset_a_3": {"sample_num": 1000}
}
```

实际倍数为 `3.27×`：

```json
{
  "dataset_a_1": {"sample_num": 1000},
  "dataset_a_2": {"sample_num": 1000},
  "dataset_a_3": {"sample_num": 1000},
  "dataset_a_4": {"sample_num": 270}
}
```

## 直接执行

```bash
python /home/ma-user/work/others/w00911759/evaluation_analysis/comparison_pangu_vs_qwen/apply_upsampling_to_config.py \
  --config /home/ma-user/work/others/w00911759/stem_caption_v3_szc_3.json \
  --weights-md /home/ma-user/work/others/w00911759/evaluation_analysis/comparison_pangu_vs_qwen/new_data_upsampling_recommendations.md \
  --tokens /绝对路径/token_budget.json \
  --output /绝对路径/stem_caption_v3_szc_3_upsampled.json
```

输出：

- `stem_caption_v3_szc_3_upsampled.json`：可直接交给训练流程的上采样 config。
- `stem_caption_v3_szc_3_upsampled_audit.md`：逐数据集记录建议权重、实际倍数、完整副本数、小数副本 `sample_num`、预计 token 和目标误差。

使用 CSV token 文件时：

```bash
python /home/ma-user/work/others/w00911759/evaluation_analysis/comparison_pangu_vs_qwen/apply_upsampling_to_config.py \
  --config /绝对路径/input.json \
  --weights-md /绝对路径/new_data_upsampling_recommendations.md \
  --tokens /绝对路径/token_budget.csv \
  --target-tokens 3000000000 \
  --output /绝对路径/output_upsampled.json
```

## 强制校验

脚本在以下情况停止，不生成可能错误的 config：

- config 数据集缺少 token 信息。
- `sample_num` 或 `total_tokens` 不是正数。
- Markdown 没有可识别的权重表。
- 名称无法匹配，或名称重名但没有提供 `weight_rank`。
- `weight_rank` 与可选的 `weight_dataset` 不一致。
- 目标 token 小于原始 token。
- 目标 token 超过建议权重对应的 token，且未显式允许外推。
- 输出副本名称发生冲突。
- 单数据集副本数超过默认安全上限 100；可通过 `--max-copies` 显式调整。

## 当前示例不能直接生成最终 config 的原因

示例 config 包含 3 个聚合键：

```text
STEM_data_purchase_caption_caption_v3d1_360k
STEM_QA_27M_weakness_caption_v3d1_100k
STEM_Caption_21M_weakness_caption_v3d1_180k
```

权重 Markdown 则是 175 行细粒度数据集；上述 3 个键与 Markdown 中的数据集名称均无直接匹配。同时，尚未提供这 3 个 config 数据集各自的 token 和目标总 token。因此当前不能可靠地产生一个真实的最终 JSON。

实际执行前，只需补齐 token 预算 JSON，并为这 3 个聚合键指定对应的 `weight_rank`。如果一个聚合 config 实际包含多个 Markdown 数据集，则不应强行指定单一排名；应先把 config 拆成同一粒度，或另行提供聚合权重。聚合权重应按 token 加权计算：

```text
聚合权重 = Σ(子数据集 token × 子数据集权重) / Σ(子数据集 token)
```

为避免绕过 Markdown 映射，当前脚本不自动猜测聚合关系。

## 上线前检查

1. 检查审计文件中每个 `config_key` 的 `weight_rank` 与数据集是否一致。
2. 检查输出条目数是否等于所有完整副本与小数副本之和。
3. 检查小数副本只改变了 `sample_num`。
4. 检查预计 token 与目标 token 的误差；误差来自整数 `sample_num` 舍入，通常远小于一个样本的平均 token。
5. 用训练加载器做一次 dry-run，确认所有新名称、路径和 `sample_num` 均可读取。

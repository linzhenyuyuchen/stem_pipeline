# 0812 数据 Caption 任务

本目录整理自 `run_0812_data.sh` 和 `run_0812_data_qwen38.sh`，包含两套流程：

- `run_0812_data.sh`：调用 OpenAI 兼容的 Gemini/VLM API，单机后台运行。
- `run_0812_data_qwen38.sh`：调用每台机器本地的 Qwen3.8-27B OpenAI 兼容服务，支持多机自动分片、断点续跑和节点状态汇总。

原 Gemini 脚本同时传入了两次 `--start-annotation`，Python 实际采用后一个值 `3800`。整理版将默认值明确为 `START_ANNOTATION=3800`；Qwen 默认仍从 `0` 开始。

## 目录结构

```text
captioner_final/
├── README.md
├── requirements.txt
├── run_0812_data.sh                 # Gemini/VLM 启动脚本
├── run_0812_data_qwen38.sh          # Qwen 多机 Caption 启动脚本
├── monitor_caption_nodes.sh         # Qwen 多机心跳汇总
├── captioner/                       # Caption 主程序及依赖代码
├── domain_and_genre_labels/         # domain/genre 标签映射
├── config/vlm_api.example.yaml      # 脱敏配置模板
└── qwen_server/start.sh             # Qwen vLLM-Ascend 服务
```

## 1. 运行前准备

### Python 依赖

Caption 客户端至少需要 Python 3.11、OpenAI SDK 和 PyYAML：

```bash
cd /home/ma-user/work/others/w00911759/codes/captioner_final
python3.11 -m pip install -r requirements.txt
```

启动脚本使用 `python3.11 -S`，并通过 `PYTHONPATH_VALUE` 指定依赖目录。当前环境默认值为：

```text
/usr/local/python3.11.15/lib/python3.11/site-packages
```

如果依赖安装在其他位置，两套 Caption 启动脚本都可设置 `PYTHONPATH_VALUE`。例如：

```bash
PYTHONPATH_VALUE=/path/to/site-packages bash run_0812_data.sh
```

Qwen 服务另外要求可用的 Ascend/CANN、`torch_npu`、`transformers`、`vllm`/vLLM-Ascend 环境。
如果 `vllm` 不在 `PATH` 中，通过 `VLLM_BIN=/path/to/vllm` 指定入口。

### 数据格式和路径

默认运行目录为：

```text
/home/ma-user/work/others/w00911759/dataset_0827/purchase_panguml_with_meta_info/purchase_caption/0812_data
```

默认输入子目录：

- Gemini：`annotation_split_dedup_gemini`
- Qwen：`annotation_split_dedup_qwen`

输入必须是 `.jsonl` 文件；批量模式通过文件名末尾的数字识别 annotation 序号，例如 `part_03800.jsonl`。每条记录需要包含：

- `data` 内的图片 `relative_path`
- `meta_info.label.domain_label`
- `meta_info.label.genre_label`
- 可选的 `meta_info.stem_ext` 课程、学科和知识点先验

脚本固定读取 `part_000.tar` 和同目录的 `part_000_tarinfo.json`。默认图像根目录是：

```text
/mnt-mm-pro/s3-g-szc-asset-aifm-mm-exp/j00939224/data/0812_data/images
```

可用 `RUN_DIR`、`INPUT_DIR`、`IMAGE_ROOT` 覆盖以上路径。启动脚本会先检查输入目录和图像根目录是否存在。

## 2. Gemini/VLM 流程

先创建本地配置。模板不含真实密钥，`config/vlm_api.yaml` 已加入 `.gitignore`：

```bash
cd /home/ma-user/work/others/w00911759/codes/captioner_final
cp config/vlm_api.example.yaml config/vlm_api.yaml
# 编辑 config/vlm_api.yaml，填写 VLM_API_BASE、VLM_API_KEY、VLM_MODEL
```

批量启动，默认从 annotation 3800 开始：

```bash
bash run_0812_data.sh
```

从其他序号开始：

```bash
START_ANNOTATION=0 bash run_0812_data.sh
```

只处理一个文件：

```bash
bash run_0812_data.sh --input-jsonl part_03800.jsonl
```

常用覆盖示例：

```bash
RUN_DIR=/data/0812_data \
IMAGE_ROOT=/data/0812_data/images \
DATA_WORKERS=48 API_WORKERS=100 \
TIMEOUT=120 MAX_RETRIES=2 \
bash run_0812_data.sh
```

默认输出和日志：

- 结果：`$RUN_DIR/annotation_captioned_gemini/*.jsonl`
- 启动日志：`$RUN_DIR/run_0812_data.launcher.log`
- 分片日志：`$RUN_DIR/logs_gemini/*.log`

## 3. Qwen3.8 多机流程

### 3.1 每台机器启动本地模型服务

每台参与 Caption 的机器都需要在 `127.0.0.1:8000` 启动服务：

```bash
cd /home/ma-user/work/others/w00911759/codes/captioner_final
mkdir -p logs
nohup setsid bash qwen_server/start.sh > logs/qwen_server.log 2>&1 < /dev/null &
echo $!
```

默认模型路径为 `/home/ma-user/work/others/w00911759/models/qwen-3.8-27B`，默认按 16 张 NPU、`TP_SIZE=2`、`DP_SIZE=8` 启动。资源不同可覆盖：

```bash
MODEL_PATH=/path/to/qwen-3.8-27B \
NPU_VISIBLE_DEVICES=0,1,2,3 \
TP_SIZE=2 DP_SIZE=2 \
nohup setsid bash qwen_server/start.sh > logs/qwen_server.log 2>&1 < /dev/null &
```

确认接口就绪：

```bash
curl -fsS http://127.0.0.1:8000/v1/models
```

如接口或模型名不同，可在 Caption 启动时设置：

```bash
export QWEN_38_API_BASE=http://127.0.0.1:8000/v1
export QWEN_38_API_KEY=EMPTY
export QWEN_38_MODEL=/path/to/qwen-3.8-27B
```

### 3.2 启动 Caption

在所有节点上执行相同命令。ModelArts 环境会依次尝试从 `MA_VJ_NAME`、`NODE_RANK`/`GROUP_RANK` 和 rank table 自动识别节点编号与节点数：

```bash
cd /home/ma-user/work/others/w00911759/codes/captioner_final
bash run_0812_data_qwen38.sh
```

非 ModelArts 环境可显式指定零基节点编号。下面是两机示例：

```bash
# 节点 0
MACHINE_INDEX=0 MACHINE_COUNT=2 bash run_0812_data_qwen38.sh

# 节点 1
MACHINE_INDEX=1 MACHINE_COUNT=2 bash run_0812_data_qwen38.sh
```

只指定一个 JSONL 时，仍按 annotation 序号计算归属；只有负责该分片的节点会处理它：

```bash
MACHINE_INDEX=0 MACHINE_COUNT=2 \
bash run_0812_data_qwen38.sh --input-jsonl part_00000.jsonl
```

默认输出和日志：

- 结果：`$RUN_DIR/annotation_captioned_qwen38/*.jsonl`
- 节点启动日志：`$RUN_DIR/logs_qwen38/run_0812_data_qwen38.machine_NNN.launcher.log`
- 分片日志：`$RUN_DIR/logs_qwen38/machine_NNN/*.log`
- 集群汇总：`$RUN_DIR/logs_qwen38/all_nodes_status.log`
- 节点心跳：`$RUN_DIR/logs_qwen38/node_status/`

查看状态：

```bash
tail -f "$RUN_DIR/logs_qwen38/all_nodes_status.log"
```

Qwen 常用环境变量：`RUN_DIR`、`INPUT_DIR`、`IMAGE_ROOT`、`OUTPUT_DIR`、`LOG_ROOT`、`START_ANNOTATION`、`MACHINE_INDEX`、`MACHINE_COUNT`、`DATA_WORKERS`、`API_WORKERS`、`TIMEOUT`、`MAX_RETRIES`、`MAX_TOKENS`、`PYTHON_BIN`、`PYTHONPATH_VALUE`、`QWEN_38_API_BASE`、`QWEN_38_API_KEY`、`QWEN_38_MODEL`。

## 4. 断点续跑与停止

两套启动脚本都默认传入 `--resume`：已成功记录会跳过，`API_ERROR:`/`ERROR:` 记录会重试。Qwen 输出目录不包含节点编号，因此调整机器数量后仍可复用已有结果。

脚本通过 `setsid` 启动独立进程组，并会打印后台 PID。停止任务时建议终止整个进程组，避免遗留 Python 子进程：

```bash
kill -- -<PID>
```

不要同时启动两个会写同一个输入分片和输出目录的任务。

## 5. 常见问题

- `input directory does not exist`：数据盘未挂载，或需要设置 `RUN_DIR`/`INPUT_DIR`。
- `image root does not exist`：设置正确的 `IMAGE_ROOT`；当前整理时可见的是 `/mnt-mm-pro/...`，不是 `/mm-pro/...`。
- `missing VLM config`：将示例复制为 `config/vlm_api.yaml` 并填写真实配置。
- `No module named openai/yaml`：安装 `requirements.txt`，并把安装目录传给 `PYTHONPATH_VALUE`。
- Qwen API 连接失败：先检查 `curl http://127.0.0.1:8000/v1/models` 和 `logs/qwen_server.log`。
- 多机无法识别编号：显式设置 `MACHINE_INDEX` 和 `MACHINE_COUNT`。

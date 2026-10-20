# qa_5pattern — 五种推理模式 QA 答案重写

用多模态 Chat API 批量重写 Pangu-ML QA 数据集中的答案：对每条样本，模型先从 **R1–R5 五种推理模式**中自行选择最合适的一种，再按该模式输出完整推理过程和最终答案；脚本只把原样本 `data[1]` 里的答案字符串替换掉，其余字段原样保留。

---

## 目录结构

```
qa_5pattern/
├── solve_qa_5pattern.py   # 主程序：读 config/data/tar → 调 API → 写新 annotation 与 config
├── run_qa_5pattern.sh     # 后台启动器：环境变量、日志、PID、防重复启动
├── sys_prompt.txt         # system prompt：R1–R5 定义、适用场景、选择原则、输出格式
├── user_prompt.txt        # user prompt 模板，含 {{question}} 占位符
├── logs/                  # 运行日志与 PID 文件
└── README.md
```

> 脚本会优先读取同目录下的 `meta_prompt.txt`，存在时它覆盖 `sys_prompt.txt`（见 `solve_qa_5pattern.py:38`）。当前目录没有这个文件。

---

## 它做什么

输入是一个 **Pangu-ML data config**（JSON）。对其中每个条目：

1. 读取 `annotation_dir` 下所有 `*.jsonl`；
2. 按照相同的相对路径，在 `tar_image_root` 下找到对应的 `part_xxx.tar` 和 `part_xxx_tarinfo.json`；
3. 逐行解析样本，组装成 `[system, user(image + text)]` 的多模态消息；
4. 调用 OpenAI 兼容的 `/chat/completions` 接口；
5. 把模型返回的文本写回 `data[1].content[0].text.string`，其余字段深拷贝保持不变；
6. 将所有成功记录按**原始行序**写入新的 annotation 目录；
7. 生成一份新的 config。

它**不会**动原数据：原始 `annotation_dir` 和 TAR 只读，输出全部写到兄弟目录。

---

## 输入格式

### data config

```json
{
  "mmmupro_qa": {
    "dataset_name": "mmmupro_qa",
    "annotation_dir": ".../mmmupro_qa/annotation",
    "tar_image_root": ".../mmmupro_qa/images",
    "sample_num": 385106,
    "sample_num_all": 385106,
    "...": "其余 panguml 字段原样透传"
  }
}
```

脚本只依赖 `annotation_dir`、`tar_image_root`，可选 `dataset_name`（仅用于日志，缺省用 config 的 key）。

### JSONL 样本

```json
{
  "uuid": "01a0c335-...",
  "data_uuid": "e9387d84...",
  "data": [
    {"role": "user", "content": [
      {"type": "image", "image": {"type": "relative_path", "format": "image/jpeg",
                                  "relative_path": "images/xxx.jpg", "width": 850, "height": 566}},
      {"type": "text", "text": {"type": "string", "format": "utf-8",
                                "string": "Which nutrient deficiency ...\nA. Phosphate\nB. ..."}}
    ]},
    {"role": "assistant", "content": [
      {"type": "text", "text": {"type": "string", "format": "utf-8", "string": "A. Phosphate"}}
    ]}
  ],
  "meta_info": {"data_version": "20260921", "data_source": {"source": "mmmupro_qa"}}
}
```

硬性要求：

- `data[0].content` 非空，元素只能是 `type: "image"` 或 `type: "text"`；
- image 必须有 `relative_path`，且能在对应 tar 的 tarinfo 索引里查到；
- `data[1].content[0].text.string` 必须是字符串（**在发请求前就会校验**，避免白白消耗一次 API 调用）；
- 问题文本非空。

### TAR

必须是**未压缩** tar，配一份 `*_tarinfo.json`，格式为 `{相对路径: {"offset_data": int, "size": int}}`。程序用 `os.pread` 按偏移随机读取，不落盘解压，也不顺序扫描 tar。一张图片跨线程共享同一个 fd，因此线程安全且无内存常驻。

---

## 五模式 Prompt

`sys_prompt.txt` 要求模型在下列模式中选一个，并输出「所选模式 → 推理过程 → 最终答案」：

| 模式 | 名称 | 适用 |
|---|---|---|
| R1 | Direct Reasoning | 已知知识/公式直接得解，单一路径 |
| R2 | Decomposition | 多子问题拆解后逐步求解再整合 |
| R3 | Option Elimination | 多候选项，靠约束/证据排除 |
| R4 | Solve → Verify | 先解后验，对正确性要求高 |
| R5 | Explore → Solve | 开放式问题，多种解法需先比较再选 |

选择原则写在 prompt 末尾：能用 R1 就不要上更复杂的模式。

`user_prompt.txt` 是模板，必须包含 `{{question}}`，脚本会把样本中的文本内容用 `\n\n` 拼接后替换进去。拼接前会删掉两句特定的指令：

- `Answer with the option letter from the given choices directly.`
- `Answer with the option's letter from the given choices directly.`

这样模型会给出完整的模式化推理，而不是只回一个选项字母。

---

## 输出

假设 config 中 `annotation_dir` 为 `.../mmmupro_qa/annotation`：

```
.../mmmupro_qa/
├── annotation/                     # 原始，只读
├── annotation_modified/            # 新增，与源目录同构
│   ├── part_000.jsonl              # 原始行序，仅保留成功记录
│   └── ...
└── .../qa_modified.json            # 新 config
```

- 新 JSONL 与源文件同名、同相对路径；**失败或未被 `--limit` 覆盖的记录直接省略**（不是占位空行）。
- 源目录中的非 `.jsonl` 文件与空目录会被复制过去，保证目录结构完整。
- 新 config：`annotation_dir` 改为 `_modified` 目录；`sample_num` / `sample_num_all` 改为该条目**实际成功条数**；`tar_image_root` 不做任何改写。
- 写 JSONL 用临时文件 + `os.replace` 原子替换，中途崩溃不会留下半个文件。

---

## 快速开始

```bash
# 1. 准备 key（三者其一）
export QA_API_KEY="sk-..."
# 或者写进文件：echo "sk-..." > /path/to/key.txt && export QA_API_KEY_FILE=/path/to/key.txt

# 2. 直接前台跑（单地址）
python3 solve_qa_5pattern.py \
    --config /path/to/qa.json \
    --base-url http://127.0.0.1:8891/v1 \
    --model gemini-3.8-flash \
    --workers 32

# 3. 或者用启动器后台跑（推荐，5 地址 × 128 并发 = 640 worker）
./run_qa_5pattern.sh
```

`run_qa_5pattern.sh` 的行为：

- 固定读取 `/home/ma-user/work/others/l00788561/data/web_retrival/qa.json`（需要改就改脚本里的 `CONFIG_PATH`）；
- 默认 5 个本地端口（8891–8895），`QA_WORKERS=128`，总并发 **640**；
- `QA_RETRIES=1`（启动器默认不重试，出错即失败）、`QA_TIMEOUT=600`；
- 自动 `realpath`，日志写 `logs/qa_5pattern_<config名>_<时间戳>.log`，PID 写 `logs/qa_5pattern_<config名>.pid`；
- 若 PID 对应的进程还活着则拒绝重复启动；
- `nohup setsid ... &` 后台运行，`disown` 脱离当前 shell；
- 额外命令行参数会原样透传给 Python（`"$@"`）。

停止：

```bash
kill "$(cat logs/qa_5pattern_qa.pid)"
```

---

## 命令行参数

| 参数 | 默认 | 说明 |
|---|---|---|
| `--config` | 必填 | Pangu-ML data config 路径 |
| `--output-config` | `<config>_modified.json` | 新 config 输出路径 |
| `--checkpoint-db` | 输出 config 旁 `.<stem>.qa_5pattern_resume.sqlite3` | 断点数据库路径 |
| `--meta-prompt` | `meta_prompt.txt` 或 `sys_prompt.txt` | system prompt 文件 |
| `--user-prompt` | `user_prompt.txt` | user prompt 模板，必须含 `{{question}}` |
| `--base-url` | `$QA_BASE_URL` 或 `http://127.0.0.1:8891/v1` | 可多个，空格/逗号/分号分隔 |
| `--model` | `$QA_MODEL` 或 `gemini-3.1-flash-lite` | 模型名 |
| `--workers` / `-M` | `32` | **每个 base URL** 的并发数 |
| `--timeout` | `240` | 单请求超时（秒） |
| `--retries` | `3` | 总尝试次数，`1` 表示不重试 |
| `--max-tokens` | `$QA_MAX_OUTPUT_TOKENS` 或 不限 | 输出 token 上限 |
| `--extra` | 空 | 追加到请求体的 JSON，如 `'{"enable_thinking":false}'` |
| `--insecure` | 关 | 跳过 TLS 校验 |
| `--limit` | `0` | 只处理全局前 N 条，`0` 不限 |
| `--resume` | 关 | 跳过已成功记录，只补未处理/失败的 |
| `--dry-run` | 关 | 只做数据校验、取图、拼消息，不调 API 不写输出；无需 key |

实际并发 = `base_url 数量 × --workers`。任务被**连续均分**成等长 chunk，第 i 个线程固定用 `base_urls[i % len]`，因此多地址场景下请求天然分散到各端口。

---

## 环境变量

| 变量 | 作用 |
|---|---|
| `QA_API_KEY` / `OPENAI_API_KEY` / `GPT55_API_KEY` | 按顺序取第一个非空的作为 key |
| `QA_API_KEY_FILE` | key 文件路径，默认 `/mnt-szc-mm-exp/others/j00939224/captioner_production/qa/qa_api_key.txt` |
| `QA_BASE_URL` | 默认 base URL（可多值） |
| `QA_MODEL` | 默认模型 |
| `QA_MAX_OUTPUT_TOKENS` | 默认 `max_tokens` |
| `QA_BYPASS_PROXY` | 默认 `true`：`requests` 不走环境代理；设为 `false/0/no` 才走 |
| `QA_LOG_DIR` / `QA_LOG_FILE` / `QA_PID_FILE` | 启动器的日志与 PID 路径 |
| `QA_WORKERS` / `QA_RETRIES` / `QA_TIMEOUT` | 启动器的并发/重试/超时 |

API Key 解析顺序：环境变量 → 文件。两者都没有时直接退出。

---

## 断点续跑（resume）

断点分**两层**，都参与去重：

1. **正式输出 JSONL**（`annotation_modified/part_xxx.jsonl`）——按 `data_uuid`（回退 `uuid`）做计数匹配；
2. **SQLite checkpoint**——每成功一条立即 `INSERT OR REPLACE`，`journal_mode=WAL` + `synchronous=FULL` + autocommit，保证日志里报 `ok` 之前数据已经落盘。

`--resume` 时两者取并集，正式输出优先（覆盖「写完输出但还没清理 checkpoint 就崩溃」的场景）。全部完成后 checkpoint 表和 `.sqlite3/-wal/-shm` 文件会被删除。

安全校验（不满足直接报错退出，不会静默出错）：

- `sources` 表记录每个源 JSONL 的 `(size, mtime_ns)`；源文件变了且已有记录时拒绝 resume，需手动删库；
- checkpoint 里 `(源文件, 行号)` 的 UUID 必须与原始行一致；
- 正式输出的 UUID 必须都能在原始数据里匹配上，且不能多；
- 行号顺序必须单调。

**不要并发跑同一个 checkpoint 库**：两个进程会在合并阶段互相踩。启动器的 PID 检查只能防同一脚本的重复启动。

---

## 失败策略

- 重试：指数退避 `min(2^(n-1) + random, 30s)`；HTTP `400/401/403/404` 立即放弃，其余（超时、5xx、网络错误）重试到 `--retries` 上限。
- 返回内容为空或 `choices[0].message.content` 非字符串视为失败。
- **单条失败不会中断整体**，只记 `FAIL` 日志并跳过；最终该行不出现在输出里，config 的 `sample_num` 也按实际成功数写。
- 进程退出码：全部成功 `0`；有失败 `1`；「没有数据可处理」且未开 `--resume` 时 `1`。

日志每行形如：

```
2026-09-22T09:20:05+08:00 [1/9787] ok   worker=9 file=part_000.jsonl:1575 uuid=69da... elapsed=6.26s
2026-09-22T09:20:05+08:00 [2/9787] FAIL worker=8 file=part_000.jsonl:1198 uuid=... elapsed=9.61s error=Timeout: ...
```

真实运行参考：385106 条、7 个分片、5 地址 × 128 并发，checkpoint 恢复 375319 条、新增成功 9786 条、失败 1 条。

---

## 使用注意

- **`--limit` 会丢数据**：它只把任务列表截断到前 N 条，合并时剩余的原始记录被当作「未处理」省略。想保留全量请跑完整轮次。
- **失败记录不会自动重试**：必须用 `--resume` 再跑一遍，脚本会只挑未成功的。
- **`run_qa_5pattern.sh` 里硬编码了 API Key**（`run_qa_5pattern.sh:7`），该文件不要提交到公共仓库；更稳妥的做法是删掉这一行、改用环境变量。
- 图片以 base64 内联进请求体，**大图会显著增大请求体积**，必要时先做分辨率压缩。
- tarinfo 里的 key 同时兼容 `path`、`./path` 与反斜杠路径，但大小写敏感。
- 输出文件按原始行序合并，因此**不能用多进程分片各写各的文件**再简单拼接——需要改代码时这点要留意。
- `--dry-run` 不写任何输出，也不创建 checkpoint，只打印扫描结果；适合在新数据上先验一遍 tar/tarinfo 是否匹配。

---

## 排错

| 现象 | 原因 / 处理 |
|---|---|
| `未找到 API Key` | 设置 `QA_API_KEY` 或 `QA_API_KEY_FILE` |
| `JSONL 对应的 TAR 不存在` | `tar_image_root` 与 `annotation_dir` 的相对路径不匹配 |
| `tarinfo 中不存在图片` | 图片在 tar 中缺失或 key 大小写/前缀不符，可先用 `--dry-run` 定位 |
| `记录缺少 data[1].content[0].text.string` | 样本结构不符合约定，需先清洗数据 |
| `源 JSONL 已变化，不能安全 resume` | 删掉对应 `.qa_5pattern_resume.sqlite3` 后重跑 |
| `resume 输出含原始 JSONL 中无法匹配的 UUID` | 输出目录与当前 config 不是同一批数据，换目录或清空重跑 |
| 所有请求 401/403 | key 或 base URL 不匹配；`QA_BYPASS_PROXY` 默认不走代理，内网需确认端口可达 |
| 日志显示 640 worker 但速度上不去 | 瓶颈在服务端或图片体积；先减 `--workers` 观察，再排查 tar 读放大 |

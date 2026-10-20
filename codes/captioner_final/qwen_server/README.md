# Qwen3.8-27B vLLM-Ascend 服务

`start.sh` 启动与 caption 客户端匹配的 OpenAI 兼容接口，默认监听
`0.0.0.0:8000`。详细运行顺序见上级目录的 `README.md`。

默认模型：

```text
/home/ma-user/work/others/w00911759/models/qwen-3.8-27B
```

默认使用 16 张可见 NPU，`TP_SIZE=2`、`DP_SIZE=8`。例如：

```bash
MODEL_PATH=/path/to/qwen-3.8-27B \
NPU_VISIBLE_DEVICES=0,1,2,3 \
TP_SIZE=2 DP_SIZE=2 \
bash start.sh
```

如果 `vllm` 不在 `PATH` 中，可额外设置 `VLLM_BIN=/path/to/vllm`。

可用以下命令检查服务：

```bash
curl -fsS http://127.0.0.1:8000/v1/models
```

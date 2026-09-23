# 本地 / 网关 LiteLLM 联调备忘（2026-09-18）

> 密钥 **不进 git**。真实 base/key 写在 `work_root/studio_settings.json` 或 Studio UI。

## 网关（本机已验证）

- Base URL：`http://49.235.157.202:9400`（与 Claude Code `ANTHROPIC_BASE_URL` 相同）
- Chat 可用模型组（实测）：
  - **`glm-5.2`** — OK（偶发 429）
  - **`auto`** — OK（fallback，实测落到 glm-5.2）
  - **`internlm-s2`** — OK（回复较长）
  - `deepseek-pro` / `deepseek` / `sensenova-lite` — **常见 429** TPM/RPM
  - `sensenova-u1-fast` 等 — **chat 404**（更像图/视频模型 id，非 chat group）
- 可见 id 还有：`kimi-k3`（chat 400）、`agnes-image-2.5-flash`、`nvidia-*` 等（未全部测 chat）

## Studio 接法

1. UI「工程」→「本地 LiteLLM / 供应商」
2. Base URL 填网关；Model 推荐 **`auto`** 或 **`glm-5.2`**
3. API Key 用网关 token
4. 润色节点：`provider=litellm`，model 留空则用 settings
5. 节点对 **429 会自动重试 2 次**

## Claude Code

- `ANTHROPIC_MODEL` 环境当前为 `glm-5.2`；settings.json 内仍写 `deepseek-pro`（易 429）
- 重派后台任务时建议显式 `--model glm-5.2` 或在 shell 里 `$env:ANTHROPIC_MODEL='glm-5.2'`
- fallback 组：`auto → ['glm-5.2','deepseek-pro','deepseek','sensenova-lite',...]`

## 联调结论（Studio 节点）

- `LlmPolishNode provider=litellm + model=auto/glm-5.2`：网关通时可返回润色文本
- `provider=sensenova + model=sensenova-u1-fast`：chat **不可用**，需图/视频专用 endpoint（后续卡）

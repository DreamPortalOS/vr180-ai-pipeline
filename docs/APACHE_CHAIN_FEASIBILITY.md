# L-1 Apache 链路可行性笔记（#368 · 2026-09-17 夜班）

> 状态：**调研骨架**，非实现卡。阻塞 VR180 商业化的核心是 DepthCrafter/StereoCrafter 许可。

## 结论（先看）

| 路径 | 许可 | 质量预期 | 12GB 可行性 | 建议 |
|---|---|---|---|---|
| 现状 DepthCrafter + StereoCrafter | 非 Apache / 研究友好 | 已验证可用 | Mac M2 / 本地 GPU | 商用前必须替换或谈授权 |
| Depth-Anything-V2 + 本仓 disparity | 多数权重 Apache/MIT 类 | 时序弱、遮挡鬼影风险 | 4070S 可 | **默认保底商业路径** |
| 纯 ffmpeg / 光流伪深度 | Apache（工具） | 立体弱 | 易 | 仅作 fallback / 调试 |
| 商业 API 深度 | 各家 ToS | 未知 | N/A | 单独评估数据出境 |

## 已有代码落点

- `pipeline/depth_estimator.py` — Depth-Anything 单帧路径
- `pipeline/stereo_renderer.py` — 视差渲染（非 Crafter）
- `pipeline/depth_crafter.py` / `stereo_crafter.py` — 可插拔后端（重、许可紧）
- Studio：`convert.vr180` CLI 默认可 `--device cpu` + mock 几何

## 今晚不做

- 不下载 Apache 权重到 CI/主检出
- 不在测试里跑真实深度推理
- 不合并任何「偷偷换后端」的静默改动

## 建议下一步（明早卡）

1. 读齐 `LICENSES.md` 中 DepthCrafter/StereoCrafter 条款原文摘录
2. 定义 **商用默认 profile**：`depth=DA-V2, stereo=shift, no crafter`
3. 用同一段 mock/本地素材做 **质量对比矩阵**（鬼影、时序闪烁、耗时）—— owner 提供非敏感样片后再跑
4. Studio 增加 `vr180.backend=apache|full` 节点参数（默认 apache）
5. PR：文档 + 参数开关 + mock 测试

## 验收（未来实现卡）

- [ ] `LICENSES.md` 更新商用路径说明
- [ ] `run_pipeline --stereo-backend apache`（或等价）存在且默认可选
- [ ] CI 不下载非 Apache 权重
- [ ] 质量对比表（哪怕只有 2 条样片）写入 docs

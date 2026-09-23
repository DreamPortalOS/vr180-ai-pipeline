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

## 实测（2026-09-24）

同一素材双臂实测（#397）。源片：`video/gen_v10_4k.mp4`（2880×2880，241 帧，24 fps）。
两臂命令完全一致：`--quality standard --comfort balanced --preset standalone
--input-projection fisheye --fisheye-fov 180 --pix-fmt yuv420p10le`；立体渲染两臂均使用
本仓默认 renderer（**非** StereoCrafter），唯一变量是深度模型。

| 臂 | 耗时 | 显存峰值 | 许可 |
|---|---|---|---|
| DepthCrafter | 1090 s（深度命中缓存；冷深度按本仓笔记约 +34 min） | 3575 MiB | 学术/研究限定，**不可商用** |
| Depth-Anything-V2-Small | 1146 s | 4132 MiB | **Apache-2.0**（已从缓存的 HF model card 核实） |

- 两条产物均通过 `vr180_qa` 6/6：5760×2880 HEVC 10-bit、sv3d+st3d box、SBS 2×2880² 方眼布局。
- **许可陷阱**：Depth-Anything-V2 的 Base/Large 变体是 CC-BY-NC，**不是** Apache；
  商用链路必须钉死 **Small**。
- **教训**：第一次对比跑在 worktree 里因 DepthCrafter 路径缺失**静默回退**到 depth-anything，
  两臂产出字节级相同（无效对比）。通过显式设置 `DEPTHCRAFTER_REPO_DIR` / `DEPTHCRAFTER_PYTHON` /
  `DEPTHCRAFTER_MODEL_DIR` 修复；防回退护栏见 issue #410。
- **质量判定**：PENDING —— 待 owner 在 Quest 上盲审 `video/ab/ab_A.mp4` vs `video/ab/ab_B.mp4`
  （臂↔文件映射密封于 KEY.txt，不入仓）。

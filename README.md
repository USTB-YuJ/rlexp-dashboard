# RL Experiment Dashboard

面向机器人强化学习实验的本地优先 Web Dashboard。它用于索引 RSL-RL 风格训练日志，汇总 TensorBoard 指标、参数配置、checkpoint、视频、人工观察记录和实验 lineage，帮助你在大量训练实验之间追踪“改了什么、结果如何、下一步该看哪个模型”。

## 中文快速开始

如果你只是想在本机打开 dashboard，最短路径是：

```bash
git clone git@github.com:USTB-YuJ/rlexp-dashboard.git
cd rlexp-dashboard
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
pip install -e ".[dev]"
rl-exp-dashboard serve --workspace ~/rl-exp-dashboard
```

然后在浏览器打开：

```text
http://127.0.0.1:7860
```

`~/rl-exp-dashboard` 是本地工作区，默认会保存 `dashboard.sqlite3` 数据库和你同步下来的远端日志缓存。

## 安装方式

### 本地开发安装

推荐当前阶段使用源码安装，方便更新和调试：

```bash
git clone git@github.com:USTB-YuJ/rlexp-dashboard.git
cd rlexp-dashboard
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
pip install -e ".[dev]"
```

其中 `.[dev]` 会安装 YAML/TensorBoard 解析和本地 server 需要的依赖。

### 普通 pip 安装

如果后续发布到 PyPI，可直接安装：

```bash
pip install rl-exp-dashboard
```

### wheel 离线安装

在一台机器上构建：

```bash
python -m pip install build
python -m build
```

把 `dist/*.whl` 拷到目标机器后安装：

```bash
pip install dist/rl_exp_dashboard-0.1.0-py3-none-any.whl
```

### Docker 运行

如果希望隔离环境，可以使用容器：

```bash
docker build -t rl-exp-dashboard .
docker run -p 7860:7860 -v ~/rl-exp-dashboard:/data rl-exp-dashboard
```

等价的公开镜像运行形式是：

```bash
docker run -p 7860:7860 -v ~/rl-exp-dashboard:/data rl-exp-dashboard
```

## 启动 Dashboard

```bash
rl-exp-dashboard serve --workspace ~/rl-exp-dashboard
```

默认监听：

```text
http://127.0.0.1:7860
```

如果需要让局域网内其他机器访问，可以指定 host：

```bash
rl-exp-dashboard serve --workspace ~/rl-exp-dashboard --host 0.0.0.0 --port 7860
```

## 索引本地日志

如果训练日志已经在本机，例如 `~/workspace/unitree_rl_mjlab/logs/rsl_rl`：

```bash
rl-exp-dashboard index \
  --project unitree_rl_mjlab \
  --log-root ~/workspace/unitree_rl_mjlab/logs/rsl_rl \
  --db ~/rl-exp-dashboard/dashboard.sqlite3
```

索引后刷新网页即可看到 run 表格、metric、checkpoint、视频、artifact 和参数配置。

## 远端同步

如果训练日志在远端服务器，先预览同步命令：

```bash
rl-exp-dashboard sync \
  --project unitree_rl_mjlab \
  --source-name x-server \
  --host 1858d9aa66b16579.natapp.cc \
  --user eai \
  --port 12188 \
  --remote-log-root /home/eai/workspace/unitree_rl_mjlab/logs/rsl_rl \
  --cache-root ~/rl-exp-dashboard/cache \
  --db ~/rl-exp-dashboard/dashboard.sqlite3 \
  --dry-run
```

确认命令没问题后，去掉 `--dry-run` 执行实际同步：

```bash
rl-exp-dashboard sync \
  --project unitree_rl_mjlab \
  --source-name x-server \
  --host 1858d9aa66b16579.natapp.cc \
  --user eai \
  --port 12188 \
  --remote-log-root /home/eai/workspace/unitree_rl_mjlab/logs/rsl_rl \
  --cache-root ~/rl-exp-dashboard/cache \
  --db ~/rl-exp-dashboard/dashboard.sqlite3
```

默认同步会尽量避免拉取体积很大的视频和 checkpoint。需要一起同步时增加：

```bash
--include-videos --include-checkpoints
```

同步完成后再索引本地缓存：

```bash
rl-exp-dashboard index \
  --project unitree_rl_mjlab \
  --log-root ~/rl-exp-dashboard/cache/x-server \
  --db ~/rl-exp-dashboard/dashboard.sqlite3
```

注意：工具不会保存 SSH 密码。推荐使用 SSH key 或本机 ssh-agent。

## 项目配置文件

可以把常用项目和远端来源写成配置文件，再导入 dashboard：

```bash
rl-exp-dashboard project import \
  --config dashboard-project.yaml \
  --db ~/rl-exp-dashboard/dashboard.sqlite3
```

配置文件可以维护：

- 项目名称。
- 本地缓存目录。
- 解析 profile。
- 重点指标列表。
- 日志匹配规则。
- 远端同步来源。

导入后，Web 界面中的 Project Configs 和 Remote Sources 会显示这些配置。

## 界面语言切换

Dashboard 顶部提供 Language / 语言 选择器：

- `English`：英文界面。
- `中文`：中文界面。

选择结果会保存到浏览器 `localStorage` 的 `dashboardLanguage` 中，下次打开会自动恢复。当前版本优先翻译主要导航、筛选器、项目配置、远端同步和常用空状态；部分实验参数名、metric tag 和用户填写内容会保持原始文本。

## 常用工作流

### 1. 查看新训练

```bash
rl-exp-dashboard sync ... --db ~/rl-exp-dashboard/dashboard.sqlite3
rl-exp-dashboard index --project unitree_rl_mjlab --db ~/rl-exp-dashboard/dashboard.sqlite3
rl-exp-dashboard serve --workspace ~/rl-exp-dashboard
```

网页里查看 Runs、Metric Trend、Videos 和 Checkpoints。

### 2. 对比两次实验

在网页中打开 Compare Runs：

- Baseline 选择旧实验。
- Target 选择新实验。
- 查看配置差异、指标曲线、人工评价和视频对比。

### 3. 记录人工观察

进入某个 run 的详情页，在 Manual Observation 中记录：

- verdict：good / mixed / bad / exported。
- summary：策略表现总结。
- tags：例如 `stairs`、`backward-walk`、`sitting`。
- recommended checkpoint：推荐 checkpoint。

这些信息会进入 run table、lineage graph 和 compare view。

### 4. 导出单次实验报告

```bash
rl-exp-dashboard report \
  --db ~/rl-exp-dashboard/dashboard.sqlite3 \
  --run-id g1_depth_parkour_amp/2026-06-23_00-26-53 \
  --output report.md
```

报告会包含 run 元信息、lineage、metric、人工观察和 checkpoint review。

## 当前会索引哪些内容

- YAML、JSON、TOML 参数文件。
- TensorBoard scalar summary 和采样后的 metric series。
- `model_*.pt` checkpoint。
- play 视频和导出的 policy artifact。
- git 元信息。
- 从 resume / load-run 参数推断出的 run lineage。
- 手动填写的 run observation 和 checkpoint review。

## 设计文档

设计稿在：

```text
docs/superpowers/specs/2026-06-26-rl-experiment-dashboard-design.md
```

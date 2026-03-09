# Cursor 对话记录与 Plan 文件查找机制

## 目的

这份文档用于说明 Cursor 在本机上与当前项目相关的三类内容如何查找：

- 侧边栏里看到的聊天历史和 UI 状态
- Agent 的原始 transcript
- Cursor 本地保存的 plan 文件

当前文档以 Linux 环境为主，结合本项目 `forklift_sim` 的实际路径整理。

## 总览

Cursor 相关内容通常不是只存在一个地方，而是分成三层：

1. `workspaceStorage`
   保存某个 workspace 的本地 UI 状态，常见包括聊天侧边栏状态、工作区状态等。
2. `~/.cursor/projects/.../agent-transcripts`
   保存新版 Agent 的原始对话转录，通常比侧边栏历史更完整。
3. `~/.cursor/plans`
   保存 Cursor 本地的 plan 文件，和项目内 `docs/...` 里的计划文档不是同一层存储。

所以当你感觉“聊天记录没了”时，实际可能是：

- 只是侧边栏没显示出来
- 但 transcript 还在
- plan 文件也还在

## 一、侧边栏聊天历史与 UI 状态

### Linux 默认位置

```text
~/.config/Cursor/User/workspaceStorage/
```

这个目录下面通常会有多个 hash 目录，例如：

```text
~/.config/Cursor/User/workspaceStorage/<hash>/
```

每个 hash 目录通常至少会有这些文件：

- `workspace.json`
- `state.vscdb`
- `state.vscdb.backup`

其中：

- `workspace.json` 用来标识这个 hash 目录对应哪个项目目录
- `state.vscdb` 是 SQLite 数据库，保存该 workspace 的本地状态
- `state.vscdb.backup` 是备份

### 如何判断哪个 hash 属于当前项目

方法是逐个看：

```text
~/.config/Cursor/User/workspaceStorage/<hash>/workspace.json
```

只要里面的 `folder` 字段等于你的项目路径，就说明这个 hash 对应这个项目。

对于当前项目，已经确认对应关系如下：

- 项目路径：`/home/uniubi/projects/forklift_sim`
- workspace hash：`ae1e10ff5420e4380fd2cb1f78c7ac3f`

对应文件：

- `~/.config/Cursor/User/workspaceStorage/ae1e10ff5420e4380fd2cb1f78c7ac3f/workspace.json`
- `~/.config/Cursor/User/workspaceStorage/ae1e10ff5420e4380fd2cb1f78c7ac3f/state.vscdb`

## 二、Agent 原始 transcript

### 根目录

```text
~/.cursor/projects/
```

Cursor 会把不同项目映射成不同目录名。当前项目：

```text
/home/uniubi/projects/forklift_sim
```

对应到了：

```text
~/.cursor/projects/home-uniubi-projects-forklift-sim/
```

### transcript 位置

原始 transcript 在：

```text
~/.cursor/projects/home-uniubi-projects-forklift-sim/agent-transcripts/
```

这里通常有两种形式：

1. 顶层转录文件

```text
agent-transcripts/<uuid>.txt
```

这是比较容易直接阅读的文本版转录。

2. 按会话分目录的原始文件

```text
agent-transcripts/<uuid>/<uuid>.jsonl
```

这是更原始的逐条消息格式，通常更完整。

有时还会看到：

```text
agent-transcripts/<uuid>/subagents/
```

这里是子 agent 的记录。排查主会话时，优先看父级 `<uuid>.jsonl` 或同名 `.txt` 即可。

### 当前项目的实际结论

已经确认当前项目的 transcript 没丢，仍然保存在：

```text
~/.cursor/projects/home-uniubi-projects-forklift-sim/agent-transcripts/
```

也就是说，如果侧边栏历史空了，不代表原始对话没了。

## 三、Cursor 本地 Plan 文件

### 默认位置

```text
~/.cursor/plans/
```

这里保存的是 Cursor 本地 plan 文件，常见命名形式类似：

```text
<标题>_<uuid>.plan.md
```

例如之前项目里出现过这类文件：

```text
/home/uniubi/.cursor/plans/视觉升级与孔位检测预训练计划_daa8b726.plan.md
```

这类 plan 文件的特点是：

- 保存在用户本机，不在仓库 git 管理里
- 可能和项目里的 `docs/...` 文档内容相似
- 但它们不是同一个文件，需要手动同步

## 四、推荐查找顺序

如果以后再遇到“聊天历史不见了”或“计划找不到了”，建议按下面顺序排查。

### 1. 先确认是不是打开了同一个 workspace

先看当前打开的项目目录是否还是原来的仓库目录。

对本项目来说，应确认是：

```text
/home/uniubi/projects/forklift_sim
```

如果打开的是别的目录，Cursor 往往会显示一个全新的空上下文。

### 2. 找到对应的 workspace hash

去：

```text
~/.config/Cursor/User/workspaceStorage/
```

逐个检查 `workspace.json`，找到 `folder` 对应当前项目的目录。

### 3. 检查 `state.vscdb`

如果 hash 对上了，但侧边栏仍然没显示历史，说明：

- 可能是 UI 状态异常
- 也可能是 Cursor 版本切换后未正确加载

这时不要直接判定历史丢失，继续看 transcript。

### 4. 检查 `agent-transcripts`

去：

```text
~/.cursor/projects/<项目映射目录>/agent-transcripts/
```

优先查看：

- 最新的 `<uuid>.txt`
- 最新的 `<uuid>/<uuid>.jsonl`

只要这些还在，原始对话通常就还在。

### 5. 检查本地 plan

去：

```text
~/.cursor/plans/
```

按文件名里的标题关键词搜索，例如：

- `视觉`
- `预训练`
- `forklift`

## 五、当前项目的快速定位结果

当前项目已经确认的几个关键位置如下。

### workspaceStorage

- `~/.config/Cursor/User/workspaceStorage/ae1e10ff5420e4380fd2cb1f78c7ac3f/workspace.json`
- `~/.config/Cursor/User/workspaceStorage/ae1e10ff5420e4380fd2cb1f78c7ac3f/state.vscdb`

### Agent transcript

- `~/.cursor/projects/home-uniubi-projects-forklift-sim/agent-transcripts/`

### 本地 plan

- `~/.cursor/plans/`

## 六、一句话结论

Cursor 的“侧边栏聊天历史”、“原始 Agent transcript”、“本地 plan 文件”是三套相互有关但不完全相同的存储：

- `workspaceStorage` 负责 workspace 级 UI 状态
- `agent-transcripts` 负责原始对话记录
- `~/.cursor/plans` 负责本地计划文件

排查时不要只盯侧边栏，应该把这三层都一起看。

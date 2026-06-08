<div align="center">

# 🎯 JobFit Agent

**LLM 提取 + 程序匹配的简历-JD 分析系统**

![Python](https://img.shields.io/badge/Python-3.11+-blue?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688?logo=fastapi&logoColor=white)
![DeepSeek](https://img.shields.io/badge/DeepSeek-Chat-4D6BFE?logo=openai&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?logo=docker&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)

上传简历 + 粘贴 JD → 结构化匹配分析报告

</div>

---

## 📖 项目简介

JobFit Agent 是一个简历-JD 匹配分析系统。核心设计原则：**LLM 负责理解，程序负责判断，LLM 负责表达**。

用户上传简历（PDF/DOCX/TXT/Markdown）或直接粘贴文本，配合岗位 JD，系统输出：

- 🎯 **匹配度评分**（程序加权计算，0-100）
- 📊 **分维度得分**（技能/经验/项目/学历）
- ✅ **逐项匹配详情**（匹配度 + 简历证据）
- ⚠️ **能力缺口与风险项**（未匹配的核心需求）
- 💡 **简历优化建议**（针对性改写）
- ❓ **高频面试问题**（基于匹配薄弱环节生成）

## 🏗️ 系统架构

```
  +--------------+       +--------------+
  |   简历文本    |       |    JD 文本    |
  +------+-------+       +------+-------+
         |                      |
         v                      v
  +--------------+       +--------------+
  |   LLM 提取   |       |   LLM 提取    |
  |  结构化 JSON |       |    需求列表   |
  +------+-------+       +------+-------+
         |                      |
         +----------+-----------+
                    |
                    v
         +--------------------+
         |    程序匹配引擎     |
         | 同义词 + Embedding |
         +---------+----------+
                   |
                   v
         +--------------------+
         |      匹配结果      |
         |     分数 + 详情    |
         +---------+----------+
                   |
                   v
         +--------------------+
         |    LLM 生成建议    |
         |    优化 + 面试题   |
         +---------+----------+
                   |
                   v
               返回结果
```

**四步流水线：**

1. **LLM 提取简历** — 从简历中提取 skills / projects / experience（只负责理解）
2. **LLM 提取 JD** — 从 JD 中提取 requirements + priority（只负责理解）
3. **程序匹配** — 同义词表 + Embedding 语义匹配，确定性计算（不幻觉）
4. **LLM 建议** — 基于匹配结果生成优化建议和面试题（只负责表达）

**核心设计原则：** LLM 只做它擅长的事（理解非结构化文本、生成人类可读输出），程序做它擅长的事（确定性计算、可复现结果）。匹配打分不经过 LLM，结果稳定不幻觉。

## ✨ 功能特性

| 功能 | 说明 |
|------|------|
| 多格式简历解析 | 支持 PDF、DOCX、TXT、Markdown，自动编码检测 |
| AI 需求提取 | LLM 从 JD 提取需求，支持任意岗位类型 |
| AI 简历提取 | LLM 从简历提取结构化数据（技能、项目、经验、学历） |
| 程序匹配引擎 | 同义词表（~80 条高频映射）+ BGE Embedding 语义匹配 |
| 加权评分 | 按需求级别（required/preferred/nice-to-have）加权计算 |
| 四层容错 | 强约束 prompt → 自动重试 → 字段兜底 → Pydantic 校验 |
| 响应式前端 | 单页报告，支持桌面/平板/手机 |
| Docker 部署 | 一键启动 |

## 🚀 快速开始

### 环境要求

- Python 3.11+
- DeepSeek API Key

### 本地运行

```bash
# 1. 克隆项目
git clone https://github.com/lzffzl13/JobFit.git
cd JobFit

# 2. 创建虚拟环境
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 3. 安装依赖
pip install -r requirements.txt

# 4. 配置环境变量
cp .env.example .env
# 编辑 .env 填入 DEEPSEEK_API_KEY

# 5. 启动服务
uvicorn app.main:app --reload --port 9000
```

访问：

- 🌐 **Web 页面：** http://127.0.0.1:9000
- 📚 **API 文档：** http://127.0.0.1:9000/docs

### Docker 部署

```bash
cp .env.example .env
docker compose up --build
```

## 📐 评分机制

程序按需求级别加权计算，LLM 不参与打分：

| 需求级别 | 权重 | 含义 |
|----------|------|------|
| `required` | 5 | 必须具备 |
| `preferred` | 2 | 优先考虑 |
| `nice-to-have` | 1 | 加分项 |

分类维度加权汇总：

| 维度 | 权重 | 匹配方式 |
|------|------|----------|
| 技能 | 4 | 同义词表 + Embedding 相似度 |
| 经验 | 3 | 年限数值比较 |
| 项目 | 3 | Embedding 语义匹配 |
| 学历 | 2 | 学位等级规则匹配 |

## 🗂️ 项目结构

```
JobFit/
├── app/
│   ├── main.py                  # FastAPI 应用入口
│   ├── api/
│   │   └── jobfit.py            # POST /jobfit/analyze 端点
│   ├── core/
│   │   ├── config.py            # 环境变量配置（pydantic-settings）
│   │   └── interfaces.py        # ILLMClient 协议
│   ├── schemas/
│   │   └── jobfit.py            # Pydantic 数据模型（Resume/JD/Match）
│   └── services/
│       ├── document_parser.py   # 文档解析（PDF/DOCX/TXT/MD）
│       ├── jobfit.py            # 编排器：提取 → 匹配 → 建议
│       ├── llm.py               # LLM 提取 + 建议生成 + 容错
│       ├── matcher.py           # 程序匹配引擎（同义词 + Embedding）
│       └── llm_clients/
│           ├── base.py          # BaseLLMClient 抽象基类
│           ├── deepseek.py      # DeepSeek API 实现
│           ├── openai.py        # OpenAI 占位（未实现）
│           └── factory.py       # get_llm_client() 工厂
├── static/
│   └── index.html               # 单页前端（原生 HTML/CSS/JS）
├── samples/                     # 示例数据
├── tests/                       # 测试用例
│   ├── test_api.py              # 集成测试（依赖真实 API）
│   ├── test_api_text.py         # 文本输入集成测试 + mock
│   ├── test_matcher.py          # 匹配引擎单元测试
│   └── test_llm_extractors.py   # LLM 提取容错测试
├── Dockerfile
├── docker-compose.yml
├── pyproject.toml
└── requirements.txt
```

## 🔌 API 接口

### `POST /jobfit/analyze`

**请求参数（form-data）：**

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `resume` | file | 否 | 简历文件（.pdf/.docx/.txt/.md） |
| `resume_text` | string | 否 | 直接粘贴的简历文本（优先级高于文件） |
| `jd_text` | string | ✅ | 岗位 JD 文本（最少 20 字） |

**响应字段：**

| 字段 | 说明 |
|------|------|
| `match_score` | 匹配度评分（0-100，程序计算） |
| `summary` | 分析总结（LLM 生成） |
| `score_breakdown` | 分维度得分（技能/经验/项目/学历） |
| `matched_strengths` | 已匹配项（匹配度 + 简历证据） |
| `gaps` | 能力缺口 |
| `resume_rewrites` | 简历优化建议 |
| `interview_questions` | 面试问题 |
| `risk_items` | 低匹配核心需求 |

## 🧪 测试

```bash
# 运行全部单元测试（不需要 API）
pytest tests/test_matcher.py tests/test_llm_extractors.py tests/test_api_text.py -v

# 运行全部测试（集成测试需要 DeepSeek API）
pytest -v

# 代码检查
ruff check app tests
```

**测试覆盖：**

- ✅ 匹配引擎（同义词、技能匹配、经验匹配、教育匹配、完整流水线）
- ✅ LLM 提取容错（JSON 解析、字段兜底、类型强制）
- ✅ API 集成测试（mock LLM，验证端到端流程）

## ⚙️ 环境变量

在 `.env` 中配置：

```env
LLM_PROVIDER=deepseek           # LLM 提供商
DEEPSEEK_API_KEY=               # DeepSeek API Key
DEEPSEEK_BASE_URL=https://api.deepseek.com
DEEPSEEK_MODEL=deepseek-chat
LLM_TIMEOUT_SECONDS=60          # LLM 请求超时时间（三次调用累加）
EMBEDDING_MODEL=BAAI/bge-small-zh-v1.5  # Embedding 模型
```

## 📌 版本演进

| 版本 | 匹配方式 | 优点 | 缺点 |
|------|----------|------|------|
| V1 | 硬编码规则 + AI 建议 | 预设岗位匹配准确，评分客观 | 代码量大，可扩展性差，评分精度低 |
| V2 | LLM 全包（提取+判断+打分+建议） | 代码简洁，适合各种岗位 | 幻觉严重（大模型底层问题） |
| V3 | LLM 提取 + 程序匹配 + LLM 建议 | 结合两者优点，程序判断不幻觉 | LLM 提取可能丢失信息，输出格式不可控 |

## 🗺️ 后续优化

### 高优先级

- [ ] **V4 架构优化** — 简历不提取，程序直接在原文上匹配（同义词 + Embedding），消除信息丢失
- [ ] **Prompt 迭代** — 提取 prompt A/B 验证，提升提取准确率
- [ ] **同义词表扩充** — 覆盖更多行业术语和技术栈

### 中优先级

- [ ] **缓存机制** — 相同 resume+JD 组合缓存结果，减少 API 调用
- [ ] **流式输出** — SSE 流式返回，前端实时显示分析进度
- [ ] **多模型支持** — 补全 OpenAI client，支持 GPT-4o / Claude 对比测试
- [ ] **Embedding 阈值调优** — 当前 0.8 阈值偏高，需要根据实际数据调优

### 低优先级

- [ ] **导出功能** — 分析报告导出为 Markdown/PDF
- [ ] **历史记录** — SQLite 存储分析结果，支持查看和对比
- [ ] **批量分析** — 一次投多个 JD 或多个简历
- [ ] **CI/CD 流水线** — 自动测试 + 部署

### 后期升级

从"分析工具"升级为"求职 Agent"——不只分析匹配度，直接帮用户优化简历。

- [ ] **简历自动优化** — 输入原始简历 + JD，输出针对该 JD 优化后的简历
- [ ] **优化前后对比** — 重新跑匹配分析对比分数，分数没提升则回退
- [ ] **多轮对话** — 用户可对话调整优化方向
- [ ] **批量投递建议** — 一份简历 + 多个 JD → 优先级排序

## 📄 License

MIT

---

<div align="center">

**如果这个项目对你有帮助，请给个 ⭐ Star 支持一下！**

</div>

<div align="center">

# JobFit Agent

**一个面向求职场景的简历分析与面试辅助项目**

![Python](https://img.shields.io/badge/Python-3.11+-blue?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688?logo=fastapi&logoColor=white)
![DeepSeek](https://img.shields.io/badge/DeepSeek-Chat-4D6BFE?logo=openai&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?logo=docker&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)

上传简历，粘贴 JD，生成匹配分析、简历优化建议和面试准备内容。

</div>

---

## 项目简介

JobFit Agent 是一个围绕简历和 JD 的分析项目，用来帮助用户更快看清一份简历和目标岗位之间的匹配情况，并给出更具体的优化和准备方向。

项目的核心思路是：

- 用 LLM 处理简历、JD 这类非结构化文本
- 用程序完成相对稳定的匹配和打分
- 再基于分析结果生成更具体的简历优化建议和面试问题

目前项目已经支持：

- 简历结构化提取
- JD 要求提取
- 程序化匹配分析
- 匹配缺口与风险项输出
- 简历优化建议
- 面试问题生成

后续会继续补强简历优化和模拟面试能力，也可以进一步扩展为更完整的求职平台。

## 系统架构

```text
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

处理流程分成四步：

1. 从简历中提取结构化信息
2. 从 JD 中提取岗位要求
3. 用程序完成匹配和打分
4. 基于结果生成建议和面试准备内容

## 功能特性

| 功能 | 说明 |
|------|------|
| 多格式简历解析 | 支持 PDF、DOCX、TXT、Markdown，自动编码检测 |
| 简历结构化提取 | 从简历中提取技能、项目、经验、学历等信息 |
| JD 分析 | 从 JD 中提取要求、优先级、关键词和风险点 |
| 程序匹配引擎 | 同义词表（~80 条高频映射）+ BGE Embedding 语义匹配 |
| 加权评分 | 按需求级别（required/preferred/nice-to-have）加权计算 |
| 匹配解释 | 输出分维度得分、匹配项、缺口和风险项 |
| 简历优化建议 | 基于匹配结果生成针对性的优化建议 |
| 面试准备 | 基于简历和 JD 生成面试问题 |
| 四层容错 | 强约束 prompt -> 自动重试 -> 字段兜底 -> Pydantic 校验 |
| Docker 部署 | 支持本地一键启动 |

## 快速开始

### 环境要求

- Python 3.11+
- DeepSeek API Key

### 本地运行

```bash
git clone https://github.com/lzffzl13/JobFit.git
cd JobFit

python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

pip install -r requirements.txt

cp .env.example .env
# 编辑 .env，填入 DEEPSEEK_API_KEY

uvicorn app.main:app --reload --port 9000
```

访问：

- Web 页面：http://127.0.0.1:9000
- API 文档：http://127.0.0.1:9000/docs

### Docker 部署

```bash
cp .env.example .env
docker compose up --build
```

## 评分机制

程序按需求级别加权计算，LLM 不直接参与打分：

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

## 项目结构

```text
JobFit/
├── app/
│   ├── main.py                  # FastAPI 应用入口
│   ├── api/
│   │   └── jobfit.py            # POST /jobfit/analyze 端点
│   ├── core/
│   │   ├── config.py            # 环境变量配置
│   │   └── interfaces.py        # LLM client 协议
│   ├── schemas/
│   │   └── jobfit.py            # Pydantic 数据模型
│   └── services/
│       ├── document_parser.py   # 文档解析
│       ├── jobfit.py            # 编排器：提取 -> 匹配 -> 建议
│       ├── llm.py               # LLM 提取与建议生成
│       ├── matcher.py           # 程序匹配引擎
│       └── llm_clients/
│           ├── base.py
│           ├── deepseek.py
│           ├── openai.py
│           └── factory.py
├── static/
│   └── index.html               # 单页前端
├── samples/                     # 示例数据
├── tests/                       # 测试用例
├── Dockerfile
├── docker-compose.yml
├── pyproject.toml
└── requirements.txt
```

## API 接口

### `POST /jobfit/analyze`

请求参数（`form-data`）：

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `resume` | file | 否 | 简历文件（.pdf/.docx/.txt/.md） |
| `resume_text` | string | 否 | 直接粘贴的简历文本，优先级高于文件 |
| `jd_text` | string | 是 | 岗位 JD 文本，最少 20 字 |

响应字段：

| 字段 | 说明 |
|------|------|
| `match_score` | 匹配度评分（0-100） |
| `summary` | 分析总结 |
| `score_breakdown` | 分维度得分（技能/经验/项目/学历） |
| `matched_strengths` | 已匹配项（匹配度 + 简历证据） |
| `gaps` | 能力缺口 |
| `resume_rewrites` | 简历优化建议 |
| `interview_questions` | 面试问题 |
| `risk_items` | 低匹配核心需求 |

## 测试

```bash
# 运行主要测试
pytest tests/test_matcher.py tests/test_llm_extractors.py tests/test_api_text.py -v

# 运行全部测试
pytest -v

# 代码检查
ruff check app tests
```

当前测试覆盖包括：

- 匹配引擎
- LLM 提取容错
- API 集成测试

## 环境变量

在 `.env` 中配置：

```env
LLM_PROVIDER=deepseek
DEEPSEEK_API_KEY=
DEEPSEEK_BASE_URL=https://api.deepseek.com
DEEPSEEK_MODEL=deepseek-chat
LLM_TIMEOUT_SECONDS=60
EMBEDDING_MODEL=BAAI/bge-small-zh-v1.5
```

## 版本演进

| 版本 | 匹配方式 | 特点 |
|------|----------|------|
| V1 | 硬编码规则 + AI 建议 | 规则多，扩展性有限 |
| V2 | LLM 全包（提取 + 判断 + 打分 + 建议） | 实现简单，但稳定性一般 |
| V3 | LLM 提取 + 程序匹配 + LLM 建议 | 可解释性和稳定性更好 |

## 后续方向

- 补强简历优化模块
- 补强模拟面试与复盘模块
- 增加历史记录与结果对比
- 逐步扩展为更完整的求职平台

## License

MIT

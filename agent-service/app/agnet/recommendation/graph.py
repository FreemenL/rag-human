from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from app.core.config import settings
import os
from .tools import tools
from .models import RecommendationPlanState
from ..models import ChatRuntimeContext

load_dotenv()

RECOMMENDATION_PLAN_SYSTEM_PROMPT = """
# 角色定义
你是保险商城的专业保险顾问，精通各类保险产品，擅长为个人客户量身定制保险组合方案。你的核心职责包括：方案推荐、产品查询、组合调整及方案保存。

# 核心工作原则
- **以客户画像为中心**：所有推荐必须基于收集到的完整被保人信息（年龄、职业、预算）。
- **诚实透明**：明确区分“方案推荐”与“正式报价”，工具调用失败时如实反馈，绝不伪造成功信息。
- **流程严谨**：严格遵循工作流顺序，避免跳步或擅自推测。

# 详细工作流程

## 1. 被保人画像收集（信息补齐）
- 必须收集三项核心信息：**年龄、职业、年预算**。
- 信息不全时，**每轮仅追问最关键的 1-2 项**，优先补齐年龄和预算。
- 信息完整后，立即调用 `update_insured_profile` 保存画像，但不要与用户提画像的事情。

## 2. 候选产品查询
- 根据画像调用 `load_candidates` 获取可推荐产品。
- 若用户指定险种，传入对应参数（medical / critical_illness / accident / life）；未指定则留空，由系统全量匹配。

## 3. 方案构建（严格限制）
- **必须基于候选产品进行横向比较**，结合用户画像（年龄风险、职业特性、预算）选出最优组合。
- 方案限制：
  - 最多包含 **4 个产品**；
  - **每个险种最多 1 个产品**；
  - 所有产品的 `product_id` **不可重复**。

## 4. 知识查询（条款与责任）
- 当用户询问保障责任、免责条款、等待期或特定术语定义时，调用 `search_insurance_knowledge`。
- 你的回答必须是基于检索到的保险条款，不要自己编造回答，如果没有相关条款，如实告知用户。
- 回答中的每个事实陈述必须紧跟[source_id]的引用标记。例如：这款产品的等待期为180天。[ref-001]
- 回答要简短、精确，不要长篇大论。

## 5. 保费表述规范
- 方案中的金额统一表示**年缴预算参考**，严禁作为正式报价。
- 必须同步说明：**最终以产品试算、健康告知、核保及保险公司实际结果为准**。

## 6. 异常处理
- 任何工具调用（Tool Call）失败时，**必须如实告知用户失败原因**，不得声称操作已成功。

""".strip()

# 利用LangChain写个简化的Agent
class Recommendation:

    def __init__(self):
        # 初始化模型，应该是一个本地小模型
        self.model = init_chat_model(
            model="qwen3.7-plus",
            model_provider="openai",
            base_url=os.getenv("DASHSCOPE_BASE_URL"),
            api_key=os.getenv("DASHSCOPE_API_KEY"),
            extra_body={
                "thinking": {
                    "type": "disabled"
                }
            }
        )

    def build_graph(self):
        return create_agent(
            model=self.model,
            tools=tools,
            system_prompt=RECOMMENDATION_PLAN_SYSTEM_PROMPT,
            state_schema=RecommendationPlanState,
            context_schema=ChatRuntimeContext,
            checkpointer=True
        )
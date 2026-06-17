import requests
import json

# 1. 设置 API Key 和 URL
API_KEY = "你的API_KEY"   # 在 AI Studio 控制台获取
API_URL = "https://aistudio-api.openai.com/v1/chat/completions"  # 示例接口地址，具体以 AI Studio 文档为准

# 2. 定义 Agent B 的 Prompt
def agent_b_refine(node_text, coarse_label):
    """
    Agent B: 根据节点属性和 Agent A 的粗标签，做语义细化
    """
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }

    # Prompt 模板
    prompt = f"""
    Task: 根据以下节点属性和 Agent A 提供的粗标签，给出更细的子类建议，并说明理由。
    若不确定，请返回 "UNKNOWN"。
    输出格式为 JSON: {{"subtype":..., "confidence":0-1, "reasons":[...]}}。

    Example:
    - text: "Google Research, Mountain View"
    - agent_a: "公司-科技"
    Expected: {{"subtype": "企业-研究机构", "confidence": 0.92, "reasons": ["包含 Research，表明为研究机构"]}}

    Now refine:
    - text: "{node_text}"
    - agent_a: "{coarse_label}"
    """

    data = {
        "model": "gpt-4o-mini",   # 你在 AI Studio 免费层级可用的模型，比如 gpt-4o-mini
        "messages": [
            {"role": "system", "content": "You are Agent B, responsible for semantic refinement."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.3
    }

    # 3. 调用 API
    response = requests.post(API_URL, headers=headers, data=json.dumps(data))
    result = response.json()

    # 4. 提取输出
    refined = result["choices"][0]["message"]["content"]
    return refined


# 5. 测试调用
if __name__ == "__main__":
    node_text = "Tsinghua University, Beijing"
    coarse_label = "学术机构"
    output = agent_b_refine(node_text, coarse_label)
    print("Agent B 输出结果：", output)

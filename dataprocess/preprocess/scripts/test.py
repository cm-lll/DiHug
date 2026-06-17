from google import genai

client = genai.Client(vertexai=True, project="my-project-id", location="us-central1")

# 创建会话
session = client.chat.create(
    model="gemini-2.0-flash"
)

# 设置系统指令（相当于 system prompt）
session.system_instruction = "You are a JSON-only extractor. Always return subtype arrays."

# 发送用户消息
resp = session.send_message("Node JSON: {\"id\": 1, \"type\": \"author\", \"name\": \"煜航\"}")
print(resp.text)
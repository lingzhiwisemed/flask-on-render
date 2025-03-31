from flask import Flask, request, render_template
import requests
import os
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

app = Flask(__name__)

# DeepSeek API 配置
DEEPSEEK_API_URL = "https://api.deepseek.com/chat/completions"  # 根据实际情况调整API地址
API_KEY = os.getenv("DEEPSEEK_API_KEY")  # 从 .env 文件中读取 API 密钥

@app.route("/", methods=["GET", "POST"])
def index():
    formData = {
        'selfReport': '',
        'medicalHistory': '',
        'faceColor': '',
        'tongueColor': '',
        'tongueCoating': '',
        'eyeLook': '',
        'voiceQuality': '',
        'breathSound': '',
        'dietHabit': '',
        'sleepQuality': '',
        'moodStatus': '',
        'bowelMovement': '',
        'pulse': ''
    }
    chat_response = ""

    if request.method == "POST":
        # 获取用户输入并更新 formData
        formData.update(request.form)

        print("FormData:", formData)

        # 构建请求消息
        messageContent = f"""
            你是一个资深中医专家，请根据以下问诊信息进行综合分析：
            【主诉】{formData['selfReport']}
            【既往史】{formData['medicalHistory']}
            【面诊】面色{formData['faceColor']}，眼神{formData['eyeLook']}
            【闻诊】声音{formData['voiceQuality']}，呼吸声{formData['breathSound']}
            【问诊】饮食习惯{formData['dietHabit']}，睡眠质量{formData['sleepQuality']}，情绪状态{formData['moodStatus']}，排泄情况{formData['bowelMovement']}
            【舌诊】舌质{formData['tongueColor']}，苔{formData['tongueCoating']}
            【脉诊】脉{formData['pulse']}
            请给出：1.中医辨证 2.治疗原则 3.方药建议 4.生活调养建议
        """

        # 调用 DeepSeek API
        headers = {
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json"
        }
        data = {
            "model": "deepseek-chat",
            "messages": [
                {"role": "system", "content": "你是一个资深中医专家。"},
                {"role": "user", "content": messageContent}
            ],
            "stream": False
        }

        print("Request Body:", data)

        response = requests.post(DEEPSEEK_API_URL, headers=headers, json=data)

        # 解析 API 响应
        if response.status_code == 200:
            api_response = response.json()
            chat_response = api_response["choices"][0]["message"]["content"]
        else:
            chat_response = f"Error: Unable to get response from DeepSeek API. Status Code: {response.status_code}, Message: {response.text}"

    return render_template("index.html", formData=formData, chat_response=chat_response)

if __name__ == "__main__":
    app.run(debug=True)
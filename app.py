from flask import Flask, request, render_template
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import os

app = Flask(__name__)

# 深度学习模型配置（修改为你的本地路径）
# MODEL_PATH = "./models/deepseek-llm-7b-chat"  # 从HuggingFace下载的模型路径
# C:\Users\13490\Desktop\smart_tcm_diagnosis\deepseek-llm-7b-chat
MODEL_PATH = r"C:\Users\admin\Desktop\smart_tcm_diagnosis\deepseek-chat"
assert os.path.exists(MODEL_PATH), f"模型路径 {MODEL_PATH} 不存在"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 初始化模型和tokenizer（全局加载避免重复加载）
print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16 if DEVICE == "cuda" else torch.float32,
    device_map="auto"
)
print("Model loaded!")


def generate_response(prompt):
    """使用本地模型生成响应"""
    try:
        # 构建符合DeepSeek格式的输入
        messages = [
            {"role": "system", "content": "你是一个资深中医专家。"},
            {"role": "user", "content": prompt}
        ]
        input_ids = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to(DEVICE)

        # 生成参数配置
        outputs = model.generate(
            input_ids,
            max_new_tokens=1024,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id
        )

        # 解码并清理响应
        response = tokenizer.decode(outputs[0][input_ids.shape[1]:], skip_special_tokens=True)
        return response.strip()

    except Exception as e:
        return f"生成错误: {str(e)}"


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
        formData.update(request.form)

        # 构建输入提示（可扩展字段）
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

        chat_response = generate_response(messageContent.strip())

    return render_template("index.html", formData=formData, chat_response=chat_response)



if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)  # 生产环境应关闭debug
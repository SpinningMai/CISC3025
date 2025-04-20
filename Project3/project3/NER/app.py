from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from MEM import MEMM  # 导入你的 NER 模型
import pickle

app = FastAPI()

# 加载模板（HTML 文件）
templates = Jinja2Templates(directory="templates")

# 加载训练好的模型
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

# 假设你的 MEM 类有一个 predict_sentence 方法
def predict_ner(sentence):
    words = sentence.split()
    predictions = []
    for i, word in enumerate(words):
        # 调用你的模型预测（需根据你的 MEM.py 调整）
        pred = model.predict(word, previous_label="O", position=i)
        predictions.append((word, pred))
    return predictions

# 主页（GET 请求）
@app.get("/")
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# 处理表单提交（POST 请求）
@app.post("/predict")
async def predict(request: Request, sentence: str = Form(...)):
    ner_results = predict_ner(sentence)
    return templates.TemplateResponse(
        "index.html",
        {"request": request, "sentence": sentence, "results": ner_results},
    )

# 运行命令：uvicorn app:app --reload
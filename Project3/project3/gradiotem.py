import pickle

import gradio as gr
import nltk

from Project3.project3.NER.MEM import MEMM

# 下载 punkt_tab 数据
nltk.download('punkt')

# 加载训练好的模型
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

# 初始化 MEMM 对象
ner_model = MEMM()
ner_model.classifier = model

def predict_ner(sentence):
    """
    预测命名实体并格式化输出。
    """
    result = ner_model.predict_sentence(sentence)
   
    formatted_result = [f"\"{word}\":\t{label}" for word, label in result]
    return "\n".join(formatted_result)

# 定义 Gradio 界面
iface = gr.Interface(
    fn=predict_ner,  # 绑定的预测函数
    inputs="text",  # 输入类型：文本框
    outputs="text",  # 输出类型：文本框
    title="命名实体识别",  # 页面标题
    description="输入一个句子来识别命名实体。"  # 页面描述
)

# 启动界面
iface.launch()

import pickle
import gradio as gr
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
from NER.MEM import MEMM
import nltk
import copy

nltk.download('punkt')

# 加载 MEMM 模型
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

ner_model = MEMM("")
ner_model.classifier = model

def f_beta_score(precision, recall, beta=1.5):
    """计算 F-beta 分数（可以自定义 beta 参数）"""
    if precision + recall == 0:
        return 0
    return (1 + beta**2) * (precision * recall) / (beta**2 * precision + recall)

def train_model_and_visualize(beta, max_iter):
    classifier = MEMM("")
    train_samples = classifier.extract_samples()

    left = 0
    right = max_iter

    best_score = -1
    best_iter = -1
    best_classifier = None

    scores = []  # 存储每次迭代的分数
    iterations = []  # 存储迭代的次数

    while right - left >= 3:
        mid1 = left + (right - left) // 3
        mid2 = right - (right - left) // 3

        # 训练并评估 mid1
        classifier.train(train_samples, mid1 + 2)
        metrics1 = classifier.test()
        beta_f1 = f_beta_score(metrics1['precision'], metrics1['recall'], beta)

        # 训练并评估 mid2
        classifier.train(train_samples, mid2 + 2)
        metrics2 = classifier.test()
        beta_f2 = f_beta_score(metrics2['precision'], metrics2['recall'], beta)

        # 存储分数，用于后续可视化
        scores.append((beta_f1, beta_f2))
        iterations.append((mid1 + 2, mid2 + 2))

        # 更新最优模型
        if beta_f1 > best_score:
            best_score = beta_f1
            best_iter = mid1 + 2
            best_classifier = copy.deepcopy(classifier.classifier)
        if beta_f2 > best_score:
            best_score = beta_f2
            best_iter = mid2 + 2
            best_classifier = copy.deepcopy(classifier.classifier)

        # 缩小搜索范围
        if beta_f1 < beta_f2:
            left = mid1
        else:
            right = mid2

    # 最终评估（可选：在剩余范围内微调）
    classifier.best_classifier = best_classifier
    classifier.save_model(classifier.best_classifier)

    # 可视化训练过程
    fig, ax = plt.subplots()
    ax.plot([i[0] for i in iterations], [score[0] for score in scores], label="Iteration mid1")
    ax.plot([i[1] for i in iterations], [score[1] for score in scores], label="Iteration mid2")

    ax.set(xlabel="Iteration", ylabel="F-beta Score",
           title="Training Progress: F-beta Score vs. Iterations")
    ax.legend()

    import os

    # 创建 tmp 文件夹（如果不存在）
    os.makedirs("tmp", exist_ok=True)

    # 保存图像到 tmp 目录下
    plt.savefig("tmp/training_metrics.png")
    plt.close()

    return f"训练完成，最优模型已保存！最优迭代次数：{best_iter}", "tmp/training_metrics.png"

def predict_ner(sentence):
    result = ner_model.predict_sentence(sentence)
    formatted_result = [f"\"{word}\":\t{label}" for word, label in result]
    return "\n".join(formatted_result)

# Gradio 界面
with gr.Blocks() as demo:
    with gr.Tab("训练与可视化"):
        # 添加 BETA 和 MAX_ITER 的滑块
        beta_slider = gr.Slider(minimum=0.0, maximum=3.0, step=0.1, value=1.5, label="BETA")
        max_iter_slider = gr.Slider(minimum=1, maximum=50, step=1, value=25, label="MAX_ITER")
        
        # 训练按钮
        train_button = gr.Button("训练并可视化")
        
        # 输出训练结果和图表
        training_output = gr.Textbox(label="训练结果")
        training_plot = gr.Image(label="训练进度图")
        
        # 连接训练按钮与函数
        train_button.click(train_model_and_visualize, inputs=[beta_slider, max_iter_slider], outputs=[training_output, training_plot])

    with gr.Tab("预测句子"):
        # 预测部分
        sentence_input = gr.Textbox(label="输入句子")
        prediction_output = gr.Textbox(label="预测输出")
        
        # 预测按钮
        gr.Button("预测").click(predict_ner, sentence_input, prediction_output)
    
    # 启动界面
    demo.launch()

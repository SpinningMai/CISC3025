import pickle
import gradio as gr
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
from NER.MEM import MEMM
import nltk

nltk.download('punkt')

with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

ner_model = MEMM()
ner_model.classifier = model

def predict_ner(sentence):
    result = ner_model.predict_sentence(sentence)
    formatted_result = [f"\"{word}\":\t{label}" for word, label in result]
    return "\n".join(formatted_result)

def train_and_visualize(BETA, MAX_ITER):
    classifier = MEMM()
    train_samples = classifier.extract_samples()

    metrics = {
        "epochs": [],
        "accuracy": [],
        "precision": [],
        "recall": [],
        "f_score": [],
    }

    # Train and test model
    for epoch in tqdm(range(MAX_ITER)):
        classifier.train(train_samples, epoch + 2)
        classifier.test()

        metrics["epochs"].append(epoch)
        metrics["accuracy"].append(classifier.best_recall)  
        metrics["precision"].append(classifier.best_recall)  
        metrics["recall"].append(classifier.best_recall)
        metrics["f_score"].append(classifier.best_recall)

        if not classifier.test():
            break  

    classifier.test()
    classifier.save_best_model()
    classifier.classifier = classifier.best_classifier
    classifier.test()

    # Convert metrics to a DataFrame
    df = pd.DataFrame(metrics)

    # Plot training metrics
    plt.figure(figsize=(10, 5))
    df.set_index("epochs").plot(kind="line", title="Training Metrics", marker="o")
    plt.xlabel('Epochs')
    plt.ylabel('Scores')
    plt.tight_layout()

    # Save the plot
    import os

    # 创建 tmp 文件夹（如果不存在）
    os.makedirs("tmp", exist_ok=True)

    # 保存图像到 tmp 目录下
    plt.savefig("tmp/training_metrics.png")
    plt.close()
    
    # 返回图像路径和训练数据
    return "tmp/training_metrics.png", df

# Gradio interface
with gr.Blocks() as demo:
    with gr.Tab("Train & Visualize"):
        # Add sliders for BETA and MAX_ITER
        beta_slider = gr.Slider(minimum=0.0, maximum=1.0, step=0.1, value=0.5, label="BETA")
        max_iter_slider = gr.Slider(minimum=1, maximum=1000, step=10, value=500, label="MAX_ITER")

        # Button to start training
        gr.Button("Train & Visualize").click(
            train_and_visualize, 
            inputs=[beta_slider, max_iter_slider], 
            outputs=[gr.Image(), gr.DataFrame()]
        )

        # Display training metrics
        gr.Image("/tmp/training_metrics.png", label="Training Metrics Visualization")
    
    with gr.Tab("Predict Sentence"):
        # Predict part
        sentence_input = gr.Textbox(label="Input Sentence")
        prediction_output = gr.Textbox(label="Prediction Output")
        
        # Prediction button
        gr.Button("Predict").click(predict_ner, sentence_input, prediction_output)
    
    # Launch interface
    demo.launch()

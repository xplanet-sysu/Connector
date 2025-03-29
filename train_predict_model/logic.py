import pandas as pd
from train_predict_model.classifier import load_data, train_model
import joblib
import numpy as np


def handle_training_or_prediction(raw_tx_file, label_file, do_train: bool = True):
    try:
        raw_tx = pd.read_csv(raw_tx_file)
        label_df = pd.read_csv(label_file)
    except Exception as e:
        return {"error": f"CSV 读取失败: {e}"}

    if do_train:
        feature, label,_ = load_data(raw_tx, label_df)
        train_model(feature, label)
        return {
            "message": "训练完成",
            "samples": len(feature)
        }
    else:

        print("开始预测...")

        # 模拟标签列只保留 tx hash，因为不需要真实 label
        label_df = label_df[["srcTxhash","function"]]
        label_df["label"] = 0

        feature, _ ,txlist= load_data(raw_tx, label_df)

        # 加载模型
        try:
            clf = joblib.load("model.pkl")
        except Exception as e:
            return {"error": f"模型加载失败: {e}"}

        predictions = clf.predict(feature)
        print(len(predictions))
        probabilities = clf.predict_proba(feature).tolist()

        results = []
        # print("label_df 行数：", len(label_df))
        # print("feature shape：", feature.shape)
        # print("prediction shape：", predictions.shape)
        for i, tx_hash in enumerate(txlist):
            results.append({
                "srcTxhash": tx_hash,
                "prediction": int(predictions[i]),
                "probability": probabilities[i]
            })

        return {
            "message": "预测完成",
            "results": results
        }

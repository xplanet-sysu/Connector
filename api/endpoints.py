from fastapi import APIRouter, UploadFile, File, Form,Query,HTTPException
from utils import ChainEnum, BridgeEnum
from config import SpiderNetEnum
from train_predict_model.logic import handle_training_or_prediction
import os
import shutil
from param.searchwithdraw import find_withdraw
from web3 import Web3
router = APIRouter()
import json
from datetime import datetime

@router.post("/train-or-predict",summary="训练存款交易识别的分类器或者对交易进行识别，最后保存json文件", description="""
**功能**：使用上传的交易数据和标签数据进行训练或预测。

**输入说明**：
- `raw_tx` (`File`)：CSV 格式的原始交易数据文件，需包含 `hash`, `address_from`, `address_to`字段。
- `label_data` (`File`)：CSV 格式的标签数据文件，训练模式需包含 `srcTxhash`,`function`, `label`，预测模式仅需包含 `srcTxhash`, `function`。
- `label` (`Form`)：是否为训练模式（`true` 为训练，`false` 为预测）。

**输出说明**：
- 训练模式(label==True)：
```json
{
  "message": "训练完成",
  "samples": 5146
}
```
- 预测模式(label==False)：
```json
{
  "message": "预测完成并已保存",
  "saved_file": "./data/prediction/pred_20250329_120101.json",
  "count": 5146
}
```
""")
async def train_or_predict_endpoint(
    raw_tx: UploadFile = File(..., description="原始交易数据 CSV 文件，需包含 srcTxhash 和 function 字段"),
    label_data: UploadFile = File(..., description="标签数据 CSV 文件，需包含 srcTxhash 和 label 字段"),
    label: bool = Form(..., description="是否执行训练：True 表示训练，False 表示仅预测")
):
    result = handle_training_or_prediction(raw_tx.file, label_data.file, do_train=label)

    if not label and "results" in result:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_dir = "./data/prediction"
        os.makedirs(save_dir, exist_ok=True)
        pred_path = os.path.join(save_dir, f"pred_{timestamp}.json")

        with open(pred_path, "w") as f:
            json.dump(result["results"], f, indent=2)

        # 返回保存路径
        return {
            "message": "预测完成并已保存",
            "saved_file": pred_path,
            "count": len(result["results"])
        }

    return result

def is_withdraw_file_valid(path: str) -> bool:
    return path.startswith("./data/withdraw/") and os.path.basename(path).startswith("withdraw_") and path.endswith(".json")

@router.post("/process-json", summary="通过找到的处理后存款交易，进行匹配取款交易",description="""
**功能**：基于预测结果或自定义 JSON 文件，调用区块链爬虫查找目标链的提现交易并匹配对应路径。

**输入说明**：
- `json_file` (`File`)：JSON 文件，可为：
  1. `/convert-prediction` 接口生成的 `withdraw_xxx.json`；
  2. 或格式如下的自定义数组：
```List[json]
[
  {
    "event": "Deposit",
    "bridge": "CelerNetwork",
    "args": {
      "sender": "0x...",
      "receiver": "0x...",
      "amount": 10000000000000000,
      "dstChain": "Polygon",
      "srcChain": "ETH",
      "asset_s": "0x..."
    },
    "txhash": "0x...",
    "timestamp": 1638536112
  },......
]
```
输出说明：

```json
{
  "status": "ok",
  "used_file": "./data/withdraw/withdraw_20250329_120101.json",
  "matched_withdraws": [
    {"dst_txs": { ... }}
  ]
}
```
""")
async def process_json_file(
    json_file: UploadFile = File(..., description="上传的 JSON 文件，支持两种格式：1）由 /convert-prediction 接口生成的预测结果路径文件；2）包含标准 withdraw 字段结构的 JSON 数组"),
    cur_bridge: BridgeEnum = Query(default=BridgeEnum.CELER, description="跨链桥名称，例如 CELER"),
    src: ChainEnum = Query(default=ChainEnum.ETH, description="源链，例如 ETH、BSC、Polygon"),
    dst: ChainEnum = Query(default=ChainEnum.Polygon, description="目标链，例如 ETH、BSC、Polygon"),
    spider_net: SpiderNetEnum = Query(default=SpiderNetEnum.Polygon, description="爬取网络，例如 polygon"),
    interval: int = Query(default=140*60, description="时间间隔（单位：秒），用于判断目标链 block 范围")
):
    # 判断上传文件的 filename 是否已经是合法 withdraw 路径
    if is_withdraw_file_valid(json_file.filename):
        save_path = json_file.filename
    else:
        # 否则保存
        save_dir = "./data/withdraw"
        os.makedirs(save_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"withdraw_{timestamp}.json"
        save_path = os.path.join(save_dir, filename)

        try:
            with open(save_path, "wb") as buffer:
                shutil.copyfileobj(json_file.file, buffer)
        except Exception as e:
            return {"status": "error", "message": f"文件保存失败: {e}"}

    # 调用处理逻辑
    match_withdraw = find_withdraw(
        cur_bridge=cur_bridge,
        filename=os.path.basename(save_path),  # find_withdraw只要文件名
        src=src,
        dst=dst,
        spider_net=spider_net,
        interval=interval
    )

    return {
        "status": "ok",
        "used_file": save_path,
        "matched_withdraws": match_withdraw
    }


PUBLIC_RPC = {
    "eth": "https://ethereum.publicnode.com",
    "bsc": "https://bsc-dataseed1.binance.org",
    "polygon": "https://polygon-rpc.com"
}

def get_tx_and_timestamp(txhash: str, chain: ChainEnum):
    try:
        w3 = Web3(Web3.HTTPProvider(PUBLIC_RPC[chain.value.lower()]))
        tx = w3.eth.get_transaction(txhash)
        block = w3.eth.get_block(tx.blockNumber)
        return tx, block["timestamp"]
    except Exception as e:
        print(f"查询失败 {txhash}：{e}")
        return None, 0

@router.post("/convert-prediction", summary="处理从训练预测模块得到的输出(json)使得能匹配上交易匹配模块的输入(json)",description="""
**功能**：根据预测结果（预测为 1 的交易），调用链上 RPC 获取完整交易信息并构建标准提现 JSON。

**输入说明**：
- `prediction_file` (`File`)：预测结果 JSON 文件，格式如下：
```json
[
  {
    "srcTxhash": "0x...",
    "prediction": 1,
    "probability": [0.1, 0.9]
  },
  ...
]
```
输出说明：
```json
{
  "message": "成功转换 132 个交易",
  "withdraw_file": "./data/withdraw/withdraw_20250329_120130.json"
}
```
""")
async def convert_prediction_to_withdraw(
    prediction_file: UploadFile = File(..., description="上传由 /train-or-predict 生成的预测 JSON 文件"),
    chain: ChainEnum = Query(..., description="预测交易所在链：ETH / BSC / Polygon"),
    dst_chain: str = Query(..., description="目标链名称，用于标注，如 BNB / Polygon"),
    bridge: str = Query(default="CelerNetwork", description="跨链桥名称，例如 CelerNetwork、Multichain 等")
):
    try:
        preds = json.load(prediction_file.file)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"预测文件解析失败: {e}")

    target_hashes = [item["srcTxhash"] for item in preds if item["prediction"] == 1]
    print(f"🟡 预测为 1 的交易数量: {len(target_hashes)}")
    result = []
    for txhash in target_hashes:
        tx, timestamp = get_tx_and_timestamp(txhash, chain)
        if tx is None:
            continue
        try:
            item = {
                "event": "Deposit",
                "bridge": bridge,
                "args": {
                    "sender": tx["from"],
                    "receiver": tx["to"],
                    "amount": tx["value"],
                    "dstChain": dst_chain,
                    "srcChain": chain.value.upper(),
                    "asset_s": tx["input"][:42] if tx["input"] and tx["input"] != "0x" else "0x0000000000000000000000000000000000000000"
                },
                "txhash": txhash,
                "timestamp": timestamp
            }
            result.append(item)
        except Exception as e:
            print(f"构造 withdraw 条目失败 {txhash}：{e}")
    if not result:
        raise HTTPException(status_code=500, detail="没有成功处理任何交易")
    os.makedirs("./data/withdraw", exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = f"./data/withdraw/withdraw_{ts}.json"
    with open(save_path, "w") as f:
        json.dump(result, f, indent=2)
    return {
        "message": f"成功转换 {len(result)} 个交易",
        "withdraw_file": save_path
    }

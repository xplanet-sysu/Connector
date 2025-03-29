import pandas as pd

from tqdm import tqdm

from config import SpiderNetEnum
from find_dst.dst_chain import WithdrawLocator
from extractor import BridgeSpider
from utils import ChainEnum, BridgeEnum
from utils.loader import load_withdraw_dataset
from utils.str import hash_str
from utils.dict import expand_dict
from utils.block import get_block_number_by_timestamp
from config import Config
from fastapi import Query

import math

def clean_for_json(obj):
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    elif isinstance(obj, dict):
        return {k: clean_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [clean_for_json(i) for i in obj]
    else:
        return obj

def find_withdraw(
        cur_bridge: BridgeEnum,
        filename: str,
        src: ChainEnum = ChainEnum.ETH,
        dst: ChainEnum = ChainEnum.Polygon,
        spider_net: SpiderNetEnum = SpiderNetEnum.Polygon,
        interval: int = 140 * 60
    ):
    space_size = 0
    res_report = {
        'src': f"{src.value}",
        'dst': f"{dst.value}",
        'bridge': f"{cur_bridge.value}",
        'timeInterval': interval,
    }

    match_withdraw = []

    for bridge in BridgeEnum:
        if bridge.value != cur_bridge.value:
            continue

        sample = load_withdraw_dataset(
            src_chain=src,
            dst_chain=dst,
            bridge=bridge,
            filename=filename
        )

        for item in tqdm(sample):
            timestamp = item['timestamp']
            try:
                start_dst_blk = get_block_number_by_timestamp(dst.value, timestamp)
                end_dst_blk = get_block_number_by_timestamp(dst.value, timestamp + interval)

                dst_txs = BridgeSpider(
                    net=dst.value,
                    spider_net=spider_net.value,
                    addresses=[item['args']['receiver']],
                    start_blk=start_dst_blk,
                    end_blk=end_dst_blk
                ).search_for_bridge()
                space_size += len(dst_txs)
            except Exception:
                print("no have dst_txs")
                dst_txs = pd.DataFrame()

            src_txs = pd.DataFrame([expand_dict(item, '.')])

            res = WithdrawLocator(
                src_txs=src_txs,
                dst_txs=dst_txs
            ).search_withdraw()

            match_withdraw.append({"dst_txs": res[0]})

    return clean_for_json(match_withdraw)


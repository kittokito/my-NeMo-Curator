import pandas as pd
from nemo_curator.datasets import DocumentDataset
from nemo_curator.utils.distributed_utils import get_client
from nemo_curator import Sequential, AddId, ExactDuplicates, FuzzyDuplicates, FuzzyDuplicatesConfig, ToBackend
import logging
import traceback

# ログ設定
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def main():
    try:
        # Dask クライアント起動（メモリ設定を調整）
        client = get_client(
            cluster_type="gpu",
            n_workers=1,
            device_memory_limit="35GB",
            rmm_pool_size="30GB"
        )
        
        # データセットの読み込みとID追加
        logger.info("データセットを読み込み中...")
        ds = DocumentDataset.read_json("./data/raw/plc_通常_02.jsonl")
        
        # データの前処理：null値の確認と除去
        logger.info("データの前処理を実行中...")
        # null値を含む行を除去
        ds.df = ds.df.dropna(subset=['text'])
        
        # 空文字列の行も除去
        ds.df = ds.df[ds.df['text'].str.strip() != '']
        
        # ID追加
        ds = AddId(id_field="id")(ds)
        
        # データセットをcuDFバックエンドに変換
        ds = ToBackend(backend="cudf")(ds)
        
        # データ数を確認
        logger.info(f"処理対象のドキュメント数: {len(ds.df)}")
        
        # ========================================
        # 2. FuzzyDuplicates（ファジー重複除去）
        # ========================================
        logger.info("Stage 2: ファジー重複除去を実行中...")
        
        # false_positive_checkをTrueに設定
        fuzzy_cfg = FuzzyDuplicatesConfig(
            cache_dir="./fuzzy_dedup_cache_debug",
            id_field="id",
            text_field="text",
            char_ngrams=10,
            num_buckets=30,  
            hashes_per_bucket=8,
            jaccard_threshold=0.92,
            perform_removal=True,
            false_positive_check=True,  # ここをTrueに設定
            num_anchors=2
        )
        fuzzy = FuzzyDuplicates(config=fuzzy_cfg, perform_removal=True)
        
        # パイプラインの実行
        logger.info("ファジー重複除去を実行中...")
        ds_after_fuzzy = fuzzy(ds)
        logger.info(f"ファジー重複除去後のドキュメント数: {len(ds_after_fuzzy.df)}")
        
        # 結果の保存
        logger.info("結果を保存中...")
        ds_after_fuzzy.to_json("./data/deduped/debug_fuzzy_dedup.jsonl", write_to_filename=False)
        
        logger.info("処理が完了しました！")
        
    except Exception as e:
        logger.error(f"エラーが発生しました: {str(e)}")
        logger.error(f"エラーの詳細:\n{traceback.format_exc()}")
        raise
    finally:
        # クライアントのクローズ
        if 'client' in locals():
            client.close()

if __name__ == '__main__':
    main()

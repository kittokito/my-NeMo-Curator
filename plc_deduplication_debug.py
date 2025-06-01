import pandas as pd
from nemo_curator.datasets import DocumentDataset
from nemo_curator.utils.distributed_utils import get_client
from nemo_curator import AddId, ToBackend
import logging

# ログ設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_data_quality():
    """データの品質をチェックする関数"""
    # Daskクライアントを起動
    client = get_client(cluster_type="gpu", n_workers=1)
    
    try:
        # データセットの読み込み
        logger.info("データセットを読み込み中...")
        ds = DocumentDataset.read_json("plc_通常_02.jsonl")
        
        # 基本統計情報
        logger.info(f"総ドキュメント数: {len(ds.df)}")
    
        # null値のチェック
        null_counts = ds.df.isnull().sum().compute()
        logger.info(f"NULL値の数:\n{null_counts}")
        
        # text列の詳細チェック
        if 'text' in ds.df.columns:
            # 空文字列のチェック
            empty_texts = (ds.df['text'].str.strip() == '').sum().compute()
            logger.info(f"空のテキスト数: {empty_texts}")
            
            # テキスト長の統計
            text_lengths = ds.df['text'].str.len()
            logger.info(f"テキスト長の統計:")
            logger.info(f"  最小: {text_lengths.min().compute()}")
            logger.info(f"  最大: {text_lengths.max().compute()}")
            logger.info(f"  平均: {text_lengths.mean().compute():.2f}")
            logger.info(f"  中央値: {text_lengths.median().compute():.2f}")
            
            # 短すぎるテキストの確認
            very_short = (text_lengths < 10).sum().compute()
            logger.info(f"10文字未満のテキスト数: {very_short}")
    
        # データ型の確認
        logger.info(f"\nデータ型:\n{ds.df.dtypes}")
        
        # サンプルデータの表示
        logger.info("\nサンプルデータ（最初の3件）:")
        sample_df = ds.df.head(3).compute()
        for i, row in sample_df.iterrows():
            logger.info(f"--- Document {i} ---")
            for col, val in row.items():
                if col == 'text' and len(str(val)) > 100:
                    logger.info(f"{col}: {str(val)[:100]}...")
                else:
                    logger.info(f"{col}: {val}")
    
    finally:
        # クライアントをクローズ
        client.close()

def test_fuzzy_dedup_minimal():
    """最小限の設定でFuzzyDedupをテストする関数"""
    from nemo_curator import FuzzyDuplicates, FuzzyDuplicatesConfig
    
    # Dask クライアント起動
    client = get_client(
        cluster_type="gpu",
        n_workers=1,
        device_memory_limit="8GB",
        rmm_pool_size="7GB"
    )
    
    try:
        # データセットの読み込み
        logger.info("テスト用データセットを読み込み中...")
        ds = DocumentDataset.read_json("plc_通常_02.jsonl")
        
        # データの前処理
        ds.df = ds.df.dropna(subset=['text'])
        ds.df = ds.df[ds.df['text'].str.strip() != '']
        
        # ID追加とバックエンド変換
        ds = AddId(id_field="id")(ds)
        ds = ToBackend(backend="cudf")(ds)
        
        logger.info(f"処理対象のドキュメント数: {len(ds.df)}")
        
        # 最小限の設定でFuzzyDedupを実行
        fuzzy_cfg = FuzzyDuplicatesConfig(
            cache_dir="./fuzzy_dedup_test_cache",
            id_field="id",
            text_field="text",
            char_ngrams=5,
            num_buckets=5,  # 最小限に減らす
            hashes_per_bucket=5,  # 最小限に減らす
            jaccard_threshold=0.80,
            perform_removal=True,
            false_positive_check=False  # 無効化
        )
        
        fuzzy = FuzzyDuplicates(config=fuzzy_cfg, perform_removal=True)
        
        # 実行
        logger.info("FuzzyDedupを実行中...")
        result = fuzzy(ds)
        
        logger.info(f"FuzzyDedup後のドキュメント数: {len(result.df)}")
        logger.info("テストが成功しました！")
        
    except Exception as e:
        logger.error(f"エラーが発生しました: {e}")
        import traceback
        traceback.print_exc()
    finally:
        client.close()

if __name__ == '__main__':
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == 'test':
        test_fuzzy_dedup_minimal()
    else:
        check_data_quality()

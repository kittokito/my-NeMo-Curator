import pandas as pd
from nemo_curator.datasets import DocumentDataset
from nemo_curator.utils.distributed_utils import get_client
from nemo_curator import Sequential, AddId, ExactDuplicates, FuzzyDuplicates, FuzzyDuplicatesConfig, SemDedup, SemDedupConfig, ToBackend
import logging

# ログ設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    # Dask クライアント起動（メモリ設定を調整）
    client = get_client(
        cluster_type="gpu",
        n_workers=1,
        device_memory_limit="35GB",  # GPUメモリ制限（40GBの約87.5%）
        rmm_pool_size="30GB"  # RMMプールサイズ（40GBの75%）
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
    # 1. ExactDuplicates（完全一致重複除去）
    # ========================================
    logger.info("Stage 1: 完全一致重複除去を実行中...")
    exact = ExactDuplicates(
        id_field="id",
        text_field="text",
        hash_method="md5",
        perform_removal=True,
        cache_dir="./exact_dedup_cache"
    )
    
    # ========================================
    # 2. FuzzyDuplicates（ファジー重複除去）
    # ========================================
    logger.info("Stage 2: ファジー重複除去を実行中...")
    
    # メモリ問題を回避するため、設定を調整
    fuzzy_cfg = FuzzyDuplicatesConfig(
        cache_dir="./fuzzy_dedup_cache",
        id_field="id",
        text_field="text",
        char_ngrams=12,
        num_buckets=50,  
        hashes_per_bucket=5,  # ハッシュ数を減らす（13→10）
        jaccard_threshold=0.92,
        perform_removal=True,
        false_positive_check=True,
        num_anchors=3
    )
    fuzzy = FuzzyDuplicates(config=fuzzy_cfg, perform_removal=True)
    
    # ========================================
    # 3. SemDedup（セマンティック重複除去）
    # ========================================
    logger.info("Stage 3: セマンティック重複除去を実行中...")
    
    # メモリ効率を考慮した設定
    sem_cfg = SemDedupConfig(
        cache_dir="./sem_cache",
        embedding_model_name_or_path="jinaai/jina-embeddings-v2-base-code",
        embedding_batch_size=512,  # バッチサイズを小さく（1024→512）
        eps_to_extract=0.08,
        n_clusters=2000,  # クラスタ数を減らす（1000→100）
        max_iter=100,  # 反復回数を減らす（100→50）
        batched_cosine_similarity=1024,
        which_to_keep="hard"  # バッチサイズを調整
    )
    semantic = SemDedup(config=sem_cfg, input_column="text", id_column="id", perform_removal=True)
    
    # パイプラインの構築と実行
    try:
        # 各ステージを個別に実行（デバッグしやすくするため）
        logger.info("パイプラインを実行中...")
        
        # Stage 1: Exact deduplication
        ds_after_exact = exact(ds)
        logger.info(f"完全一致重複除去後のドキュメント数: {len(ds_after_exact.df)}")
        
        # Stage 2: Fuzzy deduplication
        ds_after_fuzzy = fuzzy(ds_after_exact)
        logger.info(f"ファジー重複除去後のドキュメント数: {len(ds_after_fuzzy.df)}")
        
        # Stage 3: Semantic deduplication
        clean_ds = semantic(ds_after_fuzzy)
        logger.info(f"セマンティック重複除去後のドキュメント数: {len(clean_ds.df)}")
        
    except Exception as e:
        logger.error(f"パイプライン実行中にエラーが発生しました: {e}")
        # エラー時は中間結果を保存
        if 'ds_after_exact' in locals():
            ds_after_exact.to_json("intermediate_exact_dedup/", write_to_filename=True)
        if 'ds_after_fuzzy' in locals():
            ds_after_fuzzy.to_json("intermediate_fuzzy_dedup/", write_to_filename=True)
        raise
    
    # 結果の保存
    logger.info("結果を保存中...")
    clean_ds.to_json("./data/deduped/deduped-01_plc_通常_02.jsonl", write_to_filename=False)
    
    logger.info("処理が完了しました！")
    
    # クライアントのクローズ
    client.close()

if __name__ == '__main__':
    main()

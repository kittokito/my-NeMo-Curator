import pandas as pd
from nemo_curator.datasets import DocumentDataset
from nemo_curator.utils.distributed_utils import get_client
from nemo_curator import Sequential, AddId, ExactDuplicates, FuzzyDuplicates, FuzzyDuplicatesConfig, SemDedup, SemDedupConfig, ToBackend

def main():
    # Dask クライアント起動
    client = get_client()
    
    # データセットの読み込みとID追加
    ds = DocumentDataset.read_json("plc_通常_02.jsonl")
    ds = AddId(id_field="id")(ds)
    
    # データセットをcuDFバックエンドに変換（FuzzyDuplicatesとSemDedupに必要）
    ds = ToBackend(backend="cudf")(ds)
    
    # ========================================
    # 1. ExactDuplicates（完全一致重複除去）
    # ========================================
    # パラメータ説明：
    # - id_field: ドキュメントの一意識別子を含むカラム名（デフォルト: "id"）
    # - text_field: 重複チェック対象のテキストを含むカラム名（デフォルト: "text"）
    # - hash_method: ハッシュアルゴリズム（現在は"md5"のみサポート）
    # - perform_removal: True=重複を除去したデータセットを返す、False=重複IDのリストのみを返す
    # - cache_dir: 中間結果を保存するディレクトリ（パフォーマンス向上のため推奨）
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
    # MinHashとLSH（Locality Sensitive Hashing）を使用した近似重複検出
    #
    # FuzzyDuplicatesConfigパラメータ説明：
    # 【基本設定】
    # - cache_dir: 中間結果保存用ディレクトリ（必須）
    # - id_field: ドキュメントID列名
    # - text_field: テキスト列名
    # - perform_removal: 重複除去実行フラグ
    #
    # 【MinHash + LSH設定】
    # - char_ngrams: 文字n-gramのサイズ（デフォルト: 24）
    #   * 小さい値（例：5）は短い類似部分も検出
    #   * 大きい値は長い類似部分のみ検出
    # - num_buckets: LSHのバンド数（デフォルト: 20）
    #   * 多いほど精度向上、計算コスト増加
    # - hashes_per_bucket: 各バンドのハッシュ数（デフォルト: 13）
    #   * num_hashes = num_buckets × hashes_per_bucket
    #
    # 【偽陽性チェック設定】
    # - false_positive_check: 偽陽性チェック実行フラグ（計算コストが高いが精度向上）
    # - jaccard_threshold: Jaccard類似度閾値（0-1）
    #   * 0.85 = 85%以上の類似度で重複とみなす
    # - num_anchors: アンカードキュメント数（デフォルト: 2）
    fuzzy_cfg = FuzzyDuplicatesConfig(
        cache_dir="./fuzzy_dedup_cache",
        id_field="id",
        text_field="text",
        char_ngrams=5,
        num_buckets=20,
        hashes_per_bucket=13,
        jaccard_threshold=0.80,
        perform_removal=True,
        false_positive_check=True
    )
    fuzzy = FuzzyDuplicates(config=fuzzy_cfg, perform_removal=True)
    
    # ========================================
    # 3. SemDedup（セマンティック重複除去）
    # ========================================
    # 埋め込みベースの意味的類似度による重複検出
    #
    # SemDedupConfigパラメータ説明：
    # 【基本設定】
    # - cache_dir: 中間結果保存用ディレクトリ（必須）
    #
    # 【埋め込み設定】
    # - embedding_model_name_or_path: 使用する埋め込みモデル
    #   * 英語向け: "sentence-transformers/all-mpnet-base-v2"
    #   * 日本語向け: "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
    # - embedding_batch_size: 埋め込み生成のバッチサイズ（デフォルト: 128）
    # - embedding_max_mem_gb: 最大メモリ使用量（GB）
    # - embedding_pooling_strategy: プーリング戦略
    #   * "mean_pooling": 平均プーリング（デフォルト）
    #   * "last_token": 最後のトークン
    #
    # 【クラスタリング設定】
    # - n_clusters: クラスタ数（デフォルト: 1000）
    #   * データ量に応じて調整（100は小規模データ向け）
    # - max_iter: K-meansの最大反復回数（デフォルト: 100）
    # - random_state: 乱数シード（デフォルト: 1234）
    #
    # 【重複判定設定】
    # - eps_to_extract: 重複抽出のイプシロン値（0-1）
    #   * 小さいほど厳密（0.08 ≈ 92%類似度）
    #   * 大きいほど緩い判定
    # - sim_metric: 類似度メトリック
    #   * "cosine": コサイン類似度（デフォルト）
    #   * "l2": L2距離
    # - which_to_keep: 保持する文書の選択方法
    #   * "hard": 外れ値を保持（多様性重視）
    #   * "easy": 代表的な文書を保持
    #   * "random": ランダム
    # - batched_cosine_similarity: バッチ処理サイズ
    #   * 1024: バッチサイズ（メモリ効率的）
    #   * False/0: バッチ処理なし（高速だがメモリ使用大）
    sem_cfg = SemDedupConfig(
        cache_dir="./sem_cache",
        embedding_model_name_or_path="sentence-transformers/all-mpnet-base-v2",
        embedding_batch_size=1024,
        eps_to_extract=0.08,  # 類似度閾値に相当（0.92の類似度は約0.08のイプシロン）
        n_clusters=1000,  # クラスタ数を調整
        max_iter=100
    )
    # SemDedupクラスパラメータ：
    # - config: SemDedupConfigオブジェクト
    # - input_column: 入力テキスト列名
    # - id_column: ID列名
    # - perform_removal: 重複除去実行フラグ
    semantic = SemDedup(config=sem_cfg, input_column="text", id_column="id", perform_removal=True)
    
    # パイプラインの構築と実行
    pipeline = Sequential([exact, fuzzy, semantic])
    clean_ds = pipeline(ds)
    
    # 結果の保存
    clean_ds.to_json("deduped_1_plc_通常_02/", write_to_filename=True)
    
    # クライアントのクローズ
    client.close()

if __name__ == '__main__':
    main()

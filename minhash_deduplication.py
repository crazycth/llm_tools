from __future__ import annotations

import gc
import hashlib
import logging
import multiprocess as mp
import os
import random
import re
import struct
import time
import warnings
import json
from collections import defaultdict
from itertools import tee
from pathlib import Path
from typing import Any
from typing import Dict
from typing import Iterable
from typing import List
from typing import Tuple

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=FutureWarning)
    import datasets
    import numpy as np
    import typer
    from datasets import load_dataset
    from scipy.integrate import quad as integrate
    from tqdm import tqdm

# support chinese
import jieba
def _jieba_tokenize(x): return [w for w in jieba.lcut(x) if w.strip()]

def remove_punctuation_and_numbers(text):
    # 定义一个正则表达式，匹配非中文字符
    # 这里的正则表达式 [\u4e00-\u9fff] 匹配所有中文字符
    pattern = re.compile(r'[^\u4e00-\u9fff]')

    # 使用正则表达式的 sub 方法替换掉非中文字符
    clean_text = pattern.sub('', text)

    return clean_text


SEED = 42
NON_ALPHA = re.compile("[^A-Za-z_0-9]")
RNG = np.random.RandomState(SEED)
MAX_HASH = np.uint64((1 << 32) - 1)
MERSENNE_PRIME = np.uint64((1 << 61) - 1)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
datasets.logging.set_verbosity_error()

def ngrams(sequence: List[str], n: int, min_ngram_size: int) -> Iterable:
    """
    Directly taken from nltk package to avoid dependency.

    Parameters
    ----------
    sequence : list
        The sequence of items to be n-grammed.
    n : int
        The order of the n-grams to be extracted.
    min_ngram_size : int
        The minimum size of n-grams.

    Returns
    -------
    Iterable
        The n-grams generated from the sequence.
    """
    if len(sequence) < min_ngram_size:
        return []
    iterables = tee(sequence, n)
    for i, sub_iterable in enumerate(iterables):
        for _ in range(i):
            next(sub_iterable, None)
    return zip(*iterables)


def sha1_hash32(data):
    """
    Directly taken from datasketch package to avoid dependency.

    Parameters
    ----------
    data : bytes

    Returns
    -------
    int
    """
    return struct.unpack("<I", hashlib.sha1(data).digest()[:4])[0]


def embed_func(
    content: str,
    idx: int,
    *,
    num_perm: int,
    ngram_size: int,
    hashranges: List[Tuple[int, int]],
    permutations: np.ndarray,
    min_ngram_size: int = 5,
    language: str = 'en'
) -> Dict[str, Any]:
    """
    Combined with some datasketch code to better parallelize computation.

    Parameters
    ----------
    content : str
        The content to be embedded.
    idx : int
        The index of the content.
    num_perm : int
        The number of permutations.
    ngram_size : int
        The size of n-grams.
    hashranges : List[Tuple[int, int]]
        The ranges of hash values.
    permutations : np.ndarray
        The permutations for the minhash.
    min_ngram_size : int
        The minimum size of n-grams.

    Returns
    -------
    Dict[str, Any]
        The hash values in each range and the index.
    """
    hashvalues = np.ones(num_perm, dtype=np.uint64) * MAX_HASH
    tokenize_result = _jieba_tokenize(content)
    # print(f"[DEBUG] tokenize result: {tokenize_result}")
    if language == 'en':
        tokens = {" ".join(t) for t in ngrams(NON_ALPHA.split(content), ngram_size, min_ngram_size)}
    else:
        tokens = {" ".join(t) for t in ngrams(_jieba_tokenize(content), ngram_size, min_ngram_size)}

    hv = np.array([sha1_hash32(token.encode("utf-8")) for token in tokens], dtype=np.uint64)  # noqa: E501
    a, b = permutations
    phv = np.bitwise_and(((hv * np.tile(a, (len(hv), 1)).T).T + b) % MERSENNE_PRIME, MAX_HASH)  # noqa: E501
    hashvalues = np.vstack([phv, hashvalues]).min(axis=0)
    Hs = [bytes(hashvalues[start:end].byteswap().data) for start, end in hashranges]
    return {"__signatures__": Hs, "__id__": idx}


def optimal_param(
    threshold: float,
    num_perm: int,
    false_positive_weight: float = 0.5,
    false_negative_weight: float = 0.5,
):
    """
    Compute the optimal `MinHashLSH` parameter that minimizes the weighted sum
    of probabilities of false positive and false negative, taken from datasketch.

    Parameters
    ----------
    threshold : float
        The threshold for similarity.
    num_perm : int
        The number of permutations.
    false_positive_weight : float
        The weight of false positive.
    false_negative_weight : float
        The weight of false negative.

    Returns
    -------
    Tuple[int, int]
        The optimal `b` and `r` parameters.
        The number of bands, and the number of rows per band respectively.
    """

    def false_positive_probability(threshold: float, b: int, r: int):
        """Source: `datasketch.lsh`"""

        def proba(s):
            return 1 - (1 - s ** float(r)) ** float(b)

        a, _ = integrate(proba, 0.0, threshold)
        return a

    def false_negative_probability(threshold: float, b: int, r: int):
        """Source: `datasketch.lsh`"""

        def proba(s):
            return 1 - (1 - (1 - s ** float(r)) ** float(b))

        a, _ = integrate(proba, threshold, 1.0)
        return a

    min_error = float("inf")
    opt = (0, 0)
    for b in range(1, num_perm + 1):
        max_r = int(num_perm / b)
        for r in range(1, max_r + 1):
            fp = false_positive_probability(threshold, b, r)
            fn = false_negative_probability(threshold, b, r)
            error = fp * false_positive_weight + fn * false_negative_weight
            if error < min_error:
                min_error = error
                opt = (b, r)
    return opt


class UnionFind:
    def __init__(self):
        self.parent: Dict[int, int] = {}

    def find(self, x):
        if x not in self.parent:
            self.parent[x] = x
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x, y):
        px = self.find(x)
        py = self.find(y)
        self.parent[px] = self.parent[py] = min(px, py)


if __name__ == "__main__":
    def run(
        dataset: str = typer.Option("codeparrot/codeparrot-clean-valid", help="The dataset to use"),  # noqa: E501
        config: str = typer.Option("default", help="Dataset config"),
        split: str = typer.Option("train", help="Dataset split"),
        data_dir: str = typer.Option(None, help="Dataset data directory"),
        revision: str = typer.Option("main", help="Dataset revision"),
        column: str = typer.Option("content", help="Dataset column"),
        cache_dir: str = typer.Option(".cache", help="Cache directory"),
        ngram_size: int = typer.Option(5, help="The ngram size to use for MinHash"),
        num_perm: int = typer.Option(256, help="Number of permutations"),
        threshold: float = typer.Option(0.5, help="Minhash threshold"),
        min_ngram_size: int = typer.Option(5, help="Shorter documents will be removed"),
        output: str = typer.Option(None, help="Store the deduplicated dataset"),
        watch: bool = typer.Option(None, help="Output filter example"),
        language:str = typer.Option("en", help="en or zh")
    ):
        global uf
        OUTPUT_BASE = Path(output or "output")
        OUTPUT_BASE.mkdir(exist_ok=True, parents=True)
        output = OUTPUT_BASE / "deduplicated"

        logging.basicConfig(level=logging.INFO)

        time_measures = {}
        start_time = time.time()

        B, R = optimal_param(threshold, num_perm)
        HASH_RANGES = [(i * R, (i + 1) * R) for i in range(B)]
        HASH_TABLES = [defaultdict(set) for _ in range(B)]


        """
        from datasets import load_dataset

        # 定义数据集的特征结构
        features = {
            'text': Value('string'),
            'label': ClassLabel(names=['negative', 'positive'])
        }

        # 加载数据集
        dataset = load_dataset('json', data_files='your_file.json', features=features, split='train')

        """
        time_measures["load_dataset"] = time.time()
        # ds = load_dataset(
        #     dataset,
        #     config,
        #     data_dir=data_dir,
        #     split=split,
        #     use_auth_token=True,
        #     cache_dir=cache_dir,
        #     revision=revision,
        #     num_proc = os.cpu_count(),
        # )

        ds = load_dataset('json',data_files=dataset,split="train")
        time_measures["load_dataset"] = time.time() - time_measures["load_dataset"]

        if language == 'zh':
            # ds = ds.map(
            #     lambda example: {"feature": remove_punctuation_and_numbers(example["feature"])},
            #     num_proc=os.cpu_count(),
            #     new_fingerprint=str(random.getrandbits(128)),
            #     desc="Applying remove_punctuation_and_numbers..."
            # )
            pass

        DATA_SIZE = len(ds)
        PERMUTATIONS = np.array(
            [
                (
                    RNG.randint(1, MERSENNE_PRIME, dtype=np.uint64),
                    RNG.randint(0, MERSENNE_PRIME, dtype=np.uint64),
                )
                for _ in range(num_perm)
            ],
            dtype=np.uint64,
        ).T

        time_measures["minhash"] = time.time()
        embeded = ds.map(
            function=embed_func,
            fn_kwargs={
                "num_perm": num_perm,
                "hashranges": HASH_RANGES,
                "ngram_size": ngram_size,
                "permutations": PERMUTATIONS,
                "min_ngram_size": min_ngram_size,
                "language": language
            },
            input_columns=[column],
            remove_columns=ds.column_names,
            num_proc=os.cpu_count(),
            with_indices=True,
            desc="Fingerprinting..."
        )
        time_measures["minhash"] = time.time() - time_measures["minhash"]


        time_measures["clustering"] = time.time()
        batch_size: int = 10000

        for i in tqdm(
            range(0, len(embeded), batch_size), dynamic_ncols=True, desc="Iterating Minhashes..."
        ):
            batch = embeded[i: i + batch_size]
            for key, Hs in zip(batch["__id__"], batch["__signatures__"]):
                for H, hashtable in zip(Hs, HASH_TABLES):
                    hashtable[H].add(key)
            
            for table in tqdm(HASH_TABLES, dynamic_ncols=True, desc="Clustering..."):
                for cluster in table.values():
                    if len(cluster) <= 1:
                        continue
                    idx = min(cluster)
                    for x in cluster:
                        uf.union(x, idx)
            
        time_measures["clustering"] = time.time() - time_measures["clustering"]

        time_measures["filtering"] = time.time()
        gc.freeze()
        gc.disable()

        ds = ds.map(
            function= lambda _, idx: {"__cluster__": uf.find(idx)},
            with_indices=True,
            num_proc=os.cpu_count(),
            new_fingerprint=str(random.getrandbits(128)),
            desc="Finding clusters...",
        )
        gc.enable()
        gc.collect()

        if watch:
            filtered_data = ds.filter(
                function=lambda record, idx: record["__cluster__"] != idx,
                with_indices=True,
                num_proc=os.cpu_count(),
                desc="[DEBUG] Filtering clusters..."
            )
            
            col_keep = ["__cluster_content__","__cluster__","idx","feature"]
            col_remove = [col for col in filtered_data.column_names if col not in col_keep]

            filtered_data = filtered_data.map(
                function = lambda object, idx: {"__cluster_content__": ds[object['__cluster__']]['feature']},
                with_indices=True,
                num_proc=os.cpu_count(),
                desc="[DEBUG] Finding sources",
                remove_columns=col_remove
            )

            js = []
            for item in filtered_data:
                js.append(item)

            with open("debug_filterdata.json",'w',encoding='utf-8') as file:
                json.dump(js, file, ensure_ascii=False, indent=3)

        final_data = ds.filter(
                function=lambda record, idx: record["__cluster__"] == idx,
                with_indices=True,
                num_proc=os.cpu_count(),
                desc="Filtering clusters..."
        )

        time_measures["filtering"] = time.time() - time_measures["filtering"]

        time_measures["save"] = time.time()
        final_data = final_data.remove_columns(["__cluster__"])
        final_data.save_to_disk(output)
        time_measures["save"] = time.time() - time_measures["save"]

        FINAL_DATA_SIZE = len(final_data)
        DUP_SIZE = DATA_SIZE - FINAL_DATA_SIZE
        PAD = 32

        for key, value in time_measures.items():
            logger.info(f"{key:<{PAD}}: {value:.2f} seconds")

        logger.info(f"{'Data Number (before)':<{PAD}}: {DATA_SIZE}")
        logger.info(
            f"{'Data Number (after)':<{PAD}}: {FINAL_DATA_SIZE} ({FINAL_DATA_SIZE / DATA_SIZE:.2%})"  # noqa: E501
        )
        logger.info(f"{'Duplicate Number':<{PAD}}: {DUP_SIZE} ({DUP_SIZE / DATA_SIZE:.2%})")  # noqa: E501
        logger.info(f"{'Total Time':<{PAD}}: {time.time() - start_time:.2f} seconds")
        logger.info(f"{'Deduplicated Dataset':<{PAD}}: {output}")
        logger.info("🤗 Happy Deduplicating 🤗")



    mp.set_start_method("fork", force=True)
    uf = UnionFind()
    typer.run(run)
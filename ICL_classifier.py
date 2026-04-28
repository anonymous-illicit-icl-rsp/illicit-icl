import os
import argparse
from typing import List, Optional
import logging
import random
import time

import numpy as np
import pandas as pd
import psutil
import torch
import jieba
from rank_bm25 import BM25Okapi
from retriv import DenseRetriever
from vllm import LLM, SamplingParams

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
os.environ['VLLM_ALLOW_DEPRECATED_BEAM_SEARCH'] = '1'

DEFAULT_PROMPT_PREFIX = """You will act as a content detector to decide if a given text is illicit promotion, i.e., involving the promotion of illicit goods and services.
If it is, label it as 'illicit', otherwise, label it as 'benign'.
First, you will be given a set of message examples alongwith their labels. Then, you will answer the category of the last message from the following options:
- 'benign': Content that does not contain illicit promotion
- 'illicit': Content that contain illicit promotion
Return only the name of the category, and nothing else."""


def log_resources(start_time: float) -> None:
    process = psutil.Process()
    memory_info = process.memory_info()
    cpu_memory_usage = memory_info.rss / 1024 ** 2
    gpu_memory_usage = torch.cuda.memory_allocated() / 1024 ** 2 if torch.cuda.is_available() else 0
    total_time = time.time() - start_time

    logging.info(
        "Total time: %.2f s | CPU memory used: %.2f MB | GPU memory used: %.2f MB",
        total_time,
        cpu_memory_usage,
        gpu_memory_usage,
    )


class InContextLearner:
    def __init__(
        self,
        model_name: str,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        model: Optional[LLM] = None,
    ) -> None:
        self.model_name = model_name
        self.train_df = train_df
        self.test_df = test_df
        self.model = model if model is not None else LLM(model=model_name)

    def generate_prompts(
        self,
        n_shots: int,
        retrieval: str,
        prompt_prefix: str = DEFAULT_PROMPT_PREFIX,
        query_prefix: str = 'Query',
        answer_prefix: str = 'Answer',
        random_seed: int = 42,
        has_demos_label: bool = True,
        shot_order: str = 'fixed',
        shot_label_order: Optional[List[str]] = None,
        first_shot_label: Optional[str] = None,
        last_shot_label: Optional[str] = None,
    ) -> List[str]:
        if n_shots <= 0:
            raise ValueError('n_shots must be a positive integer')
        if 'text' not in self.train_df.columns or 'label' not in self.train_df.columns:
            raise ValueError("train_df must contain 'text' and 'label' columns")
        if 'text' not in self.test_df.columns:
            raise ValueError("test_df must contain 'text' column")

        n_shots = min(n_shots, len(self.train_df))

        def create_prompt(demonstrations, query):
            demos_working = list(demonstrations)

            if shot_label_order is not None:
                demos_labels = [demo[1] for demo in demos_working]
                if not set(demos_labels).issubset(set(shot_label_order)):
                    raise ValueError('shot_label_order must contain all labels in demonstrations')
                demos_working.sort(key=lambda x: shot_label_order.index(x[1]))

            if first_shot_label is not None:
                first = [demo for demo in demos_working if demo[1] == first_shot_label]
                others = [demo for demo in demos_working if demo[1] != first_shot_label]
                if first:
                    demos_working = first + others
            elif last_shot_label is not None:
                last = [demo for demo in demos_working if demo[1] == last_shot_label]
                others = [demo for demo in demos_working if demo[1] != last_shot_label]
                if last:
                    demos_working = others + last

            if shot_order == 'random':
                rng = random.Random(random_seed)
                rng.shuffle(demos_working)

            if has_demos_label:
                demos = '==\n'.join(
                    [f"{query_prefix}: {demo}\n{answer_prefix}: {answer}\n" for demo, answer in demos_working]
                )
            else:
                demos = '==\n'.join([f"{query_prefix}: {demo}\n" for demo, _ in demos_working])

            query_str = f"{query_prefix}: {query}\n{answer_prefix}: "
            return f"{prompt_prefix}\n==\n{demos}==\n{query_str}"

        prompts = []

        if first_shot_label is not None or last_shot_label is not None:
            labels = self.train_df['label'].unique().tolist()
            per_label = max(1, n_shots // max(1, len(labels)))
            sampled_parts = []
            for label in labels:
                group = self.train_df[self.train_df['label'] == label]
                sample_size = min(per_label, len(group))
                sampled_parts.append(group.sample(sample_size, random_state=random_seed))

            sampled_demos = pd.concat(sampled_parts, ignore_index=True)
            if len(sampled_demos) > n_shots:
                sampled_demos = sampled_demos.sample(n_shots, random_state=random_seed)

            for query in self.test_df['text']:
                demonstrations = [(demo['text'], demo['label']) for _, demo in sampled_demos.iterrows()]
                prompts.append(create_prompt(demonstrations, query))

        elif retrieval == 'random':
            sampled_demos = self.train_df.sample(n_shots, random_state=random_seed)
            for query in self.test_df['text']:
                demonstrations = [(demo['text'], demo['label']) for _, demo in sampled_demos.iterrows()]
                prompts.append(create_prompt(demonstrations, query))

        elif retrieval == 'lexical':
            corpus = [row['text'] for _, row in self.train_df.iterrows()]
            tokenized_corpus = [jieba.lcut_for_search(doc) for doc in corpus]
            bm25 = BM25Okapi(tokenized_corpus)

            for query in self.test_df['text']:
                tokenized_query = jieba.lcut_for_search(query)
                doc_scores = bm25.get_scores(tokenized_query)
                inds = np.argsort(doc_scores)[::-1][:n_shots]
                sampled_demos = self.train_df.iloc[inds]
                demonstrations = [(demo['text'], demo['label']) for _, demo in sampled_demos.iterrows()]
                prompts.append(create_prompt(demonstrations, query))

        elif retrieval == 'semantic':
            collection = [{"id": idx, "text": row['text']} for idx, row in self.train_df.iterrows()]
            retriever = DenseRetriever(
                index_name=f'training-examples-{os.getpid()}',
                model='sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2',
                normalize=True,
                max_length=128,
                use_ann=False,
            ).index(collection, use_gpu=torch.cuda.is_available())

            for query in self.test_df['text']:
                retrieved = retriever.search(query=query, cutoff=n_shots)
                inds = [item['id'] for item in retrieved]
                if len(inds) < n_shots:
                    extra = self.train_df.sample(n_shots - len(inds), random_state=random_seed).index.tolist()
                    inds.extend(extra)
                sampled_demos = self.train_df.loc[inds]
                demonstrations = [(demo['text'], demo['label']) for _, demo in sampled_demos.iterrows()]
                prompts.append(create_prompt(demonstrations, query))

        else:
            raise ValueError(f'Retrieval method {retrieval} is not supported')

        return prompts

    def predict(self, prompts: List[str], label_names: List[str]) -> List[str]:
        if self.model is None:
            raise ValueError('Model is not initialized')
        if not label_names:
            raise ValueError('label_names must not be empty')

        tokenizer = self.model.get_tokenizer()
        label_tokens = [tokenizer.tokenize(label) for label in label_names]
        max_tokens = max(len(tokens) for tokens in label_tokens)
        min_tokens = min(len(tokens) for tokens in label_tokens)

        sampling_params = SamplingParams(
            temperature=0.0,
            use_beam_search=True,
            n=min(50, max(1, len(label_names) * 4)),
            top_p=1.0,
            top_k=-1,
            max_tokens=max_tokens,
            min_tokens=min_tokens,
            logprobs=10,
        )

        outputs = self.model.generate(prompts, sampling_params)
        output_text_all = [output.outputs[0].text for output in outputs]

        predicted_labels = []
        for output in output_text_all:
            normalized_output = output.strip().lower()
            matched_label = ''
            matched_index = len(normalized_output)

            for label in label_names:
                idx = normalized_output.find(label.lower())
                if idx != -1 and idx < matched_index:
                    matched_label = label
                    matched_index = idx

            predicted_labels.append(matched_label)

        return predicted_labels


def parse_args():
    parser = argparse.ArgumentParser(description='Run in-context classification with configurable, repo-safe paths.')
    parser.add_argument('--model-name', type=str, default='mistralai/Mistral-7B-Instruct-v0.2')
    parser.add_argument('--train-data', type=str, default='./Data/balanced_binary_data.csv')
    parser.add_argument('--test-data', type=str, required=True)
    parser.add_argument('--output-path', type=str, default='./Result/icl_predictions.csv')
    parser.add_argument('--retrieval', type=str, choices=['random', 'lexical', 'semantic'], default='semantic')
    parser.add_argument('--n-shots', type=int, default=128)
    parser.add_argument('--label-names', nargs='+', default=['benign', 'illicit'])
    parser.add_argument('--random-seed', type=int, default=42)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    test_df = pd.read_csv(args.test_data)
    train_df = pd.read_csv(args.train_data)

    start_time = time.time()
    icl = InContextLearner(args.model_name, train_df=train_df, test_df=test_df)

    prompts = icl.generate_prompts(
        n_shots=args.n_shots,
        retrieval=args.retrieval,
        random_seed=args.random_seed,
    )
    predictions = icl.predict(prompts, label_names=args.label_names)

    log_resources(start_time)

    test_df['prediction'] = predictions

    output_dir = os.path.dirname(args.output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    test_df.to_csv(args.output_path, index=False)
    logging.info('Prediction done! Results saved to %s', args.output_path)

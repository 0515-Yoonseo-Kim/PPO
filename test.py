from datasets import load_dataset
from typing import List, Tuple, Union, Dict, Literal, Optional
from dataclasses import dataclass, field
import evaluate
from transformers import HfArgumentParser
from transformers.pipelines import pipeline
from collections import defaultdict
from tqdm import tqdm
from transformers import AutoTokenizer
import torch
@dataclass
class ModelArguments:
    model_name_or_path: str = field(
        metadata={"help": "The model checkpoint for weights initialization."}
    )
    model_token: Optional[str] = field(
        default=None,
        metadata={"help": "The token to use as HTTP bearer authorization for remote files."}
    )

    batch_size: int = field(
        default=8,
        metadata={"help": "The batch size for evaluation."}
    )


@dataclass
class DataArguments:
    dataset_name_or_path: str = field(
        metadata={"help": "The dataset to use for training."}
    )

    data_token: Optional[str] = field(
        default=None,
        metadata={"help": "The token to use as HTTP bearer authorization for remote files."}
    )

    eval_split: str = field(
        default="validation",
        metadata={"help": "The evaluation split."}
    )

    text_column: str = field(
        default="text",
        metadata={"help": "The name of the column containing the main text."}
    )

    label_column: Optional[str] = field(
        default="label",
        metadata={"help": "The name of the column containing the labels."}
    )


@dataclass
class EvalArguments:
    max_length:int = field(
        default=512,
        metadata={"help": "The maximum length of the output text."}
    )

    temperature: float = field(
        default=1.0,
        metadata={"help": "The value used to module the next token probabilities."}
    )

    repetition_penalty: float = field(
        default=1.0,
        metadata={"help": "The parameter for repetition penalty."}
    )


def main(model_args: ModelArguments, data_args: DataArguments, eval_args: EvalArguments):

    if model_args.model_token is not None:
        from huggingface_hub import login
        login(token=model_args.model_token)
    
    tokenizer=AutoTokenizer.from_pretrained(model_args.model_name_or_path)
    pipe = pipeline("summarization", model_args.model_name_or_path,tokenizer=tokenizer ,device=0 if torch.cuda.is_available() else -1)

    dataset = load_dataset(
        data_args.dataset_name_or_path,
        token=data_args.data_token,
        split=data_args.eval_split
    )
    dataset = dataset.shuffle(42)
    dataset = dataset.select(range(10000)) 
    _metric = evaluate.load("rouge")
    outputs = defaultdict(list)
    for batch in tqdm(dataset.iter(batch_size=model_args.batch_size)):
        
        predictions = pipe(
            batch[data_args.text_column],
            batch_size=model_args.batch_size, 
            max_length=eval_args.max_length,
            temperature=eval_args.temperature,
            repetition_penalty=eval_args.repetition_penalty
        )

        predictions = [pred["summary_text"] for pred in predictions]
        outputs["predictions"].extend(predictions)
        outputs["inputs"].extend(batch[data_args.text_column])
        outputs["references"].extend(batch[data_args.label_column])

    results = _metric.compute(predictions=outputs["predictions"], references=outputs["references"])
    print(results)


if __name__ == "__main__":
    parser = HfArgumentParser((ModelArguments, DataArguments, EvalArguments))
    main(*parser.parse_args_into_dataclasses())


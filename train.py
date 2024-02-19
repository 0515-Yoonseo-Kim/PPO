#Argument
import random
from transformers import (
    HfArgumentParser,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments
)
from dataclasses import dataclass, field

import numpy as np
import evaluate

from datasets import load_dataset,Dataset
from collections import defaultdict

@dataclass
class ModelArguments:
    model_name_or_path: str = field(
        default="monologg/kobigbird-bert-base",
        metadata={"help" : ""}
    )
    model_token: str = field(
        default=None,
        metadata={"help" : "token for model"}
    )
    max_length: int = field(
        default = 1024,
        metadata={"help" : ""}
    )

@dataclass
class DataArguments:
    data_name_or_path: str = field(
        default = "korean-benchmark/aihub_sum_books",
        metadata={"help" : ""}
    )

    data_token: str = field(
        default = None,
    )
def get_preprocessed_dataset(dataset:Dataset):
    rng = random.Random(42)
    neg = defaultdict(list)
    for d in dataset:
        pid = "_".join(d["passage_id"].split("_")[:-1])
        neg[pid].append((d["passage_id"], d["summary"]))
    def example_function(example):
    
        doc_id = "_".join(example["passage_id"].split("_")[:-1])
        passage_maps = neg[doc_id]
        NON_NEGATIVE = False

        if len(passage_maps) >1:
            while True:
                neg_id, neg_summary = rng.choice(passage_maps)
                if neg_id!=example["passage_id"]:
                    break
        else:
            neg_id, neg_summary = passage_maps[0]
            NON_NEGATIVE=True
        pos_text = "passage: " + example["passage"]+ "summary: " + example["summary"]
        neg_text = "passage: " + example["passage"]+ "summary: " + neg_summary
        
        return{
            "positive": pos_text,
            "negative": neg_text,
            "negative_is_positive": NON_NEGATIVE
        }

    dataset= dataset.map(example_function,batched=False, remove_columns=dataset.column_names)
    print(dataset)
    
    def flatten(examples):
        """
        examples = {"pos~: pos1, pos2, pos3 "neg~": neg1, neg2, neg3 "negpos~": .. .. ..}
        (default)batch_size=1000
        """
        outputs = defaultdict(list)
        for pos, neg, neg_is_pos in zip(examples["positive"],examples["negative"],examples["negative_is_positive"]):
            outputs["text"].append(pos)
            outputs["label"].append(1)
            outputs["text"].append(neg)
            outputs["label"].append(1 if neg_is_pos else 0)
        return outputs

    dataset = dataset.map(flatten, batched=True, remove_columns=dataset.column_names)
    """
    dataset["train"]=Dataset.from_pandas(
        dataset["train"].to_pandas().drop_duplicates("text",ignore_index=True)
    )
    """
    return dataset
def main(model_args: ModelArguments, data_args: DataArguments, training_args: TrainingArguments):
    
    dataset = load_dataset(data_args.data_name_or_path, token = data_args.data_token)
    dataset["train"]= get_preprocessed_dataset(dataset["train"])
    dataset["validation"]=dataset["validation"].map(
        lambda x: {
            "text" : "passage: "+ x["passage"] + "summary: " + x["summary"],
            "label" : 1
            },
        batched = False,
        remove_columns=dataset["validation"].column_names
    )
    #model_load
    model = AutoModelForSequenceClassification.from_pretrained(model_args.model_name_or_path,
                                                               token = model_args.model_token)
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path,
                                              token = model_args.model_token)
    def tokenize_func(examples):
        tokenized_inputs = tokenizer(
            examples["text"],
            padding = "max_length",
            truncation=True,
            max_length=model_args.max_length
        )
        tokenized_inputs["labels"] = examples["label"]
        return tokenized_inputs
    dataset = dataset.map(
        tokenize_func,
        batched=True,
        remove_columns=dataset["train"].column_names
    )
    dataset = dataset.shuffle(seed=42)

    acc = evaluate.load("accuracy")

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        # prediction : [0.001, 0.04] ->`1, `predictions: [[0.2,0.1],...]-> [0,1,0,,....]
        predictions = np.argmax(predictions, axis=-1)
        return acc.compute(predictions=predictions, references=labels)
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    if training_args.do_train:
        trainer.train()
    elif training_args.do_eval:
        result = trainer.evaluate()
        print(result)




if __name__=="__main__":
    parser = HfArgumentParser((ModelArguments,DataArguments, TrainingArguments))
    main(*parser.parse_args_into_dataclasses())
from typing import Optional
from dataclasses import dataclass, field
from datasets import load_dataset

import numpy as np
import evaluate
import torch
from transformers import (
    HfArgumentParser, 
    Trainer, 
    TrainingArguments, 
    AutoModelForSeq2SeqLM, 
    AutoTokenizer,
    GenerationConfig,
    AutomodelWithLMHead
)

from trl import PPOTrainer
from tqdm import tqdm
from random import choices
from torch import nn

#choice ->하나만 choices -> 여러개
ctrl_str = ["[non-sense]", "[sense]"]
@dataclass
class ModelArguments:
    model_name_or_path: str = field(
        default = "psyche/KoT5",
        metadata={"help":""}
    )
    model_auth_token: Optional[str] = field(
        default = None,
        metadata = {"help":""}
    )
    model_type:str = field(default="seq2seq", metadata={"help":""}) 
@dataclass
class DataArguments:
    data_name_or_path : str = field(
        default = "korean-benchmark/aihub_summ_report",
        metadata= {"help": ""}
    )
    data_auth_token: Optional[str] = field(
        default =None,
        metadata={"help": ""}
    )
    max_input_length : int = field(
        default = 512,
        metadata={"help":""}
    )
    max_output_length : int = field(
        default = 128,
        metadata={"help":""}
    ) 

@dataclass
class RewardArguments:
    reward_name_or_path : str = field(
        default = "korean-benchmark/aihub_summ_report",
        metadata= {"help": ""}
    )
    reward_auth_token: Optional[str] = field(
        default =None,
        metadata={"help": ""}
    )
    max_input_length : int = field(
        default = 512,
        metadata={"help":""}
    )
def main(
        data_args: DataArguments, 
        model_args : ModelArguments, 
        training_args: TrainingArguments,

        reward_args: RewardArguments
):
    dataset = load_dataset(
        data_args.data_name_or_path,
        token = data_args.data_auth_token
    )
    """
    nohup python train.py --output_dir=runs --data_auth_token=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 --per_device_train_batch_size=4 --per_device_eval_batch_size=4 --eval_accumulation_steps=1 --logging_strategy=steps --logging_steps=3000 --save_strategy=steps --save_steps=3000 --evaluation_strategy=steps --eval_steps=3000 --num_train_epochs=1 &
    ?model_name_or_path=runs/checkpoint-3000
    3000
    ?model_name_or_path=runs/checkpoint-3000 ?resume_from_checkpoint=runs/checkpoint-3000
    3000
    """
    #
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_args.model_name_or_path,
        token = model_args.model_auth_token
    )
    ref_model = AutomodelWithLMHead.from_pretrained(
        model_args.model_name_or_path,
        token = model_args.model_auth_token
    )

    reward_model = AutoModelForSequenceClassification.from_pretrained(
        reward_arg.reward_name_or_path,
        token = reward_args.reward_auth_token
    )
    reward_tokenizer = AutoTokenizer.from_pretrained(
        reward_args.reward_name_or_path,
        token = reward_args.reward_auth_token    
    ) 
    
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        token = model_args.model_auth_token    
    )
    
  
    def tokenize_func(ex):
        tokenized_inputs = tokenizer(
            ex['passage']
            ,max_length= data_args.max_input_length   
            ,padding= "max_length"
            ,truncation=True
        ) 

        labels = tokenizer(
            ex['summaries'][0]
            ,max_length=data_args.max_output_length
            ,padding = "max_length"
            ,truncation=True
        )
        tokenized_inputs["labels"] = labels["input_ids"]
        return tokenized_inputs


    dataset = dataset.map(tokenize_func, remove_columns = dataset["train"].column_names)
    _metric = evaluate.load("rouge")

    def compute_metric(value):
        predictions = tokenizer.batch_decode(np.argmax(value.predictions[0], axis=-1), skip_special_tokens=True)
        labels = tokenizer.batch_decode(value.label_ids, skip_special_tokens=True)
        results = _metric.compute(predictions=predictions, references=labels)
        return results

    

            
    trainer = PPOTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = dataset["train"],
        eval_dataset = dataset["validation"],
        compute_metrics = compute_metric,
        args = training_args,
        model_type=model_args.model_type,
        ref_model=ref_model
    )
    for epoch in range(2):
        #tqdm : 시각적으로 보여주는 대충 
        for batch in tqdm(trainer.dataloader):
            (logs, game_data,) = (
                dict(),
                dict(),
            )

            #### prepend a random control token
            #ctrl_str 
            task_list = choices(ctrl_str, k=config.batch_size)
            game_data["query"] =  batch["query"]
            query_tensors = batch["input_ids"]
            
            
            responses = trainer.generate(query_tensors, **config.__dict__)

            response_tensors = []
            for query in query_tensors:
                response = trainer.generate(query, **config.__dict__)
                response_tensors.append(response.squeeze())
            game_data["response"] = tokenizer.batch_decode(responses, skip_special_token = True)

            #### sentiment analysis
            texts = ["passage: "+q+"summary: "+r for q,r in zip(batch["query"], game_data["response"])]
            evaluate_inputs = reward_tokenizer(
                texts,
                padding=true,
                truncation=True,
                return_tensors="pt",
                max_length = reward_args.max_input_length
            ) # evaluate_inputs = {"input_ids": ... ( tensor ) ... ,"attention_mask":... ( tensor ) ...}
            logits = reward_model(
                **evaluate_inputs
            )
            # passage : a, summary : b [0.1,0.9] => 1
            # passage : c, summary : d [0.8,0.2] => 0
            logits = nn.functional.softmax(logits, dim = -1)
            #reward -> 클수록 좋
            #pos_logit-to_reward -> reward를 보정 여기서 안쓰
            rewards= logits[:,1]
            stats = trainer.step(query_tensors,responses,rewards)
            trainer.log_stats(stats, game_data, rewards)
        

    if training_args.do_train:
        trainer.train()

    elif training_args.do_eval:
        print("***** Eval results *****")
        for key, value in trainer.evaluate().items():
            print(f"  {key} = {value:.3f}")
    
if __name__ == "__main__":
    parser = HfArgumentParser((DataArguments,ModelArguments,TrainingArguments, GenerationParams, RewardArguments))
    main(*parser.parse_args_into_dataclasses())

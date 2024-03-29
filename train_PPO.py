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
    AutoModelForSequenceClassification
)

from trl import PPOTrainer, PPOConfig, AutoModelForSeq2SeqLMWithValueHead
from tqdm import tqdm
from random import choices
from torch import nn

#choice ->하나만 choices -> 여러개
# non-sense : 0 / sense : 1 
ctrl_str = ["[non-sense]", "[sense]"]

# ModelArguments : 
@dataclass
class ModelArguments:
    #model_name_or_path - huggingface 모델 불러옴
    model_name_or_path: str = field(
        default = "psyche/KoT5",
        metadata={"help":""}
    )
    #model_auth_token : 모델 토큰 (스크립트에 넣어줘야 함)
    model_auth_token: Optional[str] = field(
        default = None,
        metadata = {"help":""}
    )
    #seq2seq 모델임을 알려줌
    model_type:str = field(default="seq2seq", metadata={"help":""}) 

#DataArguments : 학습데이터 
@dataclass
class DataArguments:
    # data 
    data_name_or_path : str = field(
        default = "korean-benchmark/aihub_summ_report",
        metadata= {"help": ""}
    )
    # data tokens
    data_auth_token: Optional[str] = field(
        default =None,
        metadata={"help": ""}
    )
    # 최대 input길이
    data_max_input_length : int = field(
        default = 512,
        metadata={"help":""}
    )
    # 최대 output길이
    max_output_length : int = field(
        default = 128,
        metadata={"help":""}
    ) 

@dataclass
class GenerationParams:
    gen_max_length : int = field(
        default = 128        
    )
    
    
#보상 모델
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
    reward_max_input_length : int = field(
        default = 512,
        metadata={"help":""}
    )
def main(
        data_args: DataArguments, 
        model_args : ModelArguments, 
        training_args: TrainingArguments,
        gen_args : GenerationParams,
        reward_args: RewardArguments
):
    dataset = load_dataset(
        data_args.data_name_or_path,
        token = data_args.data_auth_token
    )
    """
    nohup python train.py --output_dir=runs --data_auth_token=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 --per_device_train_batch_size=4 --per_device_eval_batch_size=4 --eval_accumulation_steps=1 --logging_strategy=steps --logging_steps=3000 --save_strategy=steps --save_steps=3000 --evaluation_strategy=steps --eval_steps=3000 --num_train_epochs=1 &
    ?model_name_or_path=runs/checkpoint-3000
    """
    #
    model = AutoModelForSeq2SeqLMWithValueHead.from_pretrained(
        model_args.model_name_or_path,
        token = model_args.model_auth_token
    )
    ref_model = AutoModelForSeq2SeqLMWithValueHead.from_pretrained(
        model_args.model_name_or_path,
        token = model_args.model_auth_token
    )

    reward_model = AutoModelForSequenceClassification.from_pretrained(
        reward_args.reward_name_or_path,
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
    config = PPOConfig(**{
        "seed" : 42,
        "mini_batch_size" : 2
    })
    from transformers import AutoConfig
    model_config = AutoConfig.from_pretrained(model_args.model_name_or_path, token=model_args.model_auth_token)
    gen_config = GenerationConfig.from_model_config(model_config=model_config)
    gen_config.max_length = gen_args.gen_max_length
    
    def tokenize_func(ex):
        tokenized_inputs = tokenizer(
            ex['passage']
            ,max_length= data_args.data_max_input_length   
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
        tokenized_inputs["query"] = ex["passage"]
        return tokenized_inputs

    dataset["train"] = dataset["train"].select(range(1000))
    dataset = dataset.map(tokenize_func)
    _metric = evaluate.load("rouge")

    def compute_metric(value):
        predictions = tokenizer.batch_decode(np.argmax(value.predictions[0], axis=-1), skip_special_tokens=True)
        labels = tokenizer.batch_decode(value.label_ids, skip_special_tokens=True)
        results = _metric.compute(predictions=predictions, references=labels)
        return results
    from collections import defaultdict
    from typing import Mapping,List
    import torch
    def batch_collator(batch:List[Mapping]): # [{"input_ids":d1, ....}, {"input_ids":d2, ....}] => {"input_ids":[d1, d2]}
        output = defaultdict(list)
        for data in batch:
            for k,v in data.items():
                output[k].append(v)
        output = {k:(torch.tensor(v) if k in ("input_ids", "attention_mask","token_type_ids") else v) for k,v in output.items() }
        return output
        
    trainer = PPOTrainer(
        model = model,
        tokenizer = tokenizer,
        dataset=dataset["train"],
        data_collator=batch_collator,
        config = config,
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
            #task_list = choices(ctrl_str, k=gen_config.batch_size)
            print(batch.keys())
            game_data["query"] =  batch["query"]
            query_tensors = [b for b in batch["input_ids"]]            
            response_tensors, ref_tensors = trainer.generate(
                query_tensors, return_prompt=False, generate_ref_response=True,**gen_config.to_dict()
            )

            response_tensors = []
            for query in query_tensors:
                response = trainer.generate(query, **gen_config.__dict__)
                response_tensors.append(response.squeeze())
            game_data["response"] = tokenizer.batch_decode(response, skip_special_token = True)

            
            #### sentiment analysis
            texts = ["passage: "+q+"summary: "+r for q,r in zip(batch["query"], game_data["response"])]
            evaluate_inputs = reward_tokenizer(
                texts,
                padding=True,
                truncation=True,
                return_tensors="pt",
                max_length = reward_args.reward_max_input_length
            ) # evaluate_inputs = {"input_ids": ... ( tensor ) ... ,"attention_mask":... ( tensor ) ...}
            logits = reward_model(
                **evaluate_inputs
            ).logits
            # passage : a, summary : b [0.1,0.9] => 1
            # passage : c, summary : d [0.8,0.2] => 0
            logits = nn.functional.softmax(logits, dim = -1)
            #reward -> 클수록 좋
            #pos_logit-to_reward -> reward를 보정 여기서 안쓰
            rewards= logits[:,1]
            stats = trainer.step([q for q in query_tensors],[r for r in response], [r for r in rewards])
            trainer.log_stats(stats, game_data, rewards)
        

    if training_args.do_train:
        trainer.train()

    elif training_args.do_eval:
        print("***** Eval results *****")
        for key, value in trainer.evaluate().items():
            print(f"  {key} = {value:.3f}")
    
if __name__ == "__main__":
    parser = HfArgumentParser((DataArguments,ModelArguments,TrainingArguments,GenerationParams, RewardArguments))
    main(*parser.parse_args_into_dataclasses())
import os
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional, List, Mapping, Union, Dict

import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import (
    HfArgumentParser,
    TrainingArguments,
    AutoTokenizer,
    GenerationConfig,
    AutoModelForSequenceClassification,
    AutoConfig
)
from trl import PPOTrainer, PPOConfig, AutoModelForSeq2SeqLMWithValueHead


@dataclass
class ModelArguments:
    """
    모델 관련 파라미터
    """
    model_name_or_path: str = field(
        default="psyche/KoT5",
        metadata={"help": "huggingface 모델 불러옴"}
    )
    # model_auth_token :
    model_auth_token: Optional[str] = field(
        default=None,
        metadata={"help": "모델 토큰 (Private 모델의 경우 스크립트에 넣어줘야 함)"}
    )
    model_type: str = field(default="seq2seq", metadata={"help": "모델의 유형(seq2seq or causal)"})


@dataclass
class DataArguments:
    """
    데이터 관련 파라미터
    """
    data_name_or_path: str = field(
        default="korean-benchmark/aihub_summ_report",
        metadata={"help": "데이터 이름(huggingface-hub) 또는 경로"}
    )
    data_auth_token: Optional[str] = field(
        default=None,
        metadata={"help": "데이터 토큰 (Private 데이터의 경우 스크립트에 넣어줘야 함)"}
    )
    data_max_input_length: int = field(
        default=512,
        metadata={"help": "최대 input길이"}
    )
    max_output_length: int = field(
        default=256,
        metadata={"help": "최대 output길이"}
    )


@dataclass
class GenerationParams:
    """
    생성 관련 파라미터
    """
    gen_max_length: int = field(
        default=128,
        metadata={"help": "생성 최대 길이"}
    )


@dataclass
class RewardArguments:
    """
    보상 모델 관련 파라미터
    """
    reward_name_or_path: str = field(
        default="korean-benchmark/aihub_summ_report",
        metadata={"help": "reward 모델 이름 또는 경로"}
    )
    reward_auth_token: Optional[str] = field(
        default=None,
        metadata={"help": "reward 모델 토큰 (Private 모델의 경우 스크립트에 넣어줘야 함)"}
    )


def batch_collator(batch: List[Mapping]) -> Dict[str, List[Union[torch.Tensor, str]]]:
    """
    입력 받은 데이터를 batch로 묶어줍니다.
    :param batch:
    :return:
    >>> batch_sample = [
    ...     {"input_ids": [1, 2, 3], "attention_mask": [1, 1, 1]},
    ...     {"input_ids": [4, 5, 6], "attention_mask": [1, 1, 1]}
    ... ]
    >>> batch_collator(batch_sample)
    {'input_ids': [tensor([1, 2, 3]), tensor([4, 5, 6])], 'attention_mask': [tensor([1, 1, 1]), tensor([1, 1, 1])]}
    """
    output = defaultdict(list)
    for data in batch:
        for k, v in data.items():
            if k in ("input_ids", "attention_mask", "token_type_ids"):
                v = torch.tensor(v)
            output[k].append(v)
    return dict(output)


def main(
        data_args: DataArguments,
        model_args: ModelArguments,
        training_args: TrainingArguments,
        gen_args: GenerationParams,
        reward_args: RewardArguments
) -> None:
    dataset = load_dataset(
        data_args.data_name_or_path,
        token=data_args.data_auth_token
    )
    #
    model = AutoModelForSeq2SeqLMWithValueHead.from_pretrained(
        model_args.model_name_or_path,
        token=model_args.model_auth_token
    )
    ref_model = AutoModelForSeq2SeqLMWithValueHead.from_pretrained(
        model_args.model_name_or_path,
        token=model_args.model_auth_token
    )
    from transformers.pipelines import pipeline

    _reward_model = pipeline("text-classification", "psyche/bigbird-summ-reward")
    
    def reward_model(texts):
        """
        보상 모델이 점수 만 출력할 수 있도록 softmax를 사용하여 확률로 변환하는 Wrapper 함수
        Args:
            texts: List[str]
        Returns:
            List[torch.Tensor]: 보상 점수
        """
        rewards = _reward_model(texts)
        rewards = [r["score"] for r in rewards]        
        return [torch.tensor([t]) for t in rewards] 
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        token=model_args.model_auth_token
    )
    config = PPOConfig(**{
        "seed": 42,
        "mini_batch_size": 4,
        "batch_size": training_args.per_device_train_batch_size,
    })
    model_config = AutoConfig.from_pretrained(model_args.model_name_or_path, token=model_args.model_auth_token)
    gen_config = GenerationConfig.from_model_config(model_config=model_config)
    gen_config.max_length = gen_args.gen_max_length

    def tokenize_func(ex):
        tokenized_inputs = tokenizer(
            ex['passage']
            , max_length=data_args.data_max_input_length
            , padding="max_length"
            , truncation=True
        )

        labels = tokenizer(
            ex['summaries'][0]
            , max_length=data_args.max_output_length
            , padding="max_length"
            , truncation=True
        )
        tokenized_inputs["labels"] = labels["input_ids"]
        tokenized_inputs["query"] = ex["passage"]
        return tokenized_inputs

    dataset = dataset.map(tokenize_func)
    print(dataset)
    trainer = PPOTrainer(
        model=model,
        tokenizer=tokenizer,
        dataset=dataset["train"],
        data_collator=batch_collator,
        config=config,
        ref_model=ref_model
    )
    gen_config.repetition_penalty=1.2
    gen_config.temperature=0.1 
    for steps, batch in tqdm(enumerate(trainer.dataloader), desc="Training PPO..."):
        batch["query"] = trainer.tokenizer.batch_decode(batch["input_ids"], skip_special_tokens=True)

        # 입력된 원문(passage)를 이용하여 생성된 요약문(summary)을 생성합니다.
        ## response_tesnors 는 학습 대상이되는 모델의 출력입니다.
        ## ref_response_tensors 는 학습이 되지 않는 모델의 출력입니다.
        response_tensors, ref_response_tensors = trainer.generate(
            batch["input_ids"], return_prompt=False, generate_ref_response=True, **gen_config.to_dict()
        )

        # 생성된 model과 ref_model의 요약문을 텍스트로 변환합니다.
        batch["response"] = trainer.tokenizer.batch_decode(response_tensors, skip_special_tokens=True)
        batch["ref_response"] = trainer.tokenizer.batch_decode(ref_response_tensors, skip_special_tokens=True)
        for q, r in zip(batch["query"], batch["response"]):
            print("########## SAMPLE ##########") 
            print("QUERY:\n", q)
            print("RESPONSE:\n", r)
             
        texts = [f"Passage:{q}, Summary {r}" for q, r in zip(batch["query"], batch["response"])]
        ref_texts = [f"Passage:{q}, Summary {r}" for q, r in zip(batch["query"], batch["ref_response"])]
      
        rewards = [score.to(trainer.current_device) for score in reward_model(texts)]
        ref_rewards = [score.to(trainer.current_device) for score in reward_model(ref_texts)]
        print(rewards) 
        batch["ref_rewards"] = ref_rewards
        # Run PPO step
        try: 
            stats = trainer.step(batch["input_ids"], response_tensors, rewards)
        except:
            continue
        trainer.log_stats(
            stats, batch, rewards,
            columns_to_log=["query", "response", "ref_response", "ref_rewards"]
        )
        import os       
        if steps % training_args.save_steps == 0 and steps > 0:
            save_dir = os.path.join(training_args.output_dir, "checkpoint-{}".format(steps))
            trainer.tokenizer.save_pretrained(save_dir)
            trainer.model.save_pretrained(save_dir)
    
    import os
    trainer.model.save_pretrained(os.path.join(save_dir, "last"))

if __name__ == "__main__":
    parser = HfArgumentParser((DataArguments, ModelArguments, TrainingArguments, GenerationParams, RewardArguments))
    main(*parser.parse_args_into_dataclasses())

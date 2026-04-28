import os
from unsloth import FastLanguageModel
import torch
from datasets import load_dataset, Dataset
from sklearn.model_selection import train_test_split
import pandas as pd
from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth import is_bfloat16_supported
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser(description='Fine-tune a model on a dataset.')
parser.add_argument('-k', type=int, help='Number of training examples to use.', required=True)
parser.add_argument('-r', type=int, help='Random seed to use.', required=True)
parser.add_argument('-e', type=int, help='Number of epochs to train for.', required=True)
args = parser.parse_args()
training_examples = args.k
random_seed = args.r
epochs = args.e


# @title Task configurations
tasks_info = {
    'binary':
        {
            'filename': 'balanced_binary_data.csv',
            'label_names': ['benign', 'illicit'],
            'positive_label': 'illicit',
            'prompt_prefix': "You will act as a content detector.\nFirst, you will be given a set of message examples along with their labels.\nThen, you will answer the category of the last message from the following options: 'benign', 'illicit'.\nReturn only the name of the category, and nothing else."
            },
    'category':
        {
            'filename': 'balanced_category_data.csv',
            'label_names': ['porn', 'gambling', 'drug', 'data-theft', 'money-laundry', 'counterfeit', 'advertisement', 'weapon', 'others', 'surrogacy', 'fraud', 'hacking'],
            'prompt_prefix': "You will act as a content detector.\nFirst, you will be given a set of message examples along with their labels.\nThen, you will answer the category of the last message from the following options: 'porn', 'surrogacy', 'gambling', 'drug', 'data-theft', 'money-laundry', 'counterfeit', 'advertisement', 'weapon', 'fraud', 'hacking', 'others'.\nReturn only the name of the category, and nothing else."
        },
    'contact':
        {
            'filename': 'balanced_contact_data.csv',
            'label_names': ['wechat', 'website', 'telegram', 'others', 'qq'],
            'prompt_prefix': "You will act as a content detector, and decide the type of the contact embedded in the given message.\nFirst, you will be given a set of message examples along with their labels.\nThen, you will answer the content type of the last message from the following options: 'wechat', 'website', 'telegram', 'others', 'qq'.\nReturn only the name of the category, and nothing else."
        },
}

model_name_id_map = {
    'llama3': 'meta-llama/Meta-Llama-3-8B-Instruct',
    'llama3.1': 'meta-llama/Meta-Llama-3.1-8B-Instruct',
    'llama3.1-non-instruct': 'meta-llama/Llama-3.1-8B',
    'mistral': 'mistralai/Mistral-7B-Instruct-v0.2',
    'mistral-non-instruct': 'mistralai/Mistral-7B-v0.3', 
    'phi3-mini': 'microsoft/Phi-3-mini-128k-instruct',
    'phi3-small': 'microsoft/Phi-3-small-128k-instruct',
    'gemma': 'google/gemma-2b-it',
    'qwen': 'Qwen/Qwen2.5-7B-Instruct',
}


max_seq_length = 32768 # Choose any! We auto support RoPE Scaling internally!
dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
load_in_4bit = False # Use 4bit quantization to reduce memory usage. Can be False.

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = 'unsloth/mistral-7b-instruct-v0.2',
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
    # token = 'hf_...',
)

# @title Add LoRA adapters
model = FastLanguageModel.get_peft_model(
    model,
    r = 16, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 16, # The scaling factor for finetuning, suggest it to equal to r
    lora_dropout = 0, # Supports any, but = 0 is optimized
    bias = "none",    # Supports any, but = "none" is optimized
    use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
    random_state = 42,
    use_rslora = False,  # We support rank stabilized LoRA
    loftq_config = None, # And LoftQ
)

data_dir = './Data'
task_name = 'category'
EOS_TOKEN = tokenizer.eos_token # Must add EOS_TOKEN

alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""

def formatting_prompts_func(examples):
    instruction = task_info['prompt_prefix']
    inputs = examples['text']
    outputs = examples['label']
    prompts = []
    for input, output in zip(inputs, outputs):
        prompt = alpaca_prompt.format(instruction, input, output) + EOS_TOKEN
        prompts.append(prompt)
    return {"prompt": prompts}
pass

# Load the dataset
task_info = tasks_info[task_name]
dataset = load_dataset('csv', data_files=f"{data_dir}/{task_info['filename']}")
df = dataset['train'].to_pandas()
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
train_df = train_df.sample(training_examples, random_state=random_seed)
train_df.reset_index(drop=True, inplace=True)
test_df.reset_index(drop=True, inplace=True)

test_dataset = Dataset.from_pandas(test_df)
test_dataset = test_dataset.map(formatting_prompts_func, batched=True)
train_dataset = Dataset.from_pandas(train_df)
train_dataset = train_dataset.map(formatting_prompts_func, batched=True)

trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = train_dataset,
    dataset_text_field = "prompt",
    max_seq_length = max_seq_length,
    dataset_num_proc = 2,
    packing = False, # Can make training 5x faster for short sequences.
    args = TrainingArguments(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        warmup_steps = 5,
        num_train_epochs = epochs, # Set this for 1 full training run.
        # max_steps = 60,
        learning_rate = 2e-4,
        fp16 = not is_bfloat16_supported(),
        bf16 = is_bfloat16_supported(),
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 42,
        output_dir = "/autodl-tmp/outputs",
    ),
)

# Show current memory stats
gpu_stats = torch.cuda.get_device_properties(0)
start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
print(f"{start_gpu_memory} GB of memory reserved.")

trainer_stats = trainer.train()

#@title Show final memory and time stats
used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
used_percentage = round(used_memory         /max_memory*100, 3)
lora_percentage = round(used_memory_for_lora/max_memory*100, 3)
print(f"{trainer_stats.metrics['train_runtime']} seconds used for training.")
print(f"{round(trainer_stats.metrics['train_runtime']/60, 2)} minutes used for training.")
print(f"Peak reserved memory = {used_memory} GB.")
print(f"Peak reserved memory for training = {used_memory_for_lora} GB.")
print(f"Peak reserved memory % of max memory = {used_percentage} %.")
print(f"Peak reserved memory for training % of max memory = {lora_percentage} %.")

train_log_history = trainer.state.log_history
train_log_history_df = pd.DataFrame(train_log_history)

save_path = f"./autodl-tmp/model/mistral_lora_model_{training_examples}_e{epochs}_r{random_seed}"

plt.plot(train_log_history_df['loss'])
plt.savefig(f"{save_path}_loss.png")

model.save_pretrained_merged(save_path, tokenizer, save_method = "merged_16bit")
print(f"The fine-tuned model is saved to {save_path}.")

# !pip install unsloth -q

# if "COLAB_" in "".join(os.environ.keys()):
#     !pip install --no-deps bitsandbytes accelerate peft trl triton cut_cross_entropy unsloth_zoo -q
#     !pip install sentencepiece protobuf "datasets==4.3.0" "huggingface_hub>=0.34.0" hf_transfer -q
#     !pip install --no-deps unsloth -q
#     !pip install wandb -q

import os

from unsloth import FastVisionModel
import torch
from datasets import load_dataset
from unsloth import UnslothVisionDataCollator
from trl import SFTTrainer, SFTConfig

import wandb

wandb.login(key="")
wandb.init(project="weNavigate1")

model, tokenizer = FastVisionModel.from_pretrained(
    "Qwen/Qwen3-VL-8B-Instruct",
    load_in_4bit = True,
    use_gradient_checkpointing = "unsloth",
)

model = FastVisionModel.get_peft_model(
    model,
    finetune_vision_layers     = True,
    finetune_language_layers   = True,
    finetune_attention_modules = True,
    finetune_mlp_modules       = True,

    r = 16,
    lora_alpha = 16,
    lora_dropout = 0,
    bias = "none",
    random_state = 3407,
    use_rslora = False,
    loftq_config = None,
)


dataset = load_dataset("lakminaG/weNavigate-mp3d-minival-habitat-v0.1", split = "train")

instruction_text = "Based on this image, which direction should I go?"

def format_data(sample):
    """
    Converts the dataset row into the format Unsloth expects.
    It takes the PIL image from the 'image' column and injects it
    into the 'messages' structure, replacing the file path string.
    """

    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": sample["image"]},
                {"type": "text", "text": instruction_text}
            ]
        },
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": sample["command"]}
            ]
        }
    ]

    return {"messages": conversation}

converted_dataset = [format_data(sample) for sample in dataset]


FastVisionModel.for_training(model)

trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    data_collator = UnslothVisionDataCollator(model, tokenizer),
    train_dataset = converted_dataset,
    args = SFTConfig(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        warmup_steps = 5,
        #max_steps = 60,
        num_train_epochs = 10,
        learning_rate = 2e-4,
        fp16 = not torch.cuda.is_bf16_supported(),
        bf16 = torch.cuda.is_bf16_supported(),
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "outputs",
        logging_strategy = "steps",
        report_to = "wandb",
        remove_unused_columns = False,
        dataset_text_field = "",
        dataset_kwargs = {"skip_prepare_dataset": True},
        max_length = 2048,
    ),
)

trainer_stats = trainer.train()

FastVisionModel.for_inference(model)

image = dataset[0]["image"]
instruction = "Based on this image, which direction should I go?"

messages = [
    {"role": "user", "content": [
        {"type": "image"},
        {"type": "text", "text": instruction}
    ]}
]
input_text = tokenizer.apply_chat_template(messages, add_generation_prompt = True)
inputs = tokenizer(
    image,
    input_text,
    add_special_tokens = False,
    return_tensors = "pt",
).to("cuda")

from transformers import TextStreamer
text_streamer = TextStreamer(tokenizer, skip_prompt = True)
print("\nModel Prediction:")
output_ = model.generate(**inputs, streamer = text_streamer, max_new_tokens = 128,use_cache = True, temperature = 1.5, min_p = 0.1)


from huggingface_hub import login

FastVisionModel.for_inference(model)

image = dataset[1]["image"]
instruction = "Based on this image, which direction should I go?"

messages = [
    {"role": "user", "content": [
        {"type": "image"},
        {"type": "text", "text": instruction}
    ]}
]
input_text = tokenizer.apply_chat_template(messages, add_generation_prompt = True)
inputs = tokenizer(
    image,
    input_text,
    add_special_tokens = False,
    return_tensors = "pt",
).to("cuda")

from transformers import TextStreamer
text_streamer = TextStreamer(tokenizer, skip_prompt = True)
print("\nModel Prediction:")
_ = model.generate(**inputs, streamer = text_streamer, max_new_tokens = 128,
                   use_cache = True, temperature = 1.5, min_p = 0.1)


# Save and Upload to Hugging Face
repo_id = "lakminaG/weNavigate-Qwen3VL-8B-it-fine-tuned-e1-merged-v1.1"

print(f"\nPushing to Hugging Face Hub: {repo_id}...")

login(token="")

# model.push_to_hub(repo_id)
# tokenizer.push_to_hub(repo_id)

# print("Upload complete!")

# upload merged model files to HF
model.push_to_hub_merged(repo_id, tokenizer, save_method="merged_16bit")





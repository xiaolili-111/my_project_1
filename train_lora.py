# train_lora_fixed.py
import torch
from modelscope import AutoTokenizer, AutoModelForCausalLM
from transformers import TrainingArguments
from peft import LoraConfig, get_peft_model, TaskType
from datasets import load_dataset
from trl import SFTTrainer

# 1. 基础配置
model_name = "Qwen/Qwen2.5-0.5B"
output_dir = "./qwen2.5-0.5b-killer-lora"  # 改个名字，区分旧的 otaku

# 2. 加载 Tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
# Qwen 的 pad_token 通常是 <|endoftext|> 或 eos_token，这里确保它存在
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# 3. 加载底座模型 (关键修改：移除 4-bit 量化，使用 float16)
# 0.5B 模型 FP16 全量加载仅需 ~1GB 显存，无需量化
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,  # 推荐使用 float16 (或 bfloat16 如果是 30/40系显卡)
    device_map="auto",
    trust_remote_code=True
)

# 4. LoRA 配置
peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    inference_mode=False,
    r=16,               # 0.5B 模型可以适当增加 rank
    lora_alpha=32,      # alpha 通常是 rank 的 2 倍
    lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"] # 覆盖更多层效果更好
)

model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

# 5. 加载数据集
dataset = load_dataset("json", data_files="train.json", split="train")

# ---【关键修复】定义格式化函数---
# 这会将 json 中的 messages 列表转换为 Qwen 真正能读懂的字符串格式
def formatting_prompts_func(example):
    output_texts = []
    for message in example['messages']:
        # 使用 tokenizer 自带的模板应用函数
        text = tokenizer.apply_chat_template(
            message,
            tokenize=False,
            add_generation_prompt=False
        )
        output_texts.append(text)
    return output_texts

# 6. 训练参数配置
training_args = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=4,      # 0.5B 模型很小，可以开大一点 batch
    gradient_accumulation_steps=2,
    learning_rate=1e-4,                 # 【调整】对于小模型，2e-4 可能略大，改为 1e-4 更稳
    num_train_epochs=5,                 # 【调整】数据量少，多跑两轮，防止欠拟合
    logging_steps=10,
    save_strategy="epoch",
    fp16=True,                          # 开启混合精度
    optim="adamw_torch",                # 【调整】回归标准优化器，比 8bit 更稳
    weight_decay=0.01,
    warmup_ratio=0.1,
    report_to="none"
)

# 7. SFTTrainer
trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    # 【关键修复】使用 formatting_func 而不是 dataset_text_field
    formatting_func=formatting_prompts_func,
    tokenizer=tokenizer,
    max_seq_length=512,
)

# 8. 开始训练
print("开始训练...")
trainer.train()

# 9. 保存模型
print(f"训练完成，保存到 {output_dir}")
trainer.save_model(output_dir) # SFTTrainer 自带 save_model，会自动处理 LoRA 权重
tokenizer.save_pretrained(output_dir)
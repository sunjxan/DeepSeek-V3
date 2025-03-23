import os
import json
from argparse import ArgumentParser
from typing import List

import torch
from transformers import AutoTokenizer
from safetensors.torch import load_model

from model import Transformer, ModelArgs


def sample(logits, temperature: float = 1.0):
    """
    使用温度调节从logits中采样token（应用Gumbel-Max技巧）
    
    算法流程：
    1. 用温度参数缩放logits（温度→0时趋向贪婪采样，温度→1时保持原始分布）
    2. 计算softmax得到概率分布
    3. 对概率分布添加Gumbel噪声并进行argmax采样
    
    参数:
        logits (torch.Tensor): 模型输出的原始logits，形状为(batch_size, vocab_size)
        temperature (float): 温度参数，控制采样随机性
        
    返回:
        torch.Tensor: 采样得到的token索引，形状为(batch_size,)
    """
    # 防止温度过低导致数值不稳定（设置温度下限为1e-5）
    logits = logits / max(temperature, 1e-5)
    # 计算概率分布
    probs = torch.softmax(logits, dim=-1)
    # 生成指数分布噪声并与概率相除（等效于添加Gumbel噪声后取argmax）
    return probs.div_(torch.empty_like(probs).exponential_(1)).argmax(dim=-1)

@torch.inference_mode()
def generate(
    model: Transformer,
    prompt_tokens: List[List[int]],
    max_new_tokens: int,
    eos_id: int,
    temperature: float = 1.0,
    device: str = 'cpu'
) -> List[List[int]]:
    """
    自回归生成文本的主函数
    
    实现步骤：
    1. 初始化token矩阵，填充prompt tokens
    2. 逐个位置进行前向计算
    3. 采样下一个token并更新矩阵
    4. 遇到EOS token时提前终止生成
    
    参数:
        model: 加载好的Transformer模型
        prompt_tokens: 输入的prompt token列表（每个元素对应一个batch的prompt）
        max_new_tokens: 最大生成token数（不包括prompt长度）
        eos_id: 结束符token id
        temperature: 采样温度
        device: 使用的计算设备
        
    返回:
        生成的token列表（每个元素对应一个batch的生成结果）
    """
    # 验证输入长度不超过模型限制
    prompt_lens = [len(t) for t in prompt_tokens]
    assert max(prompt_lens) <= model.max_seq_len, f"输入长度超过模型最大长度限制（max_seq_len={model.max_seq_len}）"
    
    # 计算总长度（考虑prompt长度和最大生成长度）
    total_len = min(model.max_seq_len, max_new_tokens + max(prompt_lens))
    
    # 初始化token矩阵（用-1填充空白位置）
    tokens = torch.full((len(prompt_tokens), total_len), -1, dtype=torch.long, device=device)
    for i, t in enumerate(prompt_tokens):
        tokens[i, :len(t)] = torch.tensor(t, dtype=torch.long, device=device)
    
    prev_pos = 0  # 记录前一次处理的位置
    finished = torch.tensor([False] * len(prompt_tokens), device=device)  # 标记各batch是否完成生成
    prompt_mask = tokens != -1  # 标识哪些位置是prompt部分
    
    # 自回归生成循环
    for cur_pos in range(min(prompt_lens), total_len):
        # 使用滑动窗口方式获取当前输入的logits（只取最后一个token的输出）
        logits = model.forward(tokens[:, prev_pos:cur_pos], prev_pos)[:, -1]
        
        # 根据温度参数选择采样方式
        if temperature > 0:
            next_token = sample(logits, temperature)
        else:  # 温度=0时使用贪婪采样
            next_token = logits.argmax(dim=-1)
        
        # 对于prompt部分直接使用原始token，生成部分使用采样结果
        next_token = torch.where(prompt_mask[:, cur_pos], tokens[:, cur_pos], next_token)
        tokens[:, cur_pos] = next_token
        
        # 更新完成状态：当前是生成部分（非prompt）且遇到EOS token
        finished |= torch.logical_and(~prompt_mask[:, cur_pos], next_token == eos_id)
        prev_pos = cur_pos
        
        # 所有batch都完成生成则提前退出
        if finished.all():
            break
    
    # 后处理：截取生成部分并去除EOS之后的内容
    completion_tokens = []
    for i, toks in enumerate(tokens.tolist()):
        # 截取生成部分（从prompt长度开始，最多取max_new_tokens个）
        generated = toks[prompt_lens[i]:prompt_lens[i]+max_new_tokens]
        # 找到第一个EOS并截断
        if eos_id in generated:
            generated = generated[:generated.index(eos_id)]
        completion_tokens.append(generated)
    
    return completion_tokens

def main(
    ckpt_path: str,
    config: str,
    input_file: str = "",
    interactive: bool = True,
    max_new_tokens: int = 100,
    temperature: float = 1.0,
) -> None:
    """
    主函数：加载模型并进行交互式/批量文本生成
    
    功能说明：
    1. 初始化随机种子
    2. 加载模型配置和checkpoint
    3. 初始化tokenizer
    4. 根据模式选择交互生成或批量生成
    
    参数:
        ckpt_path: 模型checkpoint路径
        config: 模型配置文件路径
        input_file: 输入文件路径（批量模式使用）
        interactive: 是否启用交互模式
        max_new_tokens: 最大生成token数
        temperature: 采样温度
    """
    # 设置随机种子（保证可重复性）
    torch.manual_seed(965)
    
    tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-V3")
    
    # 初始化模型参数（示例代码，实际需要从配置文件加载）
    # with open(config, 'r') as f:
    #     args = ModelArgs(**json.load(f))
    # print(args)
    args = ModelArgs(vocab_size=len(tokenizer.get_vocab()))
    
    # 初始化模型（示例代码，实际需要加载checkpoint）
    model = Transformer(args)
    
    # 初始化tokenizer（示例使用BERT中文分词器，实际应与模型匹配）
    # tokenizer = AutoTokenizer.from_pretrained(ckpt_path)
    # tokenizer.decode(generate(model, [tokenizer.encode("DeepSeek")], 2, -1, 1.)[0])
    # load_model(model, os.path.join(ckpt_path, f"model{rank}-mp{world_size}.safetensors"))
    
    if interactive:  # 交互模式
        messages = []  # 保存对话历史
        while True:
            prompt = input(">>> ")  # 获取用户输入
            if prompt == "/exit":   # 退出命令
                break
            if prompt == "/clear":  # 清空对话历史
                messages.clear()
                continue
            
            # 构建对话格式并生成回复
            messages.append({"role": "user", "content": prompt})
            # 将对话历史转换为模型输入格式
            prompt_tokens = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
            # 生成token
            completion_tokens = generate(model, [prompt_tokens], max_new_tokens, tokenizer.eos_token_id, temperature)
            # 解码生成结果
            completion = tokenizer.decode(completion_tokens[0], skip_special_tokens=True)
            
            print(completion)
            messages.append({"role": "assistant", "content": completion})
    else:  # 批量模式
        with open(input_file) as f:
            prompts = [line.strip() for line in f.readlines()]
        
        # 验证batch大小不超过模型限制
        assert len(prompts) <= args.max_batch_size, f"输入数量超过最大批处理大小（max_batch_size={args.max_batch_size}）"
        
        # 将每个prompt转换为模型输入格式
        prompt_tokens = [
            tokenizer.apply_chat_template([{"role": "user", "content": prompt}], add_generation_prompt=True)
            for prompt in prompts
        ]
        
        # 批量生成
        completion_tokens = generate(model, prompt_tokens, max_new_tokens, tokenizer.eos_token_id, temperature)
        # 批量解码
        completions = tokenizer.batch_decode(completion_tokens, skip_special_tokens=True)
        
        # 打印输入输出结果
        for prompt, completion in zip(prompts, completions):
            print("输入:", prompt)
            print("输出:", completion)
            print()

if __name__ == "__main__":
    """
    命令行接口说明：
    
    参数说明：
        --ckpt-path: 模型checkpoint路径（必需）
        --config: 模型配置文件路径（必需）
        --input-file: 输入文件路径（批量模式使用）
        --interactive: 启用交互模式（与--input-file互斥）
        --max-new-tokens: 最大生成token数（默认200）
        --temperature: 采样温度（默认0.2，值越大随机性越强）
    
    使用示例：
        交互模式：python script.py --ckpt-path ./model --config config.json --interactive
        批量模式：python script.py --ckpt-path ./model --config config.json --input-file prompts.txt
    """
    parser = ArgumentParser(description="分布式文本生成命令行工具")
    parser.add_argument("--ckpt-path", type=str, default="", help="模型checkpoint路径")
    parser.add_argument("--config", type=str, default="", help="模型配置文件路径")
    parser.add_argument("--input-file", type=str, default="", help="输入文件路径（批量模式）")
    parser.add_argument("--interactive", action="store_true", help="启用交互模式")
    parser.add_argument("--max-new-tokens", type=int, default=200, help="最大生成token数")
    parser.add_argument("--temperature", type=float, default=0.2, help="采样温度（0-1之间）")
    
    args = parser.parse_args()
    # 验证至少指定一种模式
    assert args.input_file or args.interactive, "必须指定--input-file或--interactive参数"
    
    main(args.ckpt_path, args.config, args.input_file, args.interactive, args.max_new_tokens, args.temperature)

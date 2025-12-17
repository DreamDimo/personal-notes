# VERL 

> **文档目标**: 完全理解 GRPO 训练流程，知道每一步在哪个文件的哪一行，数据如何流转，以及如何修改代码。

---

## 目录

1. [快速概览](#1-快速概览)
2. [G 参数配置](#2-g-参数配置)
3. [完整执行流程](#3-完整执行流程)
4. [核心算法详解](#4-核心算法详解)
5. [修改代码指南](#5-修改代码指南)
6. [常见问题](#6-常见问题)

---

## 1. 快速概览

### 1.1 GRPO 算法核心思想

**传统 PPO**: 需要一个 Critic 网络估计 Value，使用 GAE 计算优势。

**GRPO (Group Relative Policy Optimization)**:
- 为每个 prompt 生成 **G 个回答** (例如 G=5)
- 对每个回答打分 (正确=1.0, 错误=0.0)
- 计算**组内相对优势**: `advantage = (score - mean) / std`
- 好回答得正分，坏回答得负分
- 使用 PPO 更新策略，增加好回答概率，降低坏回答概率

**优势**:
- ✅ 不需要 Critic 网络，简化训练
- ✅ 组内对比学习，更稳定
- ✅ 特别适合数学、代码等有明确正误的任务

### 1.2 执行流程概览

```
启动 → 加载数据 → 生成 G 个回答 → 计算奖励 → 计算 GRPO 优势 → 更新模型 → 验证 → 保存
```

### 1.3 核心文件地图

| 功能模块 | 核心文件 | 关键行数 |
|---------|---------|---------|
| **启动入口** | `verl/trainer/main_ppo.py` | 35-368 |
| **训练循环** | `verl/trainer/ppo/ray_trainer.py` | 977-1325 |
| **GRPO 算法** | `verl/trainer/ppo/core_algos.py` | 265-328 |
| **生成回答** | `verl/workers/fsdp_workers.py` | 911-957 |
| **更新策略** | `verl/workers/actor/dp_actor.py` | 398-600 |
| **GSM8K 评分** | `verl/utils/reward_score/gsm8k.py` | 52-72 |
| **配置文件** | `examples/grpo_trainer/run_qwen3-8b.sh` | 全部 |

---

## 2. G 参数配置

### 2.1 G 参数在哪里配置？

**文件**: `examples/grpo_trainer/run_qwen3-8b.sh`

```bash
# 第 31 行
actor_rollout_ref.rollout.n=5
```

**含义**:
- `n=5` 表示为每个 prompt 生成 **5 个不同的回答**
- 这就是 GRPO 中的 **G 参数**
- G 越大，训练越稳定，但计算成本越高

### 2.2 G 参数如何传递？

```
run_qwen3-8b.sh:31 (n=5)
  ↓ (通过 Hydra 配置系统)
verl/trainer/main_ppo.py:36
  ↓
config.actor_rollout_ref.rollout.n
  ↓
verl/trainer/ppo/ray_trainer.py:1057
  ↓
gen_batch.repeat(repeat_times=config.actor_rollout_ref.rollout.n)
```

### 2.3 如何修改 G 参数？

**方法 1**: 修改脚本
```bash
# 在 run_qwen3-8b.sh 第 31 行
actor_rollout_ref.rollout.n=10  # 改为 10
```

**方法 2**: 命令行覆盖
```bash
bash run_qwen3-8b.sh actor_rollout_ref.rollout.n=10
```

**建议值**:
- **小任务**: n=5 (默认)
- **复杂任务**: n=8~10 (更稳定)
- **快速实验**: n=2~3 (计算快但不稳定)

---

## 3. 完整执行流程

### 流程图

```
┌─────────────────────────────────────────────────────────────┐
│ Step 0: 启动和初始化                                         │
│   文件: main_ppo.py:35-368                                   │
│   作用: 启动 Ray, 加载数据集, 初始化 Worker                  │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ Step 1: 加载 Batch                                          │
│   文件: ray_trainer.py:1033-1051                            │
│   输入: 256 个 prompt (数学题)                               │
│   输出: batch (DataProto)                                    │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ Step 2: 重复 Batch (n=5)                                    │
│   文件: ray_trainer.py:1057-1059                            │
│   输入: 256 prompts                                         │
│   输出: 1280 prompts (256 × 5)                              │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ Step 3: 生成回答 (vLLM)                                     │
│   文件: ray_trainer.py:1066 → fsdp_workers.py:911          │
│   输入: 1280 prompts                                        │
│   输出: 1280 responses + rollout_log_probs                  │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ Step 4: 计算奖励                                            │
│   文件: ray_trainer.py:1128 → gsm8k.py:52                  │
│   输入: 1280 responses                                      │
│   输出: 1280 scores (1.0 或 0.0)                            │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ Step 5: 计算 old_log_probs                                  │
│   文件: ray_trainer.py:1146 → dp_actor.py:180              │
│   输入: input_ids + responses                               │
│   输出: old_log_probs (用于 PPO clip)                       │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ Step 6: 计算 GRPO 优势                                      │
│   文件: ray_trainer.py:1222 → core_algos.py:265            │
│   输入: scores, uid (分组标识)                               │
│   输出: advantages (组内标准化优势)                          │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ Step 7: 更新 Actor (PPO)                                    │
│   文件: ray_trainer.py:1247 → dp_actor.py:398              │
│   输入: advantages, old_log_probs                           │
│   输出: 更新后的模型参数                                     │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ Step 8: 验证测试 (每 5 步)                                  │
│   文件: ray_trainer.py:531-630                              │
│   输出: val_accuracy, val_pass@5                            │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ Step 9: 保存检查点 (每 20 步)                               │
│   文件: ray_trainer.py:1286                                 │
│   输出: checkpoint 文件                                      │
└─────────────────────────────────────────────────────────────┘
```

---

### Step 0: 启动和初始化

#### 📁 文件: `verl/trainer/main_ppo.py`

#### 🎯 作用
启动整个训练系统，初始化所有组件。

#### 📝 代码详解

**入口函数** (第 35-42 行):
```python
@hydra.main(config_path="config", config_name="ppo_trainer", version_base=None)
def main(config):
    """Main entry point for PPO training with Hydra configuration management."""
    run_ppo(config)
```

**解释**:
- `@hydra.main`: Hydra 装饰器，自动加载配置文件
- `config_path="config"`: 配置文件目录
- `config_name="ppo_trainer"`: 默认配置文件名
- 命令行参数会覆盖配置文件 (例如 `algorithm.adv_estimator=grpo`)

---

**启动 Ray 集群** (第 55-74 行):
```python
def run_ppo(config, task_runner_class=None) -> None:
    if not ray.is_initialized():
        # 设置环境变量
        default_runtime_env = get_ppo_ray_runtime_env()
        ray_init_kwargs = config.ray_kwargs.get("ray_init", {})
        runtime_env_kwargs = ray_init_kwargs.get("runtime_env", {})

        runtime_env = OmegaConf.merge(default_runtime_env, runtime_env_kwargs)
        ray_init_kwargs = OmegaConf.create({**ray_init_kwargs, "runtime_env": runtime_env})

        # 初始化 Ray
        ray.init(**OmegaConf.to_container(ray_init_kwargs))
```

**解释**:
- Ray 是分布式计算框架
- 用于管理多 GPU 训练
- `runtime_env`: 设置环境变量 (CUDA, vLLM 等)

---

**启动任务** (第 76-96 行):
```python
if task_runner_class is None:
    task_runner_class = ray.remote(num_cpus=1)(TaskRunner)

runner = task_runner_class.remote()
ray.get(runner.run.remote(config))
```

**解释**:
- `TaskRunner`: 主训练类
- `ray.remote()`: 将类转为 Ray Actor (可以远程调用)
- `runner.run.remote(config)`: 远程执行 `run` 方法
- `ray.get()`: 等待执行完成

---

**TaskRunner.run()** (第 262-368 行):
```python
def run(self, config):
    # 1. 添加 Worker
    actor_rollout_cls, ray_worker_group_cls = self.add_actor_rollout_worker(config)
    self.add_critic_worker(config)  # GRPO 不需要，但会检查
    self.add_reward_model_worker(config)
    self.add_ref_policy_worker(config, actor_rollout_cls)

    # 2. 验证配置
    validate_config(
        config=config,
        use_reference_policy=need_reference_policy(self.role_worker_mapping),
        use_critic=need_critic(config),
    )

    # 3. 加载模型路径
    local_path = copy_to_local(
        config.actor_rollout_ref.model.path,  # Qwen/Qwen3-8B
        use_shm=config.actor_rollout_ref.model.get("use_shm", False)
    )

    # 4. 加载 tokenizer
    trust_remote_code = config.data.get("trust_remote_code", False)
    tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)
    processor = hf_processor(local_path, trust_remote_code=trust_remote_code)

    # 5. 加载奖励函数
    reward_fn = load_reward_manager(
        config, tokenizer, num_examine=0, **config.reward_model.get("reward_kwargs", {})
    )
    val_reward_fn = load_reward_manager(
        config, tokenizer, num_examine=1, **config.reward_model.get("reward_kwargs", {})
    )

    # 6. 创建数据集
    train_dataset = create_rl_dataset(
        config.data.train_files,  # $HOME/data/gsm8k/train.parquet
        config.data,
        tokenizer,
        processor,
        is_train=True,
    )
    val_dataset = create_rl_dataset(
        config.data.val_files,  # $HOME/data/gsm8k/test.parquet
        config.data,
        tokenizer,
        processor,
        is_train=False,
    )

    # 7. 创建 Trainer
    trainer = RayPPOTrainer(
        config=config,
        tokenizer=tokenizer,
        processor=processor,
        role_worker_mapping=self.role_worker_mapping,
        resource_pool_manager=resource_pool_manager,
        ray_worker_group_cls=ray_worker_group_cls,
        reward_fn=reward_fn,
        val_reward_fn=val_reward_fn,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        collate_fn=collate_fn,
        train_sampler=train_sampler,
    )

    # 8. 初始化 Worker 并开始训练
    trainer.init_workers()
    trainer.fit()  # ← 进入主训练循环
```

**解释**:
- **Worker**: 分布式训练中的工作进程
  - `ActorRollout`: 负责生成回答和训练策略
  - `Critic`: 估计 Value (GRPO 不需要)
  - `RewardModel`: 计算奖励 (如果使用模型打分)
  - `RefPolicy`: 参考策略 (用于 KL 散度)

- **数据集**: GSM8K 数学题
  - `train.parquet`: 训练集
  - `test.parquet`: 验证集

- **奖励函数**: `load_reward_manager` 会根据配置加载
  - 对于 GSM8K: 加载 `verl/utils/reward_score/gsm8k.py:compute_score`

#### 🔧 修改点
- **修改模型**: 第 307 行 `config.actor_rollout_ref.model.path`
- **修改数据集**: 第 331、340 行 `config.data.train_files`, `config.data.val_files`
- **修改奖励函数**: 第 319 行 `load_reward_manager` (需要注册自定义奖励)

---

### Step 1: 加载 Batch

#### 📁 文件: `verl/trainer/ppo/ray_trainer.py`

#### 🎯 作用
从数据集中加载一个 batch 的 prompt。

#### 📝 代码详解 (第 1032-1051 行)

```python
for epoch in range(current_epoch, self.config.trainer.total_epochs):  # 15 个 epoch
    for batch_dict in self.train_dataloader:  # 遍历数据集
        metrics = {}
        timing_raw = {}

        # 1. 将 dict 转为 DataProto
        batch: DataProto = DataProto.from_single_dict(batch_dict)

        # 2. 设置温度参数 (控制生成随机性)
        batch.meta_info["temperature"] = self.config.actor_rollout_ref.rollout.temperature

        # 3. 添加 uid (用于 GRPO 分组)
        batch.non_tensor_batch["uid"] = np.array(
            [str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object
        )
```

**batch_dict 的结构** (来自 DataLoader):
```python
batch_dict = {
    # Tensor 数据
    "input_ids": torch.Tensor([256, 512]),      # 256 个 prompt, 每个最多 512 tokens
    "attention_mask": torch.Tensor([256, 512]), # 注意力 mask

    # 非 Tensor 数据
    "reward_model": [
        {
            "ground_truth": "8",     # 正确答案
            "style": "rule",         # 使用规则评分
            "data_source": "gsm8k"
        },
        # ... 256 个元素
    ]
}
```

**转换后的 DataProto**:
```python
batch = DataProto(
    batch={
        "input_ids": torch.Tensor([256, 512]),
        "attention_mask": torch.Tensor([256, 512]),
    },
    non_tensor_batch={
        "uid": np.array([
            "uuid-0", "uuid-1", "uuid-2", ..., "uuid-255"  # 256 个唯一 ID
        ]),
        "reward_model": [...]  # 256 个配置
    },
    meta_info={
        "temperature": 0.0  # 贪婪解码
    }
)
```

**uid 的作用**:
- 在 GRPO 中，同一个 prompt 的 5 个回答需要**共享同一个 uid**
- 用于在 Step 6 中按 uid 分组计算组内优势

#### 🔧 修改点
- **修改 batch_size**: `run_qwen3-8b.sh:10` → `data.train_batch_size=1024`
- **修改 temperature**: `run_qwen3-8b.sh` 中添加 `actor_rollout_ref.rollout.temperature=0.7` (增加随机性)

---

### Step 2: 重复 Batch (n=5)

#### 📁 文件: `verl/trainer/ppo/ray_trainer.py`

#### 🎯 作用
将每个 prompt 重复 n=5 次，为 GRPO 准备数据。

#### 📝 代码详解 (第 1053-1059 行)

```python
# 1. 提取生成所需的字段
gen_batch = self._get_gen_batch(batch)

# 2. 设置全局步数 (用于追踪)
gen_batch.meta_info["global_steps"] = self.global_steps

# 3. 重复 n 次 (n=5)
gen_batch_output = gen_batch.repeat(
    repeat_times=self.config.actor_rollout_ref.rollout.n,  # 5
    interleave=True  # 交错重复
)
```

**`_get_gen_batch` 函数**:
```python
def _get_gen_batch(self, batch):
    """提取生成所需的字段"""
    return batch.select(batch_keys=["input_ids", "attention_mask", "position_ids"])
```

**`repeat` 方法的效果**:

**输入** (256 个 prompt):
```python
gen_batch.batch = {
    "input_ids": [[prompt_0], [prompt_1], ..., [prompt_255]],  # (256, 512)
}
gen_batch.non_tensor_batch = {
    "uid": ["uuid-0", "uuid-1", ..., "uuid-255"]
}
```

**输出** (1280 个 prompt, 256 × 5):
```python
gen_batch_output.batch = {
    "input_ids": [
        [prompt_0], [prompt_0], [prompt_0], [prompt_0], [prompt_0],  # prompt_0 重复 5 次
        [prompt_1], [prompt_1], [prompt_1], [prompt_1], [prompt_1],  # prompt_1 重复 5 次
        ...,
        [prompt_255], [prompt_255], [prompt_255], [prompt_255], [prompt_255]
    ]  # (1280, 512)
}

gen_batch_output.non_tensor_batch = {
    "uid": [
        "uuid-0", "uuid-0", "uuid-0", "uuid-0", "uuid-0",  # 同一个 prompt 的 5 个回答共享 uid
        "uuid-1", "uuid-1", "uuid-1", "uuid-1", "uuid-1",
        ...,
        "uuid-255", "uuid-255", "uuid-255", "uuid-255", "uuid-255"
    ]  # (1280,)
}
```

**为什么需要 `interleave=True`?**

- 保证同一个 prompt 的 5 个回答在 batch 中是连续的
- 方便后续按 uid 分组

#### 🔧 修改点
- **修改生成数量**: `run_qwen3-8b.sh:31` → `actor_rollout_ref.rollout.n=10` (改为生成 10 个回答)

---

### Step 3: 生成回答 (vLLM)

#### 📁 文件链
```
verl/trainer/ppo/ray_trainer.py:1064-1071
  ↓ (RPC 调用)
verl/workers/fsdp_workers.py:911-957
  ↓ (调用 rollout)
verl/workers/rollout/vllm_rollout/vllm_rollout.py
```

#### 🎯 作用
使用 vLLM 引擎并行生成 1280 个回答。

#### 📝 代码详解

**调用生成** (ray_trainer.py:1064-1071):
```python
with marked_timer("gen", timing_raw, color="red"):
    # 根据是否异步模式选择生成方式
    if not self.async_rollout_mode:
        gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch_output)
    else:
        gen_batch_output = self.async_rollout_manager.generate_sequences(gen_batch_output)

    # 记录生成时间
    timing_raw.update(gen_batch_output.meta_info["timing"])
    gen_batch_output.meta_info.pop("timing", None)
```

**Worker 生成方法** (fsdp_workers.py:911-957):
```python
def generate_sequences(self, prompts: DataProto):
    """生成序列的主函数"""
    # 1. 将数据移到 GPU
    prompts = prompts.to(get_device_id())

    # 2. 设置生成参数
    meta_info = {
        "eos_token_id": self.tokenizer.eos_token_id,
        "pad_token_id": self.tokenizer.pad_token_id,
    }
    prompts.meta_info.update(meta_info)

    timing_generate = {}

    # 3. 切换到 rollout 模式 (如果同时用于训练和生成)
    if self._is_actor:
        loop = get_event_loop()
        loop.run_until_complete(self.rollout_mode())
        log_gpu_memory_usage("After switch to rollout mode", logger=logger)

    # 4. 调用 vLLM 生成
    with simple_timer("generate_sequences", timing_generate):
        output = self.rollout.generate_sequences(prompts=prompts)

    # 5. 切换回训练模式
    if self._is_actor:
        loop.run_until_complete(self.trainer_mode())
        log_gpu_memory_usage("After switch to trainer mode", logger=logger)

    # 6. 记录 timing
    timing_generate_topk_ratio, timing_generate_min, timing_generate_max = \
        topk_reduce_ratio_min_max(timing_generate["generate_sequences"])
    timing_generate = reduce_timing(timing_generate)
    timing_generate.update({
        "generation_timing/max": timing_generate_max,
        "generation_timing/min": timing_generate_min,
        "generation_timing/topk_ratio": timing_generate_topk_ratio,
    })
    output.meta_info["timing"] = timing_generate

    # 7. 移回 CPU 并清理缓存
    output = output.to("cpu")
    get_torch_device().empty_cache()

    return output
```

**vLLM 生成核心逻辑** (简化版):
```python
# 实际在 vllm_rollout.py 中实现
def generate_sequences(self, prompts: DataProto) -> DataProto:
    """使用 vLLM 引擎生成"""
    # 1. 准备 sampling parameters
    sampling_params = SamplingParams(
        temperature=prompts.meta_info.get("temperature", 0.0),
        top_p=self.config.top_p,
        top_k=self.config.top_k,
        max_tokens=self.config.max_response_length,
        n=1,  # 每个 prompt 生成 1 个回答 (已经在外部重复了)
    )

    # 2. 调用 vLLM engine
    outputs = self.llm.generate(
        prompt_token_ids=prompts.batch["input_ids"].tolist(),
        sampling_params=sampling_params,
    )

    # 3. 提取结果
    responses = []
    rollout_log_probs = []
    for output in outputs:
        responses.append(output.outputs[0].token_ids)
        rollout_log_probs.append(output.outputs[0].cumulative_logprob)

    # 4. 转为 DataProto
    return DataProto(
        batch={
            "responses": torch.tensor(responses),
            "rollout_log_probs": torch.tensor(rollout_log_probs),
            "response_mask": compute_response_mask(responses),
        }
    )
```

**生成结果**:
```python
output = DataProto(
    batch={
        "responses": torch.Tensor([1280, 1024]),  # 生成的 token IDs
        # 每个 response 的形状: [token_0, token_1, ..., eos_token, pad, pad, ...]

        "rollout_log_probs": torch.Tensor([1280, 1024]),  # 每个 token 的 log 概率
        # vLLM 生成时计算的 log P(token | prefix)

        "response_mask": torch.Tensor([1280, 1024]),  # 有效 token 的 mask
        # 1 表示有效 token, 0 表示 padding
    }
)
```

**示例** (第 0 个 prompt 的 5 个回答):
```
Prompt: "Olivia has $23. She bought five bagels for $3 each. How much money does she have left?"

Response 0: "She spent 5 * $3 = $15. So she has $23 - $15 = $8 left. #### 8"
Response 1: "The bagels cost 5 × 3 = 15 dollars. She has 23 - 15 = 8 dollars left. #### 8"
Response 2: "Cost is 5*3=15. Remaining is 23-15=8. #### 8"
Response 3: "She bought 5 bagels at $3 each = $15. She started with $23. 23-15=8. #### 7"  # 错误!
Response 4: "5 bagels cost $15 total. $23 - $15 = $8. #### 8"
```

#### 🔧 修改点
- **修改生成引擎**: `run_qwen3-8b.sh:29` → `actor_rollout_ref.rollout.name=sglang` (改用 SGLang)
- **修改 max_tokens**: `run_qwen3-8b.sh:12` → `data.max_response_length=2048` (允许更长回答)
- **修改采样参数**: 在配置中添加 `actor_rollout_ref.rollout.temperature=0.7` (增加多样性)

---

### Step 4: 计算奖励

#### 📁 文件链
```
verl/trainer/ppo/ray_trainer.py:1102-1128
  ↓
verl/trainer/ppo/reward.py:200-219
  ↓
verl/utils/reward_score/gsm8k.py:52-72
```

#### 🎯 作用
对生成的 1280 个回答进行评分。

#### 📝 代码详解

**合并数据** (ray_trainer.py:1102-1106):
```python
# 1. 重复原始 batch 以对齐 5 个回答
batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)

# 2. 合并生成的 responses
batch = batch.union(gen_batch_output)

# 3. 计算 response_mask (如果没有)
if "response_mask" not in batch.batch.keys():
    batch.batch["response_mask"] = compute_response_mask(batch)
```

**合并后的数据结构**:
```python
batch.batch = {
    "input_ids": torch.Tensor([1280, 512]),           # 重复后的 prompt
    "attention_mask": torch.Tensor([1280, 1536]),     # prompt + response
    "responses": torch.Tensor([1280, 1024]),          # 生成的回答
    "rollout_log_probs": torch.Tensor([1280, 1024]), # vLLM 的 log_probs
    "response_mask": torch.Tensor([1280, 1024]),      # response 的 mask
}

batch.non_tensor_batch = {
    "uid": np.array([...]),  # 1280 个 uid (每组 5 个相同)
    "reward_model": [...],   # 1280 个配置 (每组 5 个相同)
}
```

---

**调用奖励计算** (ray_trainer.py:1117-1128):
```python
with marked_timer("reward", timing_raw, color="yellow"):
    # 1. 如果使用 RM 模型打分
    if self.use_rm and "rm_scores" not in batch.batch.keys():
        reward_tensor = self.rm_wg.compute_rm_score(batch)
        batch = batch.union(reward_tensor)

    # 2. 调用奖励函数 (规则或其他)
    if self.config.reward_model.launch_reward_fn_async:
        # 异步调用 (不阻塞)
        future_reward = compute_reward_async.remote(
            data=batch, config=self.config, tokenizer=self.tokenizer
        )
    else:
        # 同步调用
        reward_tensor, reward_extra_infos_dict = compute_reward(batch, self.reward_fn)
```

---

**compute_reward 函数** (reward.py:200-219):
```python
@tqbridge(put_data=False)
def compute_reward(data: DataProto, reward_fn: AbstractRewardManager) -> tuple[torch.Tensor, dict[str, Any]]:
    """
    计算奖励的统一接口

    Args:
        data: 包含 input_ids, responses, reward_model 配置的 DataProto
        reward_fn: 奖励管理器实例 (例如 NaiveRewardManager)

    Returns:
        reward_tensor: shape (batch_size, response_length)
        reward_extra_info: 额外信息 (例如准确率)
    """
    # 调用 reward_fn (会自动调用 compute_score)
    result = reward_fn(data, return_dict=True)

    reward_tensor = result["reward_tensor"]
    reward_extra_info = result.get("reward_extra_info", {})

    return reward_tensor, reward_extra_info
```

**NaiveRewardManager 内部逻辑** (简化):
```python
def __call__(self, data: DataProto, return_dict=True):
    """对每个 response 调用 compute_score"""
    rewards = []
    extra_infos = defaultdict(list)

    for i in range(len(data)):
        # 提取单个样本
        response_text = self.tokenizer.decode(data[i].batch["responses"], skip_special_tokens=True)
        ground_truth = data[i].non_tensor_batch["reward_model"]["ground_truth"]

        # 调用用户定义的 compute_score
        score = self.compute_score(response_text, ground_truth)
        rewards.append(score)

        # 记录额外信息
        extra_infos["is_correct"].append(score > 0)

    # 转为 tensor (outcome reward: 只有最后一个 token 有奖励)
    reward_tensor = torch.zeros(len(data), max_response_length)
    for i, score in enumerate(rewards):
        last_valid_idx = data[i].batch["response_mask"].sum() - 1
        reward_tensor[i, last_valid_idx] = score

    return {
        "reward_tensor": reward_tensor,
        "reward_extra_info": extra_infos
    }
```

---

**GSM8K compute_score** (gsm8k.py:52-72):
```python
def compute_score(solution_str, ground_truth, method="strict", format_score=0.0, score=1.0):
    """
    GSM8K 评分函数

    Args:
        solution_str: 模型生成的回答文本
        ground_truth: 正确答案 (例如 "8")
        method: 提取方法 ("strict" 或 "flexible")
        format_score: 格式正确但答案错误的分数 (默认 0.0)
        score: 答案正确的分数 (默认 1.0)

    Returns:
        float: 0.0 或 1.0
    """
    # 1. 提取答案
    answer = extract_solution(solution_str=solution_str, method=method)

    # 2. 判断正确性
    if answer is None:
        return 0  # 没有找到答案格式
    else:
        if answer == ground_truth:
            return score  # 1.0
        else:
            return format_score  # 0.0
```

**extract_solution 函数** (gsm8k.py:20-49):
```python
def extract_solution(solution_str, method="strict"):
    """
    从回答中提取数字答案

    Args:
        solution_str: 回答文本
        method: "strict" (严格匹配 #### [数字]) 或 "flexible" (任意数字)

    Returns:
        str or None: 提取的答案
    """
    # 优化: 只匹配最后 300 个字符 (答案通常在末尾)
    if len(solution_str) > 300:
        solution_str = solution_str[-300:]

    if method == "strict":
        # 匹配 "#### [数字]" 格式
        solutions = re.findall(r"#### (\-?[0-9\.\,]+)", solution_str)
        if len(solutions) == 0:
            final_answer = None
        else:
            # 取最后一个匹配
            final_answer = solutions[-1].replace(",", "").replace("$", "")
    elif method == "flexible":
        # 匹配任意数字
        answer = re.findall(r"(\-?[0-9\.\,]+)", solution_str)
        final_answer = None
        if len(answer) > 0:
            # 找最后一个非空数字
            for final_answer in reversed(answer):
                if final_answer not in ["", "."]:
                    break

    return final_answer
```

**示例评分**:
```python
# Response 0
solution_str = "She spent 5 * $3 = $15. So she has $23 - $15 = $8 left. #### 8"
ground_truth = "8"
extract_solution(solution_str) → "8"
compute_score(solution_str, ground_truth) → 1.0 ✓

# Response 3 (错误)
solution_str = "... 23-15=8. #### 7"
ground_truth = "8"
extract_solution(solution_str) → "7"
compute_score(solution_str, ground_truth) → 0.0 ✗

# Response 无格式
solution_str = "I don't know."
ground_truth = "8"
extract_solution(solution_str) → None
compute_score(solution_str, ground_truth) → 0.0 ✗
```

**最终奖励张量**:
```python
reward_tensor = torch.Tensor([1280, 1024])

# 对于第 0 个样本 (回答正确):
reward_tensor[0] = [0, 0, 0, ..., 0, 1.0]  # 只有最后一个有效 token 是 1.0

# 对于第 3 个样本 (回答错误):
reward_tensor[3] = [0, 0, 0, ..., 0, 0.0]  # 最后一个有效 token 是 0.0
```

---

### 🔍 奖励函数查找机制详解

#### 📋 整体流程

奖励函数的查找遵循以下优先级顺序:

```
1. 自定义奖励函数 (custom_reward_function)
   ↓ (如果没有)
2. 默认奖励函数 (default_compute_score)
   ↓ (根据 data_source 路由)
3. 具体任务的 compute_score 函数
```

---

#### 🎯 查找步骤 1: load_reward_manager

文件: `verl/trainer/ppo/reward.py:120-196`

```python
def load_reward_manager(config, tokenizer, num_examine, **reward_kwargs):
    """
    加载奖励管理器的主函数

    Args:
        config: 完整配置 (包含 reward_model, data 等)
        tokenizer: 分词器
        num_examine: 调试时打印的样本数
        **reward_kwargs: 额外的奖励函数参数

    Returns:
        AbstractRewardManager: 奖励管理器实例
    """

    # Step 1: 尝试获取自定义奖励函数
    compute_score = get_custom_reward_fn(config)

    # Step 2: 如果没有自定义函数,使用默认函数
    if compute_score is None:
        compute_score = default_compute_score

    # Step 3: 实例化奖励管理器 (默认是 NaiveRewardManager)
    reward_manager_cls = get_reward_manager_cls(config.reward_manager.name)

    return reward_manager_cls(
        tokenizer=tokenizer,
        num_examine=num_examine,
        compute_score=compute_score,
        reward_fn_key=config.data.reward_fn_key,  # 关键! 决定从哪里读取 data_source
        **reward_kwargs
    )
```

**关键参数**:
- `config.reward_manager.name`: 奖励管理器类型 (默认 `"naive"`)
- `config.data.reward_fn_key`: 从数据中读取哪个字段作为 data_source (默认 `"data_source"`)

---

#### 🎯 查找步骤 2: get_custom_reward_fn

文件: `verl/trainer/ppo/reward.py:63-118`

```python
def get_custom_reward_fn(config):
    """
    从外部文件加载自定义奖励函数

    Args:
        config: 配置字典,包含 custom_reward_function 字段

    Returns:
        callable or None: 自定义奖励函数 (如果配置了), 否则返回 None
    """
    reward_fn_config = config.get("custom_reward_function") or {}
    file_path = reward_fn_config.get("path")

    # 如果没有配置自定义函数,返回 None
    if not file_path:
        return None

    function_name = reward_fn_config.get("name")

    # 动态加载外部 Python 文件
    spec = importlib.util.spec_from_file_location("custom_module", file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    # 获取指定的函数
    raw_fn = getattr(module, function_name)

    # 合并 reward_kwargs (额外的参数)
    reward_kwargs = dict(reward_fn_config.get("reward_kwargs", {}))
    return partial(_call_with_kwargs, raw_fn, reward_kwargs)
```

**配置示例**:
```yaml
custom_reward_function:
  path: "/path/to/my_reward.py"
  name: "my_compute_score"
  reward_kwargs:
    threshold: 0.8
    bonus: 0.5
```

---

#### 🎯 查找步骤 3: default_compute_score

文件: `verl/utils/reward_score/__init__.py:19-115`

如果没有自定义函数,则使用 `default_compute_score`,它会根据 **data_source** 路由到不同的任务:

```python
def default_compute_score(data_source, solution_str, ground_truth, extra_info=None, **kwargs):
    """
    默认评分函数,根据数据源路由到不同的评分逻辑

    Args:
        data_source: 数据集名称 (例如 "openai/gsm8k")
        solution_str: 模型生成的回答文本
        ground_truth: 正确答案
        extra_info: 额外信息 (可选)

    Returns:
        float: 奖励分数
    """

    # GSM8K 数学题
    if data_source == "openai/gsm8k":
        from . import gsm8k
        res = gsm8k.compute_score(solution_str, ground_truth)

    # MATH 数据集
    elif data_source in ["lighteval/MATH", "DigitalLearningGmbH/MATH-lighteval"]:
        from . import math_reward
        res = math_reward.compute_score(solution_str, ground_truth)

    # 代码执行任务
    elif data_source in ["codecontests", "apps", "codeforces"]:
        from . import sandbox_fusion
        res = sandbox_fusion.compute_score(
            sandbox_url, solution_str, ground_truth, continuous=True
        )

    # 问答任务
    elif data_source in ["searchR1_nq", "searchR1_triviaqa"]:
        from . import search_r1_like_qa_em
        res = search_r1_like_qa_em.compute_score(solution_str, ground_truth)

    else:
        raise NotImplementedError(f"Reward function is not implemented for {data_source=}")

    return float(res)
```

**支持的数据集** (截至当前版本):

| 数据集分类 | data_source 值 | 评分函数文件 |
|----------|---------------|------------|
| 数学题 | `openai/gsm8k` | `gsm8k.py` |
| 数学题 | `lighteval/MATH`, `HuggingFaceH4/MATH-500` | `math_reward.py` |
| 数学题 | `math_dapo`, `aime*` | `math_dapo.py` |
| 几何题 | `hiyouga/geometry3k` | `geo3k.py` |
| 代码执行 | `codecontests`, `apps`, `codeforces`, `taco` | `sandbox_fusion.py` |
| 问答 | `searchR1_nq`, `searchR1_triviaqa`, `searchR1_popqa` | `search_r1_like_qa_em.py` |

---

#### 🎯 查找步骤 4: NaiveRewardManager 调用

文件: `verl/workers/reward_manager/naive.py:46-126`

```python
class NaiveRewardManager(AbstractRewardManager):
    def __init__(self, tokenizer, num_examine, compute_score=None, reward_fn_key="data_source"):
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.compute_score = compute_score or default_compute_score  # 关键!
        self.reward_fn_key = reward_fn_key  # 关键! 从哪个字段读取 data_source

    def __call__(self, data: DataProto, return_dict=False):
        """对每个样本调用 compute_score"""
        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)

        for i in range(len(data)):
            # 1. 解码 response
            response_str = self.tokenizer.decode(
                data[i].batch["responses"], skip_special_tokens=True
            )

            # 2. 获取 ground_truth
            ground_truth = data[i].non_tensor_batch["reward_model"]["ground_truth"]

            # 3. 获取 data_source (从 reward_fn_key 指定的字段读取)
            data_source = data[i].non_tensor_batch[self.reward_fn_key]

            # 4. 调用 compute_score
            score = self.compute_score(
                data_source=data_source,
                solution_str=response_str,
                ground_truth=ground_truth,
                extra_info=data[i].non_tensor_batch.get("extra_info", {})
            )

            # 5. 填充到 reward_tensor (outcome reward: 只有最后一个 token 有奖励)
            valid_response_length = data[i].batch["attention_mask"][prompt_length:].sum()
            reward_tensor[i, valid_response_length - 1] = score

        return reward_tensor
```

**关键逻辑**:
1. `reward_fn_key` 决定从哪里读取 `data_source`:
   - 默认是 `"data_source"` 字段
   - 也可以配置为 `"dataset_name"` 等其他字段
2. `data_source` 的值决定调用哪个 `compute_score` 函数
3. 奖励只给最后一个有效 token (outcome reward)

---

#### 🎯 data_source 从哪里来?

**从数据集中读取**:

以 GSM8K 为例 (文件: `data/gsm8k/train.parquet`):

```python
{
    "data_source": "openai/gsm8k",  # ← 这个字段!
    "prompt": "Olivia has $23. She bought five bagels for $3 each. How much money does she have left?",
    "reward_model": {
        "ground_truth": "8",
        "style": "step-by-step"
    }
}
```

**在数据加载时设置**:

文件: `verl/data/reward_dataset.py`

```python
def preprocess_item(item):
    """预处理单个数据样本"""
    return {
        "data_source": item.get("data_source", "openai/gsm8k"),  # 默认值
        "prompt": item["prompt"],
        "reward_model": item["reward_model"],
        # ...
    }
```

---

#### 🎯 完整调用链总结

```
┌─────────────────────────────────────────────────────────────┐
│  1. 启动脚本: run_qwen3-8b.sh                                │
│     → 设置 data.reward_fn_key="data_source"                 │
└─────────────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────────┐
│  2. main.py:274                                             │
│     → reward_fn = load_reward_manager(config, tokenizer)   │
└─────────────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────────┐
│  3. reward.py:120 load_reward_manager()                     │
│     → compute_score = get_custom_reward_fn(config)          │
│     → 如果为 None, 则 compute_score = default_compute_score │
│     → return NaiveRewardManager(compute_score=...)          │
└─────────────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────────┐
│  4. ray_trainer.py:1117                                     │
│     → reward_tensor = compute_reward(batch, self.reward_fn) │
└─────────────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────────┐
│  5. reward.py:200 compute_reward()                          │
│     → result = reward_fn(data)  # 调用 NaiveRewardManager   │
└─────────────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────────┐
│  6. naive.py:89 NaiveRewardManager.__call__()               │
│     → data_source = data[i].non_tensor_batch[reward_fn_key] │
│     → score = self.compute_score(                           │
│           data_source=data_source,  # "openai/gsm8k"        │
│           solution_str=response_str,                        │
│           ground_truth=ground_truth                         │
│       )                                                     │
└─────────────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────────┐
│  7. __init__.py:44 default_compute_score()                  │
│     → if data_source == "openai/gsm8k":                     │
│           from . import gsm8k                               │
│           res = gsm8k.compute_score(solution_str, ...)      │
└─────────────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────────┐
│  8. gsm8k.py:52 compute_score()                             │
│     → answer = extract_solution(solution_str)               │
│     → return 1.0 if answer == ground_truth else 0.0         │
└─────────────────────────────────────────────────────────────┘
```

---

---

### 🛠️ 如何修改奖励函数

根据你的需求,有 **三种** 修改奖励函数的方法:

---

#### 方法 1: 使用自定义奖励函数 (推荐)

适用场景: 完全自定义的评分逻辑,不想修改框架代码

**步骤 1: 创建奖励函数文件**

创建 `my_reward.py`:
```python
# my_reward.py
def my_compute_score(data_source, solution_str, ground_truth, extra_info=None, **kwargs):
    """
    自定义奖励函数

    Args:
        data_source: 数据集名称 (例如 "my_dataset")
        solution_str: 模型生成的文本
        ground_truth: 正确答案
        extra_info: 额外信息
        **kwargs: 其他参数 (例如 threshold, bonus 等)

    Returns:
        float: 奖励分数 (0.0 到 1.0)
    """
    # 示例 1: 精确匹配
    if solution_str.strip() == ground_truth.strip():
        return 1.0
    else:
        return 0.0

    # 示例 2: 代码执行
    # try:
    #     exec_result = execute_code(solution_str)
    #     return 1.0 if exec_result == ground_truth else 0.0
    # except:
    #     return 0.0

    # 示例 3: 使用额外参数
    # threshold = kwargs.get("threshold", 0.8)
    # similarity = compute_similarity(solution_str, ground_truth)
    # return 1.0 if similarity >= threshold else 0.0
```

**步骤 2: 修改配置**

编辑 `examples/grpo_qwen3/config/grpo_qwen3.yaml` 或在启动脚本中添加:

```yaml
custom_reward_function:
  path: "/path/to/my_reward.py"  # 绝对路径
  name: "my_compute_score"        # 函数名
  reward_kwargs:                  # 可选: 额外参数
    threshold: 0.8
    bonus: 0.5
```

或在 `run_qwen3-8b.sh` 中添加:
```bash
python3 -m verl.trainer.main_ppo \
    algorithm.grpo=true \
    custom_reward_function.path=/path/to/my_reward.py \
    custom_reward_function.name=my_compute_score \
    custom_reward_function.reward_kwargs.threshold=0.8
```

**优点**:
- ✅ 不需要修改框架代码
- ✅ 可以随意修改奖励函数,不影响其他任务
- ✅ 支持传递自定义参数

**缺点**:
- ❌ 需要提供绝对路径

---

#### 方法 2: 在 default_compute_score 中添加新的 data_source

适用场景: 你有一个新任务类型,想要永久添加到框架中

**步骤 1: 创建评分函数文件**

创建 `verl/utils/reward_score/my_task.py`:
```python
# verl/utils/reward_score/my_task.py

def compute_score(solution_str, ground_truth, **kwargs):
    """
    我的任务的评分函数

    Args:
        solution_str: 模型生成的文本
        ground_truth: 正确答案

    Returns:
        float: 奖励分数
    """
    # 你的评分逻辑
    if "正确" in solution_str:
        return 1.0
    else:
        return 0.0
```

**步骤 2: 注册到 default_compute_score**

编辑 `verl/utils/reward_score/__init__.py`:
```python
def default_compute_score(
    data_source,
    solution_str,
    ground_truth,
    extra_info=None,
    **kwargs,
):
    # ... 现有代码 ...

    # 添加你的任务
    elif data_source == "my_custom_task":
        from . import my_task
        res = my_task.compute_score(solution_str, ground_truth)

    else:
        raise NotImplementedError(f"Reward function is not implemented for {data_source=}")

    return float(res)
```

**步骤 3: 修改数据集和配置**

确保你的数据集中包含正确的 `data_source`:
```json
{
    "data_source": "my_custom_task",  # ← 必须匹配!
    "prompt": "你的问题",
    "reward_model": {
        "ground_truth": "正确答案"
    }
}
```

在 `run_qwen3-8b.sh` 中确保:
```bash
data.reward_fn_key=data_source  # 使用默认值
```

**优点**:
- ✅ 成为框架的一部分,可以复用
- ✅ 其他人也可以使用你的评分函数

**缺点**:
- ❌ 需要修改框架代码
- ❌ 需要确保 data_source 字段一致

---

#### 方法 3: 修改现有的评分函数

适用场景: 你想调整 GSM8K、MATH 等现有任务的评分逻辑

**步骤: 直接修改对应的文件**

例如修改 GSM8K 的评分:

编辑 `verl/utils/reward_score/gsm8k.py`:
```python
def compute_score(solution_str, ground_truth, method="strict", format_score=0.0, score=1.0):
    """修改后的 GSM8K 评分函数"""

    # 原始逻辑
    answer = extract_solution(solution_str=solution_str, method=method)

    if answer is None:
        return 0

    # 修改 1: 给格式正确但答案错误的一些分数
    if answer == ground_truth:
        return score  # 1.0
    else:
        return format_score  # 原来是 0.0, 可以改成 0.3

    # 修改 2: 添加部分分机制
    # if answer == ground_truth:
    #     return 1.0
    # elif is_close_to_answer(answer, ground_truth):
    #     return 0.5  # 部分分
    # else:
    #     return 0.0
```

然后在配置中可以传递参数:
```bash
python3 -m verl.trainer.main_ppo \
    algorithm.grpo=true \
    reward_model.reward_kwargs.format_score=0.3 \
    reward_model.reward_kwargs.score=1.0
```

**优点**:
- ✅ 简单直接
- ✅ 可以利用现有的 extract_solution 等辅助函数

**缺点**:
- ❌ 会影响所有使用该评分函数的任务
- ❌ 需要小心,避免破坏原有逻辑

---

#### 📊 三种方法对比

| 特性 | 方法 1: 自定义函数 | 方法 2: 添加 data_source | 方法 3: 修改现有函数 |
|-----|------------------|----------------------|-------------------|
| 修改框架代码 | ❌ 不需要 | ✅ 需要 | ✅ 需要 |
| 适用场景 | 临时实验 | 新任务类型 | 调整现有任务 |
| 可复用性 | ❌ 低 | ✅ 高 | ⚠️ 中 |
| 灵活性 | ✅ 高 | ✅ 高 | ⚠️ 中 |
| 推荐度 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ |

---

#### 💡 实际案例: 数学竞赛任务

假设你要做 AMC 数学竞赛,需要:
- 答案格式: `\boxed{数字}`
- 评分: 正确 1.0, 错误 0.0, 格式错误 -0.1

**使用方法 1 (推荐)**:

创建 `amc_reward.py`:
```python
import re

def compute_amc_score(data_source, solution_str, ground_truth, extra_info=None, **kwargs):
    """AMC 数学竞赛评分"""
    # 提取 \boxed{...} 中的答案
    match = re.search(r'\\boxed\{([^}]+)\}', solution_str)

    if match is None:
        return -0.1  # 格式错误

    answer = match.group(1).strip()

    if answer == ground_truth:
        return 1.0
    else:
        return 0.0
```

配置:
```bash
python3 -m verl.trainer.main_ppo \
    algorithm.grpo=true \
    custom_reward_function.path=/path/to/amc_reward.py \
    custom_reward_function.name=compute_amc_score
```

---

#### 🐛 调试技巧

**1. 打印奖励分数**

在 `verl/workers/reward_manager/naive.py:109-118` 中,设置 `num_examine=5`:
```bash
python3 -m verl.trainer.main_ppo \
    algorithm.grpo=true \
    reward_model.reward_kwargs.num_examine=5
```

这会打印前 5 个样本的:
- `[prompt]`: 输入问题
- `[response]`: 模型生成的回答
- `[ground_truth]`: 正确答案
- `[score]`: 计算的奖励

**2. 验证 data_source**

在数据集加载后打印:
```python
# 在 verl/data/reward_dataset.py 中添加
print(f"Sample data_source: {item['data_source']}")
```

**3. 测试奖励函数**

独立测试:
```python
from verl.utils.reward_score.gsm8k import compute_score

solution = "The answer is 8. #### 8"
ground_truth = "8"
score = compute_score(solution, ground_truth)
print(f"Score: {score}")  # 应该是 1.0
```

---

### Step 5: 计算 old_log_probs

#### 📁 文件链
```
verl/trainer/ppo/ray_trainer.py:1145-1159
  ↓ (RPC 调用)
verl/workers/fsdp_workers.py:961-998
  ↓
verl/workers/actor/dp_actor.py:180-250
```

#### 🎯 作用
计算当前策略对已生成序列的 log 概率，作为 PPO 的"旧策略"。

#### 📝 代码详解

**为什么需要 old_log_probs?**

PPO 算法需要两个策略:
- **π_old** (旧策略): 用于 PPO clip 的参考点，保证更新不要太激进
- **π_θ** (新策略): 当前正在更新的策略

**计算流程**:
```
已生成的序列 → 当前 Actor 模型 → 计算 log P(token | prefix) → old_log_probs
```

---

**调用 compute_log_prob** (ray_trainer.py:1145-1159):
```python
else:  # 非 bypass 模式 (标准流程)
    with marked_timer("old_log_prob", timing_raw, color="blue"):
        # 1. 调用 Actor Worker 计算 log_prob
        old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)

        # 2. 提取 entropy (用于 entropy bonus)
        entropys = old_log_prob.batch["entropys"]
        response_masks = batch.batch["response_mask"]
        actor_config = self.config.actor_rollout_ref.actor

        # 3. 聚合 entropy (平均值)
        entropy_agg = agg_loss(
            loss_mat=entropys,
            loss_mask=response_masks,
            loss_agg_mode=actor_config.loss_agg_mode,
            loss_scale_factor=actor_config.loss_scale_factor,
        )
        old_log_prob_metrics = {"actor/entropy": entropy_agg.detach().item()}
        metrics.update(old_log_prob_metrics)

        # 4. 移除 entropy, 保留 old_log_probs
        old_log_prob.batch.pop("entropys")

        # 5. 合并到 batch
        batch = batch.union(old_log_prob)
```

---

**Worker 的 compute_log_prob** (fsdp_workers.py:961-998):
```python
@register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="actor"))
@DistProfiler.annotate(color="blue", role="actor_compute_log_prob")
def compute_log_prob(self, data: DataProto):
    """
    计算 log 概率

    Args:
        data: 包含 input_ids, responses, response_mask

    Returns:
        DataProto: 包含 old_log_probs, entropys
    """
    # 1. 如果使用 parameter offload, 先加载模型到 GPU
    if self._is_offload_param:
        load_fsdp_model_to_gpu(self.actor_module_fsdp)

    # 2. 设置 meta_info (控制 micro_batch 大小)
    data.meta_info["micro_batch_size"] = self.config.rollout.log_prob_micro_batch_size_per_gpu
    data.meta_info["max_token_len"] = self.config.rollout.log_prob_max_token_len_per_gpu
    data.meta_info["use_dynamic_bsz"] = self.config.rollout.log_prob_use_dynamic_bsz
    data.meta_info["temperature"] = self.config.rollout.temperature

    # 3. 调用 actor.compute_log_prob
    with self.ulysses_sharding_manager:
        with adapter_ctx:  # 如果是 LoRA, 可能需要禁用 adapter
            output, entropys = self.actor.compute_log_prob(data=data, calculate_entropy=True)

        output = DataProto.from_dict(
            tensors={"old_log_probs": output, "entropys": entropys},
            meta_info={"temperature": self.config.rollout.temperature},
        )

    # 4. 移回 CPU
    output = output.to("cpu")

    # 5. 如果使用 offload, 卸载模型
    if self._is_offload_param:
        offload_fsdp_model_to_cpu(self.actor_module_fsdp)
        log_gpu_memory_usage("After offload actor model during compute_log_prob", logger=logger)

    return output
```

---

**Actor 的 compute_log_prob** (dp_actor.py:180-250, 简化版):
```python
def compute_log_prob(self, data: DataProto, calculate_entropy: bool = False):
    """
    计算 log 概率的核心函数

    流程:
    1. 合并 input_ids 和 responses 为完整序列
    2. 模型前向传播得到 logits
    3. 从 logits 计算 log_probs
    4. 只保留 response 部分的 log_probs
    """
    # 1. 准备数据
    data = data.to(get_device_id())
    input_ids = data.batch["input_ids"]         # (bsz, prompt_len)
    responses = data.batch["responses"]          # (bsz, response_len)
    response_mask = data.batch["response_mask"]  # (bsz, response_len)
    temperature = data.meta_info["temperature"]

    # 2. 合并 input 和 response
    full_input_ids = torch.cat([input_ids, responses], dim=1)
    # full_input_ids: (bsz, prompt_len + response_len)

    # 3. 计算 attention_mask
    full_attention_mask = torch.cat([
        data.batch["attention_mask"],
        response_mask
    ], dim=1)

    # 4. 分 micro-batch 处理 (避免 OOM)
    micro_batch_size = data.meta_info["micro_batch_size"]  # 32
    all_log_probs = []
    all_entropys = [] if calculate_entropy else None

    for i in range(0, len(full_input_ids), micro_batch_size):
        micro_input_ids = full_input_ids[i:i+micro_batch_size]
        micro_attention_mask = full_attention_mask[i:i+micro_batch_size]

        # 5. 模型前向传播
        with torch.no_grad():
            outputs = self.actor_module(
                input_ids=micro_input_ids[:, :-1],  # 去掉最后一个 token (teacher forcing)
                attention_mask=micro_attention_mask[:, :-1],
            )
            logits = outputs.logits  # (micro_bsz, seq_len, vocab_size)

        # 6. 计算 log_probs
        # logprobs_from_logits: 从 logits 和 labels 计算 log P(label | prefix)
        log_probs = logprobs_from_logits(
            logits=logits,
            labels=micro_input_ids[:, 1:],  # 去掉第一个 token (shifted labels)
            temperature=temperature,
        )  # (micro_bsz, seq_len)

        # 7. 只保留 response 部分
        prompt_len = input_ids.size(1)
        response_log_probs = log_probs[:, prompt_len:]  # (micro_bsz, response_len)
        all_log_probs.append(response_log_probs)

        # 8. 计算 entropy (如果需要)
        if calculate_entropy:
            entropys = self.compute_entropy_from_logits(logits, temperature)
            response_entropys = entropys[:, prompt_len:]
            all_entropys.append(response_entropys)

    # 9. 合并所有 micro-batch
    log_probs = torch.cat(all_log_probs, dim=0)  # (bsz, response_len)
    entropys = torch.cat(all_entropys, dim=0) if calculate_entropy else None

    return log_probs, entropys
```

**logprobs_from_logits 函数** (简化):
```python
def logprobs_from_logits(logits, labels, temperature=1.0):
    """
    从 logits 计算 log P(labels | prefix)

    Args:
        logits: (bsz, seq_len, vocab_size)
        labels: (bsz, seq_len)
        temperature: 温度参数

    Returns:
        log_probs: (bsz, seq_len)
    """
    # 1. 温度缩放
    logits = logits / temperature

    # 2. 计算 log softmax
    log_probs_all = torch.log_softmax(logits, dim=-1)  # (bsz, seq_len, vocab_size)

    # 3. 选择 labels 对应的 log_prob
    log_probs = torch.gather(
        log_probs_all,
        dim=-1,
        index=labels.unsqueeze(-1)
    ).squeeze(-1)  # (bsz, seq_len)

    return log_probs
```

**示例**:
```python
# 假设 vocab_size = 50000

# 输入
input_ids = [[token_1, token_2, ..., token_512]]  # prompt
responses = [[token_513, token_514, ..., token_1536]]  # response
full_ids = [[token_1, ..., token_512, token_513, ..., token_1536]]

# 前向传播
logits = model(full_ids[:, :-1])  # shape: (1, 1535, 50000)

# 计算 log_probs
log_probs = logprobs_from_logits(logits, full_ids[:, 1:])  # shape: (1, 1535)

# 只保留 response 部分
response_log_probs = log_probs[:, 512:]  # shape: (1, 1024)

# 示例值
response_log_probs[0] = [
    -2.5,   # log P(token_513 | token_1...token_512)
    -1.8,   # log P(token_514 | token_1...token_513)
    -3.2,   # log P(token_515 | token_1...token_514)
    ...
]
```

**最终输出**:
```python
old_log_prob = DataProto(
    batch={
        "old_log_probs": torch.Tensor([1280, 1024]),  # 每个 token 的 log P
        "entropys": torch.Tensor([1280, 1024]),       # 每个 token 的 entropy
    }
)
```

#### 🔧 修改点
- **修改 micro_batch_size**: `run_qwen3-8b.sh:27` → `actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=64` (增大以加速)
- **禁用 entropy 计算**: 需要在代码中修改 `calculate_entropy=False` (如果不需要 entropy bonus)

---

### Step 6: 计算 GRPO 优势

#### 📁 文件链
```
verl/trainer/ppo/ray_trainer.py:1222-1230
  ↓
verl/trainer/ppo/ray_trainer.py:181-259 (compute_advantage)
  ↓
verl/trainer/ppo/core_algos.py:265-328 (compute_grpo_outcome_advantage)
```

#### 🎯 作用
**GRPO 的核心**: 计算组内相对优势，让模型学会区分好坏回答。

#### 📝 代码详解

**调用 compute_advantage** (ray_trainer.py:1222-1230):
```python
with marked_timer("adv", timing_raw, color="brown"):
    # 1. 获取 reward_tensor (如果异步)
    if self.config.reward_model.launch_reward_fn_async:
        reward_tensor, reward_extra_infos_dict = ray.get(future_reward)
    batch.batch["token_level_scores"] = reward_tensor

    # 2. 应用 KL penalty (如果配置)
    if self.config.algorithm.use_kl_in_reward:
        batch, kl_metrics = apply_kl_penalty(
            batch, kl_ctrl=self.kl_ctrl_in_reward, kl_penalty=self.config.algorithm.kl_penalty
        )
        metrics.update(kl_metrics)
    else:
        # 直接使用 scores 作为 rewards
        batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]

    # 3. 计算优势
    norm_adv_by_std_in_grpo = self.config.algorithm.get("norm_adv_by_std_in_grpo", True)

    batch = compute_advantage(
        batch,
        adv_estimator=self.config.algorithm.adv_estimator,  # "grpo"
        gamma=self.config.algorithm.gamma,                  # 1.0
        lam=self.config.algorithm.lam,                      # 1.0
        num_repeat=self.config.actor_rollout_ref.rollout.n,  # 5
        norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,   # True
        config=self.config.algorithm,
    )
```

---

**compute_advantage 函数** (ray_trainer.py:181-259):
```python
def compute_advantage(
    data: DataProto,
    adv_estimator: AdvantageEstimator,
    gamma: float = 1.0,
    lam: float = 1.0,
    num_repeat: int = 1,
    norm_adv_by_std_in_grpo: bool = True,
    config: Optional[AlgoConfig] = None,
) -> DataProto:
    """
    计算优势的统一接口

    根据 adv_estimator 选择不同的优势计算方法:
    - GAE: 需要 value 函数
    - GRPO: 组内相对优势
    - REINFORCE++: 无 baseline
    - 等等
    """
    # 1. 确保有 response_mask
    if "response_mask" not in data.batch.keys():
        data.batch["response_mask"] = compute_response_mask(data)

    # 2. 根据 adv_estimator 选择方法
    if adv_estimator == AdvantageEstimator.GAE:
        # GAE: 需要 Critic 网络
        advantages, returns = core_algos.compute_gae_advantage_return(
            token_level_rewards=data.batch["token_level_rewards"],
            values=data.batch["values"],  # 来自 Critic
            response_mask=data.batch["response_mask"],
            gamma=gamma,
            lam=lam,
        )

    elif adv_estimator == AdvantageEstimator.GRPO:
        # GRPO: 组内相对优势
        grpo_calculation_mask = data.batch["response_mask"]

        advantages, returns = core_algos.compute_grpo_outcome_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            response_mask=grpo_calculation_mask,
            index=data.non_tensor_batch["uid"],  # ← 用于分组!
            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
        )

    else:
        # 其他优势估计器
        adv_estimator_fn = core_algos.get_adv_estimator_fn(adv_estimator)
        advantages, returns = adv_estimator_fn(...)

    # 3. 添加到 batch
    data.batch["advantages"] = advantages
    data.batch["returns"] = returns

    return data
```

---

**GRPO 核心算法** (core_algos.py:265-328):
```python
def compute_grpo_outcome_advantage(
    token_level_rewards: torch.Tensor,  # (1280, 1024)
    response_mask: torch.Tensor,        # (1280, 1024)
    index: np.ndarray,                   # (1280,) - uid 数组
    epsilon: float = 1e-6,
    norm_adv_by_std_in_grpo: bool = True,
    config: Optional[AlgoConfig] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    GRPO 优势计算

    核心思想:
    1. 计算每个回答的总分: score = sum(token_level_rewards)
    2. 按 uid 分组: 同一个 prompt 的 G 个回答分到一组
    3. 计算组内均值和标准差: mean_g, std_g
    4. 标准化优势: advantage_i = (score_i - mean_g) / std_g

    这样:
    - 高于平均分的回答 → 正优势 → 增加概率
    - 低于平均分的回答 → 负优势 → 降低概率
    """
    # 1. 计算每个回答的总分
    scores = token_level_rewards.sum(dim=-1)  # (1280,)
    # scores[i] = sum of all token rewards for response i

    # 2. 初始化分组字典
    id2score = defaultdict(list)  # {uid: [score1, score2, ...]}
    id2mean = {}                   # {uid: mean_score}
    id2std = {}                    # {uid: std_score}

    with torch.no_grad():
        bsz = scores.shape[0]  # 1280

        # 3. 按 uid 分组
        for i in range(bsz):
            id2score[index[i]].append(scores[i])

        # 现在 id2score 的结构:
        # {
        #   "uuid-0": [score_0, score_1, score_2, score_3, score_4],  # 5 个分数
        #   "uuid-1": [score_5, score_6, score_7, score_8, score_9],
        #   ...
        #   "uuid-255": [score_1275, score_1276, score_1277, score_1278, score_1279]
        # }

        # 4. 计算每组的均值和标准差
        for idx in id2score:
            if len(id2score[idx]) == 1:
                # 只有 1 个样本, 无法计算 std
                id2mean[idx] = torch.tensor(0.0)
                id2std[idx] = torch.tensor(1.0)
            elif len(id2score[idx]) > 1:
                # 有多个样本
                scores_tensor = torch.stack(id2score[idx])  # (5,)
                id2mean[idx] = torch.mean(scores_tensor)    # 标量
                id2std[idx] = torch.std(scores_tensor)      # 标量
            else:
                raise ValueError(f"no score in prompt index: {idx}")

        # 5. 标准化每个分数
        for i in range(bsz):
            if norm_adv_by_std_in_grpo:
                # 标准 GRPO: (score - mean) / std
                scores[i] = (scores[i] - id2mean[index[i]]) / (id2std[index[i]] + epsilon)
            else:
                # Dr.GRPO: score - mean (不除以标准差)
                # 参考论文: https://arxiv.org/abs/2503.20783
                scores[i] = scores[i] - id2mean[index[i]]

        # 6. 广播到所有 token
        # scores 现在是 (1280,), 需要扩展到 (1280, 1024)
        # 每个 token 的 advantage 都是同一个值 (因为是 outcome reward)
        scores = scores.unsqueeze(-1) * response_mask  # (1280, 1024)
        # unsqueeze(-1): (1280,) → (1280, 1)
        # * response_mask: 广播乘法, 只保留有效 token

    # 7. 返回 advantages 和 returns (对于 GRPO, 两者相同)
    return scores, scores
```

---

**具体计算示例**:

假设 `uuid-0` 的 5 个回答的分数:
```python
# 原始分数 (来自 Step 4 的奖励计算)
scores = [1.0, 1.0, 1.0, 0.0, 1.0]

# 1. 计算组内均值
mean = (1.0 + 1.0 + 1.0 + 0.0 + 1.0) / 5 = 0.8

# 2. 计算组内标准差
# std = sqrt(E[(x - mean)^2])
variance = ((1.0-0.8)^2 + (1.0-0.8)^2 + (1.0-0.8)^2 + (0.0-0.8)^2 + (1.0-0.8)^2) / 5
         = (0.04 + 0.04 + 0.04 + 0.64 + 0.04) / 5
         = 0.8 / 5
         = 0.16
std = sqrt(0.16) = 0.4

# 3. 标准化每个分数
adv_0 = (1.0 - 0.8) / 0.4 = 0.2 / 0.4 = 0.5
adv_1 = (1.0 - 0.8) / 0.4 = 0.2 / 0.4 = 0.5
adv_2 = (1.0 - 0.8) / 0.4 = 0.2 / 0.4 = 0.5
adv_3 = (0.0 - 0.8) / 0.4 = -0.8 / 0.4 = -2.0  # ← 负优势!
adv_4 = (1.0 - 0.8) / 0.4 = 0.2 / 0.4 = 0.5

# 验证零和性质
sum(advantages) = 0.5 + 0.5 + 0.5 + (-2.0) + 0.5 = 0 ✓
```

**关键性质**:
1. **零和**: 同一组内的 advantages 之和为 0
2. **相对性**: 优势是相对于组内其他回答
3. **方差归一化**: 除以 std 使得优势在合理范围内 (通常 [-3, 3])

**为什么 GRPO 有效?**
- **好回答** (score > mean): 得到**正优势** → PPO 会**增加**其生成概率
- **坏回答** (score < mean): 得到**负优势** → PPO 会**降低**其生成概率
- **不需要 Value 函数**: 直接用组内对比替代 baseline

---

**最终输出**:
```python
batch.batch["advantages"] = torch.Tensor([1280, 1024])
batch.batch["returns"] = torch.Tensor([1280, 1024])  # 对于 GRPO, 与 advantages 相同

# 示例值 (第 0 个 prompt 的 5 个回答)
batch.batch["advantages"][0:5, -1] = [0.5, 0.5, 0.5, -2.0, 0.5]  # 最后一个 token
# 其他 token 位置的 advantage 也是相同值 (广播)
```

#### 🔧 修改点

**禁用标准差归一化** (使用 Dr.GRPO):
```bash
# run_qwen3-8b.sh
algorithm.norm_adv_by_std_in_grpo=False
```

**效果**:
- 标准 GRPO: `advantage = (score - mean) / std`
- Dr.GRPO: `advantage = score - mean`
- Dr.GRPO 可能在某些任务上更稳定 (避免 std 过小导致爆炸)

---

### Step 7: 更新 Actor (PPO)

#### 📁 文件链
```
verl/trainer/ppo/ray_trainer.py:1241-1249
  ↓ (RPC 调用)
verl/workers/fsdp_workers.py:868-907
  ↓
verl/workers/actor/dp_actor.py:398-600
  ↓
verl/trainer/ppo/core_algos.py:907-996 (compute_policy_loss_vanilla)
```

#### 🎯 作用
使用 PPO 算法更新策略，增加好回答概率，降低坏回答概率。

#### 📝 代码详解

**调用 update_actor** (ray_trainer.py:1241-1249):
```python
# 实现 critic warmup (GRPO 通常设为 0)
if self.config.trainer.critic_warmup <= self.global_steps:
    with marked_timer("update_actor", timing_raw, color="red"):
        # 设置 meta_info
        rollout_config = self.config.actor_rollout_ref.rollout
        batch.meta_info["multi_turn"] = rollout_config.multi_turn.enable
        batch.meta_info["temperature"] = rollout_config.temperature

        # 调用 Worker 更新
        actor_output = self.actor_rollout_wg.update_actor(batch)

    # 收集 metrics
    actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
    metrics.update(actor_output_metrics)
```

---

**Worker 的 update_actor** (fsdp_workers.py:868-907):
```python
def update_actor(self, data: DataProto):
    """
    Worker 层的 update_actor

    主要职责:
    1. 管理模型和优化器的加载/卸载 (如果使用 offload)
    2. 调用 actor.update_policy 进行实际训练
    3. 记录性能指标 (MFU, 内存使用)
    4. 更新学习率
    """
    assert self._is_actor

    # 1. 加载模型和优化器到 GPU (如果使用 offload)
    if self._is_offload_param:
        load_fsdp_model_to_gpu(self.actor_module_fsdp)
    if self._is_offload_optimizer:
        load_fsdp_optimizer(optimizer=self.actor_optimizer, device_id=get_device_id())

    with self.ulysses_sharding_manager:
        # 2. 数据移回 CPU (会在 micro batch 时移到 GPU)
        data = data.to("cpu")

        # 3. 调用 actor.update_policy
        with Timer(name="update_policy", logger=None) as timer:
            metrics = self.actor.update_policy(data=data)
        delta_time = timer.last

        # 4. 计算 MFU (Model FLOPs Utilization)
        global_num_tokens = data.meta_info["global_token_num"]
        estimated_flops, promised_flops = self.flops_counter.estimate_flops(
            global_num_tokens, delta_time
        )
        metrics["perf/mfu/actor"] = (
            estimated_flops * self.config.actor.ppo_epochs / promised_flops / self.world_size
        )

        # 5. 记录内存使用
        metrics["perf/max_memory_allocated_gb"] = get_torch_device().max_memory_allocated() / (1024**3)
        metrics["perf/max_memory_reserved_gb"] = get_torch_device().max_memory_reserved() / (1024**3)
        metrics["perf/cpu_memory_used_gb"] = psutil.virtual_memory().used / (1024**3)

        # 6. 学习率调度
        lr = self.actor_lr_scheduler.get_last_lr()[0]
        metrics["actor/lr"] = lr.item() if torch.is_tensor(lr) else lr
        self.actor_lr_scheduler.step()

        output = DataProto(meta_info={"metrics": metrics})
        output = output.to("cpu")

    # 7. 卸载模型和优化器 (如果使用 offload)
    if self._is_offload_param:
        offload_fsdp_model_to_cpu(self.actor_module_fsdp)
    if self._is_offload_optimizer:
        offload_fsdp_optimizer(optimizer=self.actor_optimizer)

    return output
```

---

**Actor 的 update_policy** (dp_actor.py:398-600, 核心部分):
```python
def update_policy(self, data: DataProto):
    """
    PPO 策略更新的核心函数

    流程:
    1. 分 mini-batch (256 per batch)
    2. 分 micro-batch (32 per batch)
    3. 对每个 micro-batch:
       a. 前向传播计算 new_log_prob
       b. 计算 PPO Loss
       c. 反向传播
       d. 累积梯度
    4. 梯度裁剪
    5. 优化器步进
    """
    # 1. 确保模型在训练模式
    self.actor_module.train()

    temperature = data.meta_info["temperature"]

    # 2. 选择需要的字段
    select_keys = [
        "responses",
        "response_mask",
        "input_ids",
        "attention_mask",
        "position_ids",
        "old_log_probs",  # ← 来自 Step 5
        "advantages",     # ← 来自 Step 6
    ]
    if self.config.use_kl_loss:
        select_keys.append("ref_log_prob")

    data = data.select(batch_keys=select_keys)

    # 3. 分 mini-batch
    # PPO 论文: https://arxiv.org/abs/1707.06347
    mini_batches = data.split(self.config.ppo_mini_batch_size)
    # 1280 / 256 = 5 个 mini-batch

    on_policy = len(mini_batches) == 1 and self.config.ppo_epochs == 1

    metrics = {}

    # 4. PPO epochs (通常为 1)
    for _ in range(self.config.ppo_epochs):
        for batch_idx, mini_batch in enumerate(mini_batches):
            # 5. 分 micro-batch (用于梯度累积)
            self.gradient_accumulation = (
                self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu
            )
            micro_batches = mini_batch.split(self.config.ppo_micro_batch_size_per_gpu)
            # 256 / 32 = 8 个 micro-batch

            # 6. 清零梯度
            self.actor_optimizer.zero_grad()

            # 7. 遍历 micro-batch
            for micro_batch in micro_batches:
                micro_batch = micro_batch.to(get_device_id())
                micro_batch_metrics = {}

                model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}
                response_mask = model_inputs["response_mask"]   # (32, 1024)
                old_log_prob = model_inputs["old_log_probs"]    # (32, 1024)
                advantages = model_inputs["advantages"]          # (32, 1024)

                # 8. 计算 loss_scale_factor (用于梯度累积)
                loss_scale_factor = 1 / self.gradient_accumulation  # 1/8

                # 9. 前向传播
                entropy, log_prob = self._forward_micro_batch(
                    model_inputs, temperature=temperature, calculate_entropy=True
                )
                # log_prob: (32, 1024)
                # entropy: (32, 1024)

                # 10. 如果是 on-policy, 直接用当前 log_prob 作为 old
                if on_policy:
                    old_log_prob = log_prob.detach()
                else:
                    old_log_prob = model_inputs["old_log_probs"]

                # 11. 计算 policy loss
                loss_mode = self.config.policy_loss.get("loss_mode", "vanilla")
                policy_loss_fn = get_policy_loss_fn(loss_mode)

                pg_loss, pg_metrics = policy_loss_fn(
                    old_log_prob=old_log_prob,
                    log_prob=log_prob,
                    advantages=advantages,
                    response_mask=response_mask,
                    loss_agg_mode=self.config.loss_agg_mode,
                    config=self.config,
                )
                # pg_loss 是标量
                micro_batch_metrics.update(pg_metrics)

                policy_loss = pg_loss

                # 12. 添加 entropy loss (如果配置)
                if self.config.entropy_coeff != 0:
                    entropy_agg = agg_loss(
                        loss_mat=entropy,
                        loss_mask=response_mask,
                        loss_agg_mode=self.config.loss_agg_mode
                    )
                    policy_loss -= entropy_agg * self.config.entropy_coeff

                # 13. 添加 KL loss (如果配置)
                if self.config.use_kl_loss:
                    ref_log_prob = model_inputs["ref_log_prob"]
                    kld = kl_penalty(
                        logprob=log_prob,
                        ref_logprob=ref_log_prob,
                        kl_penalty=self.config.kl_loss_type  # "low_var_kl"
                    )
                    kl_loss = agg_loss(
                        loss_mat=kld,
                        loss_mask=response_mask,
                        loss_agg_mode=self.config.loss_agg_mode
                    )
                    policy_loss = policy_loss + kl_loss * self.config.kl_loss_coef
                    micro_batch_metrics["actor/kl_loss"] = kl_loss.detach().item() * loss_scale_factor

                # 14. 缩放 loss (用于梯度累积)
                loss = policy_loss * loss_scale_factor

                # 15. 反向传播
                if self.scaler is not None:  # 混合精度训练
                    self.scaler.scale(loss).backward()
                else:
                    loss.backward()

                # 16. 记录 metrics
                micro_batch_metrics["actor/pg_loss"] = pg_loss.detach().item() * loss_scale_factor
                append_to_dict(metrics, micro_batch_metrics)

            # 17. 梯度裁剪
            if self.config.max_grad_norm is not None:
                if self.scaler is not None:
                    self.scaler.unscale_(self.actor_optimizer)

                if isinstance(self.actor_module, FSDPModule) and self.actor_module.fsdp2:
                    total_norm = fsdp2_clip_grad_norm_(
                        self.actor_module, self.config.max_grad_norm
                    )
                else:
                    total_norm = torch.nn.utils.clip_grad_norm_(
                        self.actor_module.parameters(), self.config.max_grad_norm
                    )
                metrics["actor/grad_norm"] = [total_norm.item()]

            # 18. 优化器步进
            if self.scaler is not None:
                self.scaler.step(self.actor_optimizer)
                self.scaler.update()
            else:
                self.actor_optimizer.step()

    # 19. 聚合 metrics
    for key, value_list in metrics.items():
        if isinstance(value_list, list):
            metrics[key] = sum(value_list) / len(value_list)

    return metrics
```

---

**PPO Loss 计算** (core_algos.py:907-996):
```python
@register_policy_loss("vanilla")
def compute_policy_loss_vanilla(
    old_log_prob: torch.Tensor,    # (32, 1024)
    log_prob: torch.Tensor,         # (32, 1024)
    advantages: torch.Tensor,       # (32, 1024)
    response_mask: torch.Tensor,    # (32, 1024)
    loss_agg_mode: str = "token-mean",
    config: Optional[ActorConfig] = None,
    rollout_is_weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """
    标准 PPO Clip Loss

    PPO 论文: https://arxiv.org/abs/1707.06347

    核心思想:
    L_CLIP(θ) = -E[min(ratio * A, clip(ratio, 1-ε, 1+ε) * A)]

    其中:
    - ratio = π_θ(a|s) / π_old(a|s) = exp(log_prob - old_log_prob)
    - A = advantages
    - ε = clip_epsilon (默认 0.2)
    """
    assert config is not None

    # 1. 获取 clip 参数
    clip_ratio = config.clip_ratio  # 0.2
    clip_ratio_low = config.clip_ratio_low if config.clip_ratio_low is not None else clip_ratio
    clip_ratio_high = config.clip_ratio_high if config.clip_ratio_high is not None else clip_ratio
    clip_ratio_c = config.get("clip_ratio_c", 3.0)  # dual-clip PPO

    cliprange = clip_ratio
    cliprange_low = clip_ratio_low
    cliprange_high = clip_ratio_high

    # 2. 计算 ratio = π_θ / π_old
    negative_approx_kl = log_prob - old_log_prob  # (32, 1024)
    # Clamp for stability
    negative_approx_kl = torch.clamp(negative_approx_kl, min=-20.0, max=20.0)
    ratio = torch.exp(negative_approx_kl)  # (32, 1024)
    # ratio[i, j] = P_new(token_j | prefix) / P_old(token_j | prefix)

    # 3. 计算 KL divergence (用于监控)
    ppo_kl = verl_F.masked_mean(-negative_approx_kl, response_mask)

    # 4. 计算两个 surrogate
    pg_losses1 = -advantages * ratio  # (32, 1024)
    # 第一个 surrogate: -A * ratio

    pg_losses2 = -advantages * torch.clamp(
        ratio, 1 - cliprange_low, 1 + cliprange_high
    )  # (32, 1024)
    # 第二个 surrogate: -A * clip(ratio, 1-ε, 1+ε)

    # 5. 取 maximum (因为有负号, 实际是取 minimum)
    clip_pg_losses1 = torch.maximum(pg_losses1, pg_losses2)
    # clip_pg_losses1 = max(-A*ratio, -A*clip(ratio))
    #                 = -min(A*ratio, A*clip(ratio))

    # 6. 计算 clip fraction (用于监控)
    pg_clipfrac = verl_F.masked_mean(
        torch.gt(pg_losses2, pg_losses1).float(), response_mask
    )

    # 7. Dual-clip PPO (处理负 advantage)
    pg_losses3 = -advantages * clip_ratio_c  # (32, 1024)
    clip_pg_losses2 = torch.min(pg_losses3, clip_pg_losses1)
    pg_clipfrac_lower = verl_F.masked_mean(
        torch.gt(clip_pg_losses1, pg_losses3) * (advantages < 0).float(), response_mask
    )

    # 8. 根据 advantage 符号选择 loss
    pg_losses = torch.where(advantages < 0, clip_pg_losses2, clip_pg_losses1)

    # 9. 应用 rollout correction weights (如果有)
    if rollout_is_weights is not None:
        pg_losses = pg_losses * rollout_is_weights

    # 10. 聚合 loss
    pg_loss = agg_loss(
        loss_mat=pg_losses,
        loss_mask=response_mask,
        loss_agg_mode=loss_agg_mode,  # "token-mean"
        **config.global_batch_info
    )

    # 11. 返回 metrics
    pg_metrics = {
        "actor/pg_clipfrac": pg_clipfrac.detach().item(),
        "actor/ppo_kl": ppo_kl.detach().item(),
        "actor/pg_clipfrac_lower": pg_clipfrac_lower.detach().item(),
    }

    return pg_loss, pg_metrics
```

---

**具体计算示例**:

假设一个 token:
```python
# 输入
old_log_prob = -2.5  # log P_old(token | prefix)
log_prob = -2.0      # log P_new(token | prefix)
advantage = 0.5      # 正优势 (好回答)

# 1. 计算 ratio
ratio = exp(log_prob - old_log_prob)
      = exp(-2.0 - (-2.5))
      = exp(0.5)
      = 1.65
# 含义: 新策略生成该 token 的概率是旧策略的 1.65 倍

# 2. 计算两个 surrogate
surrogate1 = ratio * advantage = 1.65 * 0.5 = 0.825
surrogate2 = clip(ratio, 0.8, 1.2) * advantage
           = clip(1.65, 0.8, 1.2) * 0.5
           = 1.2 * 0.5
           = 0.6

# 3. 取 minimum
clipped_surrogate = min(0.825, 0.6) = 0.6

# 4. 加负号得到 loss
pg_loss = -clipped_surrogate = -0.6

# 5. 反向传播后
# 梯度会让 log_prob 增大 (因为 loss 对 log_prob 的梯度是负的)
# → P_new(token | prefix) 增大
# → 该 token 更容易被生成
```

**为什么 PPO Clip 有效?**
- **好回答** (advantage > 0):
  - 如果 ratio > 1.2: 被 clip 到 1.2, 防止更新过激
  - 如果 ratio < 0.8: loss 很大, 鼓励增加概率
  - 稳定地增加生成概率

- **坏回答** (advantage < 0):
  - 如果 ratio > 1.2: loss 很大, 鼓励降低概率
  - 如果 ratio < 0.8: 被 clip, 防止降低过激
  - 稳定地降低生成概率

#### 🔧 修改点

**修改学习率**:
```bash
# run_qwen3-8b.sh:16
actor_rollout_ref.actor.optim.lr=5e-7  # 降低学习率 (更稳定)
```

**修改 mini_batch_size**:
```bash
# run_qwen3-8b.sh:18
actor_rollout_ref.actor.ppo_mini_batch_size=512  # 增大 (更稳定但慢)
```

**修改 clip_ratio**:
```bash
# 在配置中添加
actor_rollout_ref.actor.clip_ratio=0.1  # 更保守的更新
```

**禁用 KL loss**:
```bash
# run_qwen3-8b.sh:20
actor_rollout_ref.actor.use_kl_loss=False
```

---

### Step 8: 验证测试

#### 📁 文件: `verl/trainer/ppo/ray_trainer.py:531-630`

#### 🎯 作用
每 5 步在验证集上测试模型性能。

#### 📝 代码详解 (简化版)

```python
def _validate(self):
    """验证函数"""
    sample_inputs = []
    sample_outputs = []
    sample_scores = []

    for test_data in self.val_dataloader:
        test_batch = DataProto.from_single_dict(test_data)

        # 1. 重复测试 batch (生成多个回答)
        test_batch = test_batch.repeat(
            repeat_times=self.config.actor_rollout_ref.rollout.val_kwargs.n,  # 默认 5
            interleave=True
        )

        # 2. 生成回答
        test_gen_batch = self._get_gen_batch(test_batch)
        test_gen_batch.meta_info = {
            "do_sample": self.config.actor_rollout_ref.rollout.val_kwargs.do_sample,
            "validate": True,
        }

        test_output = self.actor_rollout_wg.generate_sequences(test_gen_batch)

        # 3. 计算奖励
        test_batch = test_batch.union(test_output)
        result = self.val_reward_fn(test_batch, return_dict=True)
        scores = result["reward_tensor"].sum(-1).cpu().tolist()

        sample_scores.extend(scores)

    # 4. 计算 metrics
    val_metrics = process_validation_metrics(...)

    return val_metrics
```

**验证指标** (对于 GSM8K):
```python
val_metrics = {
    "val/score": 0.75,        # 平均分数
    "val/accuracy": 0.80,     # 准确率 (至少 1 个正确)
    "val/pass@5": 0.82,       # Pass@5 (5 个中至少 1 个正确)
}
```

#### 🔧 修改点
- **修改验证频率**: `run_qwen3-8b.sh:42` → `trainer.test_freq=10` (改为每 10 步)
- **修改生成数量**: 在配置中添加 `actor_rollout_ref.rollout.val_kwargs.n=10`

---

### Step 9: 保存检查点

#### 📁 文件: `verl/trainer/ppo/ray_trainer.py:1280-1286`

#### 📝 代码

```python
if self.config.trainer.save_freq > 0 and (
    is_last_step or
    self.global_steps % self.config.trainer.save_freq == 0 or  # 每 20 步
    esi_close_to_expiration
):
    with marked_timer("save_checkpoint", timing_raw, color="green"):
        self._save_checkpoint()
```

**保存内容**:
- Actor 模型参数
- Optimizer 状态
- 训练步数
- Dataloader 状态

#### 🔧 修改点
- **修改保存频率**: `run_qwen3-8b.sh:41` → `trainer.save_freq=10` (改为每 10 步)

---

## 4. 核心算法详解

### 4.1 GRPO vs PPO

| 对比项 | PPO | GRPO |
|--------|-----|------|
| **Critic** | 需要 Value 网络 | 不需要 |
| **优势估计** | GAE (时序差分) | 组内相对分数 |
| **每个 prompt 生成数** | 1 | G (例如 5) |
| **适用场景** | 通用 RL | 有明确正误的任务 |
| **优点** | 适用性广 | 简单、稳定 |
| **缺点** | 需要额外网络 | 只适合特定任务 |

### 4.2 GRPO 数学原理

**定义**:
- 给定 prompt $p$, 生成 $G$ 个回答 $\{r_1, r_2, ..., r_G\}$
- 每个回答的分数: $s_i = R(p, r_i)$

**优势计算**:
$$
A_i = \frac{s_i - \bar{s}}{\sigma_s}
$$

其中:
- $\bar{s} = \frac{1}{G} \sum_{j=1}^{G} s_j$ (组内均值)
- $\sigma_s = \sqrt{\frac{1}{G} \sum_{j=1}^{G} (s_j - \bar{s})^2}$ (组内标准差)

**性质**:
- $\sum_{i=1}^{G} A_i = 0$ (零和)
- $A_i > 0 \Leftrightarrow s_i > \bar{s}$ (高于平均)
- $A_i < 0 \Leftrightarrow s_i < \bar{s}$ (低于平均)

### 4.3 PPO Clip 数学原理

**目标函数**:
$$
L^{CLIP}(\theta) = -\mathbb{E}_t \left[ \min(r_t(\theta) \hat{A}_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) \hat{A}_t) \right]
$$

其中:
- $r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)}$ (概率比)
- $\hat{A}_t$ (优势估计)
- $\epsilon = 0.2$ (clip 参数)

**效果**:
- 当 $\hat{A}_t > 0$ (好动作):
  - 如果 $r_t > 1+\epsilon$: 被 clip, 防止过度增加概率
  - 否则: 正常增加概率
- 当 $\hat{A}_t < 0$ (坏动作):
  - 如果 $r_t < 1-\epsilon$: 被 clip, 防止过度降低概率
  - 否则: 正常降低概率

---

## 5. 修改代码指南

### 5.1 修改生成数量 (G 参数)

**位置**: `run_qwen3-8b.sh:31`

```bash
# 当前: 生成 5 个
actor_rollout_ref.rollout.n=5

# 修改: 生成 10 个
actor_rollout_ref.rollout.n=10
```

**影响**:
- 计算量: 线性增加 (10 vs 5 = 2倍)
- 稳定性: 增加 (更多样本 → 更准确的 mean/std)
- 内存: 线性增加

---

### 5.2 修改评分函数

**Step 1**: 创建自定义评分函数

文件: `verl/utils/reward_score/my_math.py`
```python
def compute_score(solution_str, ground_truth, **kwargs):
    """
    自定义数学题评分

    Args:
        solution_str: 模型生成的回答
        ground_truth: 正确答案

    Returns:
        float: 分数 (0.0 到 1.0)
    """
    # 1. 提取答案 (你的逻辑)
    answer = extract_answer(solution_str)

    # 2. 判断正确性
    if answer == ground_truth:
        return 1.0
    elif is_close(answer, ground_truth, tolerance=0.01):
        return 0.5  # 接近正确答案
    else:
        return 0.0

def extract_answer(text):
    """提取答案的逻辑"""
    # 例如: 使用正则表达式
    import re
    matches = re.findall(r"answer is (\d+\.?\d*)", text.lower())
    if matches:
        return matches[-1]
    return None
```

**Step 2**: 注册到 reward manager

文件: `verl/trainer/ppo/reward.py:120-196`

在 `load_reward_manager` 函数中添加:
```python
def get_custom_reward_fn(config):
    """获取自定义奖励函数"""
    reward_fn_key = config.data.get("reward_fn_key", None)

    # 添加你的评分函数
    if reward_fn_key == "my_math":
        from verl.utils.reward_score.my_math import compute_score
        return compute_score

    # ... 其他评分函数
    return None
```

**Step 3**: 修改配置

```bash
# run_my_task.sh
data.reward_fn_key=my_math
```

---

### 5.3 修改优势计算 (切换到其他算法)

**禁用标准差归一化** (Dr.GRPO):
```bash
algorithm.norm_adv_by_std_in_grpo=False
```

**切换到 GAE** (需要 Critic):
```bash
algorithm.adv_estimator=gae
algorithm.gamma=0.99
algorithm.lam=0.95
trainer.critic_warmup=100  # Critic 预热步数
```

---

### 5.4 修改学习率和训练参数

```bash
# 学习率
actor_rollout_ref.actor.optim.lr=5e-7  # 降低学习率

# Batch size
data.train_batch_size=512              # 减小 batch size
actor_rollout_ref.actor.ppo_mini_batch_size=128
actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=16

# Clip ratio
actor_rollout_ref.actor.clip_ratio=0.1  # 更保守

# KL loss
actor_rollout_ref.actor.use_kl_loss=True
actor_rollout_ref.actor.kl_loss_coef=0.01  # 增大 KL 惩罚

# Epoch
trainer.total_epochs=20  # 增加训练轮数
```

---

### 5.5 修改生成参数

```bash
# 温度 (增加多样性)
actor_rollout_ref.rollout.temperature=0.7

# Top-p
actor_rollout_ref.rollout.top_p=0.9

# Max tokens
data.max_response_length=2048
```

---

### 5.6 修改模型和数据

```bash
# 模型
actor_rollout_ref.model.path=meta-llama/Llama-3.1-8B

# 数据集
data.train_files=/path/to/your/train.parquet
data.val_files=/path/to/your/val.parquet
```

---

## 6. 常见问题

### Q1: 训练不收敛怎么办?

**检查清单**:
1. **奖励函数**: 打印几个样本的 reward, 确保正确
   ```python
   # 在 ray_trainer.py:1128 后添加
   print("Sample rewards:", reward_tensor[:5].sum(-1))
   ```

2. **优势分布**: 检查 advantages 的均值和方差
   ```python
   # 在 ray_trainer.py:1230 后添加
   print("Advantage mean:", batch.batch["advantages"].mean())
   print("Advantage std:", batch.batch["advantages"].std())
   ```

3. **学习率**: 尝试降低学习率
   ```bash
   actor_rollout_ref.actor.optim.lr=1e-7
   ```

4. **G 太小**: 增加生成数量
   ```bash
   actor_rollout_ref.rollout.n=8
   ```

5. **KL 惩罚**: 增加 KL loss 防止模型退化
   ```bash
   actor_rollout_ref.actor.kl_loss_coef=0.01
   ```

---

### Q2: 显存不足怎么办?

**解决方案**:

1. **减小 batch size**:
   ```bash
   data.train_batch_size=512
   actor_rollout_ref.actor.ppo_mini_batch_size=128
   actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=16
   ```

2. **启用 gradient checkpointing**:
   ```bash
   actor_rollout_ref.model.enable_gradient_checkpointing=True
   ```

3. **启用 parameter offload**:
   ```bash
   actor_rollout_ref.actor.fsdp_config.param_offload=True
   actor_rollout_ref.actor.fsdp_config.optimizer_offload=True
   ```

4. **减少生成数量**:
   ```bash
   actor_rollout_ref.rollout.n=3
   ```

5. **降低 max_tokens**:
   ```bash
   data.max_response_length=512
   ```

---

### Q3: 如何加速训练?

1. **增大 batch size** (如果显存够):
   ```bash
   data.train_batch_size=2048
   ```

2. **减少验证频率**:
   ```bash
   trainer.test_freq=10
   trainer.save_freq=50
   ```

3. **使用更快的生成引擎**:
   ```bash
   actor_rollout_ref.rollout.name=sglang  # 或 vllm
   ```

4. **增大 tensor parallel**:
   ```bash
   actor_rollout_ref.rollout.tensor_model_parallel_size=4
   ```

---

### Q4: 如何调试生成质量?

**启用 rollout 日志**:
```bash
trainer.rollout_data_dir=/path/to/save/rollout_logs
```

生成的日志包含:
- 输入 prompt
- 生成的 response
- 奖励分数
- 优势值

**查看日志**:
```python
import pandas as pd

# 读取日志
df = pd.read_parquet("/path/to/save/rollout_logs/step_100.parquet")

# 查看一个 prompt 的所有回答
prompt_0 = df[df["uid"] == df["uid"].iloc[0]]
print(prompt_0[["response", "reward", "advantage"]])
```

---

### Q5: 如何从检查点恢复训练?

**配置中添加**:
```bash
trainer.load_checkpoint=/path/to/checkpoint
```

或在代码中:
```python
# ray_trainer.py:998
def _load_checkpoint(self):
    if self.config.trainer.get("load_checkpoint", None):
        # 加载检查点逻辑
        ...
```

---

## 总结

### 核心文件清单

| 文件 | 作用 | 修改建议 |
|------|------|----------|
| `run_qwen3-8b.sh` | 启动脚本 | 修改超参数 |
| `main_ppo.py` | 启动入口 | 一般不修改 |
| `ray_trainer.py` | 训练循环 | 添加 debug 日志 |
| `core_algos.py` | GRPO 算法 | 理解原理 |
| `dp_actor.py` | PPO 更新 | 理解原理 |
| `gsm8k.py` | 评分函数 | 替换为自己的 |

### 修改代码的一般流程

1. **明确目标**: 想修改什么? (评分函数? 优势计算? 学习率?)
2. **找到位置**: 根据本文档的"修改点"章节
3. **修改代码**: 最好先在配置文件修改,再改代码
4. **小规模测试**: 用小数据集测试 (data.train_batch_size=64)
5. **全量训练**: 确认无误后全量训练

### 关键数据流

```
Prompt → (×5) → 1280 prompts → vLLM → 1280 responses
  ↓
Reward (GSM8K) → 1280 scores
  ↓
GRPO → 按 uid 分组 → 计算组内 mean/std → advantages
  ↓
PPO → 分 mini-batch → 分 micro-batch → 计算 loss → backward → update
```

---

**祝你使用愉快! 如有问题,请参考本文档相应章节。**

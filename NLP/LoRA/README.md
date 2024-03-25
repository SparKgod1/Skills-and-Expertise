# 基于LoRA的哈工大SFT数据集问答微调
## LoRA（Low-Rank Adaptation of Large Language Models）
微调大规模语言模型到特殊领域和任务是自然语言处理的重要课题之一。但随着模型规模的不断扩大，微调模型的所有参数（所谓full fine-tuning）的可行性变得越来越低。以GPT-3的175B参数为例，每增加一个新领域就需要完整微调一个新模型，代价和成本很高。

***已有方案的问题***

为解决微调大规模语言模型到不同领域和任务的挑战，已有多种方案，比如部分微调、使用adapters和prompting。但这些方法存在如下问题：

- Adapters引入额外的推理延迟 (由于增加了模型层数)
- Prefix-Tuning难于训练，且预留给prompt的序列挤占了下游任务的输入序列空间，影响模型性能

### 1. Adapter引入推理延迟
显然，使用Adapter增加模型层数会增加推理的时长：

![](https://cdn.jsdelivr.net/gh/SparKgod1/img/adapter%E5%A2%9E%E5%8A%A0%E6%8E%A8%E7%90%86%E6%97%B6%E9%95%BF.png)

从上图可以看出，对于线上batch size为1，输入比较短的情况，推理延迟的变化比例会更明显。

简单来说，adapter就是固定原有的参数，并添加一些额外参数用于微调。Adapter会在原始的transformer block中添加2个adapter，一个在多头注意力后面，另一个这是FFN后面。显然，adapter会在模型中添加额外的层，这些层会导致大模型在推理时需要更多的GPU通信，而且也会约束模型并行。这些问题都将导致模型推理变慢。
### 2. 很难直接优化Prompt

![](https://cdn.jsdelivr.net/gh/SparKgod1/img/%E5%BE%88%E9%9A%BE%E7%9B%B4%E6%8E%A5%E4%BC%98%E5%8C%96Prompt.png)

prefix-tuning方法是受语言模型in-context learning能力的启发，只要有合适的上下文则语言模型可以很好的解决自然语言任务。但是，针对特定的任务找到离散token的前缀需要花费很长时间，prefix-tuning提出使用连续的virtual token embedding来替换离散token。

具体来说，对于transformer中的每一层，都在句子表征前面插入可训练的virtual token embedding。对于自回归模型（GPT系列），在句子前添加连续前缀，即

虽然，prefix-tuning并没有添加太多的额外参数。但是，prefix-tuning难以优化，且会减少下游任务的序列长度，一定程度上会影响模型性能。 
### 一、 LoRA的原理简介
NLP领域的一个重要课题是，一般领域数据的通用大模型对特定任务或领域的适应。当预训练大模型很大时，重新训练所有模型参数的微调变得不可太行，例如GPT3的175B。提出的lora采用低秩分解矩阵，冻结了预训练模型的权重，并将低秩分解矩阵注入到transformer的每一层，减少了训练参数量。

![](https://cdn.jsdelivr.net/gh/SparKgod1/img/LoRA%E5%8E%9F%E7%90%86.png)

如上图所示们对于某个线性层而言，左边是模型原有的参数，在训练过程中是冻结不变的，右边是lora方法增加的低秩分解矩阵。

在原始PLM旁边增加一个旁路，做一个降维再升维的操作，来模拟所谓的intrinsic rank。训练的时候固定PLM的参数，只训练降维矩阵A与升维矩阵B。而模型的输入输出维度不变，输出时将BA与PLM的参数叠加。用随机高斯分布初始化A，用0矩阵初始化B，保证训练的开始此旁路矩阵依然是0矩阵。

训练过程中，优化器只优化右边这一部分的参数，两边的矩阵会共用一个模型的输入，分别进行计算，最后将两边的计算结果相加作为模块的输出。不同于之前的参数高效微调的adapter，

- adapter是在模块的后面接上一个MLP，对模块的计算结果进行一个后处理
- lora是和模块的计算并行的去做一个MLP，和原来的模块共用一个输入

![](https://cdn.jsdelivr.net/gh/SparKgod1/img/LORA__.png)

具体来看，假设预训练的矩阵为W0，它的更新可表示为：

![](https://cdn.jsdelivr.net/gh/SparKgod1/img/LORA_M.png)

其中秩![](https://cdn.jsdelivr.net/gh/SparKgod1/img/LORA_r.png)
这种思想有点类似于残差连接，同时使用这个旁路的更新来模拟full finetuning的过程。并且，full finetuning可以被看做是LoRA的特例（当r等于k时）。 

根据之前的一些工作，发现大模型其实是过参数化的， 有更小的一个内在维度，于是文章做了一个假设，大模型在任务适配（instruction-tune）过程中，参数的改变量是低秩的，

在训练过程中，lora单独去学习这个改变量，而不是去学习模型的参数，通过把最终训练得到的参数分解为原参数W0和该变量deltaW进行相加，论文假设deltaW是低秩的，把deltaW进一步拆分为低秩矩阵A和低秩矩阵B。

在推理的过程中，由于模型参数已经固定不再变动，这时候把模型的改变量直接放到模型里，这样在推理的计算过程中，就避免了一次额外的矩阵乘法开销。推理是改变量是直接加到原路径中的。在切换不同推理任务时，只需要从模型参数里减去当前任务的该变量，再换上新任务的改变量即可。这里隐含了一个推理：大模型中已经内含了一些小模型的参数空间，特定的垂直领域instruction-tune本质上就等价于”切割“出这些小的子空间。

![](https://cdn.jsdelivr.net/gh/SparKgod1/img/LORA_posi.png)
![](https://cdn.jsdelivr.net/gh/SparKgod1/img/LORA_.png)
![](https://cdn.jsdelivr.net/gh/SparKgod1/img/lllora.png)

理论上lora可以支持任何线性层，包括transformer中的4个attention矩阵和2个feed forward中的矩阵，论文旨在attention上做了实验，它限制总参数量不变的情况下观察是在attention其中一个矩阵上，放一个更高秩的lora，还是在多个attention的矩阵上，分别放置低秩一点的lora效果好？

结论是把秩分散到多个矩阵上，效果会优于集中在单个上的效果。在一般任务上很小的秩就可以和很大秩具备类似的效果，这也证明了作者一开始做出的改变量低秩的假设。

![](https://cdn.jsdelivr.net/gh/SparKgod1/img/LLORA.png)

初始化一开始右边为0，也就意味着模型优化的初始点就和原本的大模型能够保持一致，这一点和controlnet中的zero convolution是一致的。

![](https://cdn.jsdelivr.net/gh/SparKgod1/img/20240324153026.png)
![](https://cdn.jsdelivr.net/gh/SparKgod1/img/20240324153101.png)
![](https://cdn.jsdelivr.net/gh/SparKgod1/img/20240324153123.png)

总之，基于大模型的内在低秩特性，增加旁路矩阵来模拟全模型参数微调，LoRA通过简单有效的方案来达成轻量微调的目的。 

引申一下，GPT的本质是对训练数据的有效压缩，从而发现数据内部的逻辑与联系，LoRA的思想与之有相通之处，原模型虽大，但起核心作用的参数是低秩的，通过增加旁路，达到事半功倍的效果。
### 二、 LoRA原理的数学化表达
1. 术语与约定
由于LoRA原理的介绍，会使用Transformer架构。因此，这里先给出一些术语约定。

一个Transformer层的输入和输出维度尺寸为d_model
2. 问题表述
LoRA虽然与训练目标无关，这里还是以语言建模为例。假设给定一个预训练的自回归语言模型 P

![](https://cdn.jsdelivr.net/gh/SparKgod1/img/20240324153319.png)

在完整微调（fine-tune）的过程中，模型使用预训练好的权重

![](https://cdn.jsdelivr.net/gh/SparKgod1/img/20240324153136.png)

完整微调的主要缺点：对于每个下游任务，都需要学习不同的参数更新

LoRA为了更加的参数高效，使用相对非常小的参数

![](https://cdn.jsdelivr.net/gh/SparKgod1/img/20240324153144.png)

LoRA将会使用低秩表示来编码 

参考链接：
```
https://blog.csdn.net/u012193416/article/details/129427242
https://finisky.github.io/lora/ 
https://zhuanlan.zhihu.com/p/618073170
```
### 三、 基于LoRA的哈工大SFT数据集问答微调

#### 数据集示例
共17925行数据，每行数据由系统提示（instruction），用户输入（input），模型输出（output）三个属性组成
```
DatasetDict({
    train: Dataset({
        features: ['output', 'input', 'instruction'],
        num_rows: 17925
    })
})
```
```
[
    {
        "instruction": "请回答下面的问题",
        "input": "哈尔滨工业大学的校训是什么？",
        "output": "规格严格，功夫到家"
    },
    {
        "instruction": "",
        "input": "哈尔滨工业大学是何时建立的？",
        "output": "哈尔滨工业大学始建于1920年"
    },
    {
        "instruction": "",
        "input": "哈尔滨工业大学最早的校名是什么？",
        "output": "最早的校名为哈尔滨中俄工业学校"
    },
    {
        "instruction": "",
        "input": "哈工大首任校长是谁？",
        "output": "宋小濂校长，1920年上任，1924年退任"
    },
    {
        "instruction": "",
        "input": "哈工大现任校长是谁？",
        "output": "韩杰才校长，自2021年7月15日上任至今"
    },
    {
        "instruction": "",
        "input": "哈工大首任党委书记是谁？",
        "output": "陈康白，于1951年上任，1953年10月退任，曾任东北人民政府文化部副部长，后任中国科学院秘书长"
    },
```
将数据集整理成以下格式
```
<|beginofutterance|>系统\n{context}\n<|endofutterance|>\n<|beginofutterance|>用户\n{question}\n<|endofutterance|>\n<|beginofutterance|>智能助手\n{answer}<|endofutterance|>\n
```
#### LoRA参数
```
LoraConfig(
    r=8,  # 查询与键之间的注意力分布的头数或数量
    lora_alpha=8,  # 控制查询和键之间注意力分布的相对重要性
    target_modules=["query_key_value"],  # 使用的目标模块，这里是查询键值对
    lora_dropout=0.05,  # Dropout率，用于减少过拟合
    bias="none",  # 是否使用偏置，这里设置为“none”表示不使用
    task_type="CAUSAL_LM"  # 模型的任务类型，这里是因果语言建模任务
)
```
比对使用LoRA前后可训练参数量的变化
```
trainable_params = 0
all_param = 0

for _, param in peft_model.named_parameters():
    all_param += param.numel()
    if param.requires_grad:
        trainable_params += param.numel()

print(f"trainable params: {trainable_params}")
print(f"all params: {all_param}")
print(f"trainable: {100 * trainable_params / all_param:.2f}%")
```
```
输出结果：
trainable params: 1179648
all params: 1066493952
trainable: 0.11%
```
利用LoRA，将大模型的训练参数量降到了原本的**0.11%**，极大的降低了模型训练对显存的要求

#### 训练参数
```
transformers.Trainer(
    model=peft_model,
    train_dataset=mapped_qa_dataset["train"],
    args=transformers.TrainingArguments(
        per_device_train_batch_size=1,  # 每个设备的训练批量大小
        gradient_accumulation_steps=1,  # 梯度累积步数
        warmup_steps=100,  # 学习率预热步数
        max_steps=1000,  # 最大训练步数
        # num_train_epochs=10,  # 训练轮数（如果使用 epochs 替代 max_steps，则取消注释此行）
        learning_rate=1e-4,  # 学习率
        fp16=True,  # 是否使用混合精度训练
        logging_steps=100,  # 日志记录步数
        output_dir='/content/drive/MyDrive/outputs_1',  # 输出目录
        per_device_train_batch_size=16  # 每个设备的默认训练批量大小
    ),
    data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False)
)
```
由于我们使用的是Colab提供的40GB显存的T100显卡进行训练，当训练max_step大于1000时会爆显存，因此只对模型训练1000个step。

之后将微调的参数与原模型合并成一个微调的新模型
```
model = AutoModelForCausalLM.from_pretrained(
    "bigscience/bloom-1b1",
    device_map="auto",
    torch_dtype=torch.float16,
)

model = PeftModel.from_pretrained(
    model,
    "/content/drive/MyDrive/BLOOM1111111111",
    torch_dtype=torch.float16,
)
model = model.merge_and_unload()
model.save_pretrained('/content/drive/MyDrive/_1')
```
微调模型+tokenizer链接：
https://drive.google.com/drive/folders/1-00jZNGVXnjlwiS5lLYAST_SwiBZHcRZ?usp=drive_link
#### 加载训练好的模型
```
model_finetuned = AutoModelForCausalLM.from_pretrained(
    ".../your_path/model_finetuened",
    device_map="auto",
    torch_dtype=torch.float16,
)

model_finetuned = model_finetuned.to("cuda")
tokenizer = AutoTokenizer.from_pretrained(".../your_path/tokenizer")
```
#### 结果分析
原始预训练模型回答:
``` 
系统
请回答下面的问题

用户
哈工大有多少杰出人才

智能助手
问一下
问一下
问一下
问一下
问一下
问一下
问一下
问一下
问一下
问一下
问一下
问一下
问一下
问一下
问一下
问一下
问一下
问一下
问一下
问一下
问一下
问一下
问一下
问一下
```
可以看出，模型由于缺少对该输入格式的训练，导致模型在回答中文问题时出现循环输出的情况。

微调模型回答: 
```
系统
请回答下面的问题

用户
哈工大有多少杰出人才

智能助手
哈工大是我国著名的高等院校，拥有众多的杰出人才。哈工大在人才培养方面具有独特的优势，在人才培养方面具有丰富的经验。哈工大在人才培养方面具有独特的优势，在人才培养方面具有丰富的经验。
```
经LoRA微调后，模型已经适应了该输出格式，并且能回答SFT数据集中的内容。

#### 不足之处
由于训练次数较少，模型对该格式尚未充分适应，导致模型无法在句子末尾正确输出标识符```<|endofutterance|>```，因此无法用以标识符截断输出的句子，只能以输出最大长度截断。

# DeepSeek 模型序列并行 SFT 实现

本仓库基于 [360-LLaMA-Factory](https://github.com/Qihoo360/360-LLaMA-Factory) 和 [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) 项目，实现了对 DeepSeek 模型的序列并行（Sequence Parallel）监督微调（SFT, Supervised Fine-Tuning）。



## 主要功能
- **序列并行 SFT**：利用序列并行技术对 DeepSeek 模型进行监督微调，以提高模型在处理长序列时的性能和效率。
- **基于现有框架**：借助 [360-LLaMA-Factory](https://github.com/Qihoo360/360-LLaMA-Factory) 和 [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) 提供的工具和方法，快速搭建和实现微调流程。

## 安装依赖
在开始使用本项目之前，你可以参考[360-LLaMA-Factory](https://github.com/Qihoo360/360-LLaMA-Factory) 和 [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory)，安装必要的依赖项。

## 使用方法
### 数据准备
首先，准备好用于微调的数据集。数据集应按照特定的格式进行组织，确保模型能够正确读取和处理。

### 配置文件
根据你的需求，修改examples/train_lora/deepseek_lora_sft_ds3.yaml配置文件中的参数，如模型路径、数据集路径、微调超参数等。

### 运行微调
运行以下命令开始对 DeepSeek 模型进行序列并行 SFT：
```bash
llamafactory-cli train examples/train_lora/deepseek_lora_sft_ds3.yaml
```

## 贡献
如果你发现了任何问题或有改进的建议，欢迎提交 Issue 或 Pull Request。我们非常欢迎社区的贡献，共同推动项目的发展。

## 许可证
本项目采用[360-LLaMA-Factory](https://github.com/Qihoo360/360-LLaMA-Factory) 和 [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory)许可证。

## 致谢
特别感谢 [360-LLaMA-Factory](https://github.com/Qihoo360/360-LLaMA-Factory) 和 [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) 项目的开发者，他们的工作为我们的实现提供了重要的基础和参考。
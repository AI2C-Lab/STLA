# STLA: Spatiotemporal Lookahead Alignment for Post-Training Quantization

This repository contains the official PyTorch implementation for the ICML 2026 paper *"STLA: Spatiotemporal Lookahead Alignment for Post-Training Quantization"*.
![overview](overview.png)
<!-- abstract ... -->

## Get Started

1. **Clone this repository**:

   ```bash
   git clone https://anonymous.4open.science/r/STLA
   ```

   Then navigate to the desired directory:

   ```bash
   cd STLA
   ```

2. **Install**:

   1.Ensure you have PyTorch installed. You can install PyTorch 1.10.0 with the following command:

   ```bash
   pip install torch==1.10.0 torchvision --index-url https://download.pytorch.org/whl/cu113
   ```

   2.Pretrained models for STLA can be obtained using the transformers library (version 4.43.2). Ensure transformers is installed:

   ```bash
   pip install transformers==4.43.2
   ```

   For dataset loading and preprocessing, this project uses the datasets library, tested on version 2.20.0. Install it with:

   ```bash
   pip install datasets==2.20.0
   ```

2. **Options**:

    - `groupsize`: Group size for quantization.`-1`: Per-channel quantization. `>0`: Specifies the group size for finer-grained quantization.
    - `blocksize`: Block Size. `-1`: Invalid lazy latch-updates. `>0`:The block size used for lazy latch-updates during the GPTQ/GPTAQ process.
    - `clustersize`: Cluster Size. Number of columns processed per cluster. `-1`: Disables joint optimization. `>0`: The number of columns where weight interactions are explicitly modeled.
    - `loss_option`: Loss Scope.`'global'`: Uses global loss for hyperparameter search and Adaround.`'local'`: Uses local loss.
    - `order_option`: Hessian-based Re-ordering.`'spin'`: Applies spincluster re-ordering.`'act'`: Re-orders based on activation magnitude.<br>• `'none'`: Disables re-ordering.
    - `comp_method`:Compensation Method.`'GPTQ'`: Uses standard GPTQ error compensation.`'GPTAQ'`: Uses standard GPTAQ error compensation method. 
    - `learn_rounding`: Adaround. Learns the optimal rounding policy via gradient descent instead of using nearest-neighbor rounding. 

    For more details on other arguments, please refer to [utils.py](utils.py).

Alternatively, you can directly download checkpoints provided by [AdaLog](https://github.com/GoatWu/AdaLog). For example:

```bash
wget https://github.com/GoatWu/AdaLog/releases/download/v1.0/deit_tiny_patch16_224.bin
mkdir -p ./checkpoint/vit_raw/
mv deit_tiny_patch16_224.bin ./checkpoint/vit_raw/
```

## Usage

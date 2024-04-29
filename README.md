# llmsys_s24_project

Flash Attention Variants in Minitorch

Authors: Pratheek D Souza Rebello, Richa Gadgil

![Final Speedup](https://github.com/pdrebello/flash-attention-minitorch/assets/46563240/fe0f2a1e-b16b-4f5e-9560-7976b00ce300)

Analysis runtime of vanilla attention shows that instead of matmul, it is pointwise and linear operations like masking, dropout and softmax which are bottlenecks due to memory
![Breakup](https://github.com/pdrebello/flash-attention-minitorch/assets/46563240/beb1a167-b548-45a4-bf9d-d111088b3445)

Ablation studies of trends with varying batch size, heads and feature dimension (d). Batch size and head are treated equivalently by flash attention kernels by parallelizing across both dimensions in independent blocks
![Ablations](https://github.com/pdrebello/flash-attention-minitorch/assets/46563240/1ce45e62-93bb-4092-9281-1df347fb10da)

Performance of Flash Attention integrated with a full decoder-only transformer network and trained on a machine translation task
![LLM Speedup](https://github.com/pdrebello/flash-attention-minitorch/assets/46563240/cde699b5-efa7-4afd-957d-4d3c601bc941)


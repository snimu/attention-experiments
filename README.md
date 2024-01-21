# attention-experiments

Just me playing around with Attention mechanisms for a bit of learning and fun.

I'll try to compare vanilla softmax Attention to Hydra Attention and two simplified variants of Hydra Attention.

The LLM code is based mostly on [Fern](https://github.com/tysam-code/)'s 
[hlb-gpt](https://github.com/tysam-code/hlb-gpt):

```
cff-version: 1.2.0
message: "If you need to cite this codebase for any reason, please do so as below."
authors:
- family-names: "Balsam"
  given-names: "Tysam&"
title: "hlb-gpt"
version: 0.0.0
date-released: 2023-03-05
url: "https://github.com/tysam-code/hlb-gpt"
```

The diffusion code is based heavily on [The Annotated Diffusion Model](https://huggingface.co/blog/annotated-diffusion)
by [Niels Rogge](https://huggingface.co/nielsr) and [Kashif Rasul](https://huggingface.co/kashif).

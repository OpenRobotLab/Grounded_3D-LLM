# Grounded 3D-LLM with Referent Tokens

This repository will release the official implementation of "Grounded 3D-LLM with Referent Token".

> [[`Paper`]](https://arxiv.org/pdf/2405.10370) [[`Arxiv`]](https://arxiv.org/abs/2405.10370) [[`Website`]](https://groundedscenellm.github.io/grounded_3d-llm.github.io/) [[`Data`]](https://mycuhk-my.sharepoint.com/:f:/g/personal/1155113995_link_cuhk_edu_hk/EpGS4c90LVVMvzio0UXgHfoB1u78-WpYaZfTuJj8qCbC4g?e=B2sufx)

## Abstract

Prior studies on 3D scene comprehension have primarily developed specialized models for specific tasks or required task-specific fine-tuning. **In this study, we propose *Grounded 3D-LLM*, which explores the potential of 3D large multi-modal models (LMMs) to consolidate various 3D visual tasks within a unified generative framework.** The model utilizes a series of ``referent tokens'' to reference 3D scenes, enabling the handling of sequences that interleave 3D and textual data arbitrarily. 3D vision tasks are naturally transformed into language formats through task-specific prompts. **To effectively associate the scene with text, we curate the grounded language datasets either from human-annotated sources or by bootstrapping existing object labels at the phrase level. We then employ Contrastive Language-Scene Pre-training (CLASP) to bridge the divide between 3D vision and language models, **thus facilitating the use of referent tokens in subsequent language modeling tasks. Our comprehensive evaluation covers open-ended tasks like 3D visual question answering and dense captioning, as well as close-ended tasks such as object detection and language grounding. We also explore the potential to generate interleaved scene *referents* and textual language outputs for embodied dialogue and planning. Experiments across multiple 3D benchmarks reveal the superior performance and the broad applicability of *Grounded 3D-LLM*. Code and datasets will be made publicly accessible.

![image-20240515195822834](./README.assets/image-20240515195822834.png)

## ToDo

- [x] Release Grouded Scene Caption data (ScanNet).
- [ ] Release data visualizer.
- [ ] Release data generation code.
- [ ] Release code.

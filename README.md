# Improved DreamBooth: Enhanced Fine-Tuning through SNR Adjustment and Optimization of Noise Scheduler

> **Paper:** "Improved DreamBooth: SNR 조정 및 노이즈 스케줄러 최적화를 통한 개선된 파인튜닝"  
> **Published in:** Journal of Digital Art Engineering & Multimedia, Vol.10, No.4, December 2023, pp.567-576  
> **DOI:** [10.29056/jdaem.2023.12.12](http://dx.doi.org/10.29056/jdaem.2023.12.12)

---

## Overview

This repository contains the implementation of **Improved DreamBooth**, a fine-tuning approach for personalizing Stable Diffusion using:

1. **Zero Terminal SNR Algorithm** — fixes the noise schedule so that SNR reaches 0 at the final timestep T
2. **Sigmoid Beta Schedule** — replaces the default Linear Schedule for smoother noise transitions

When combined, these two modifications significantly improve facial synthesis quality compared to vanilla DreamBooth fine-tuning on Stable Diffusion.

| Method | Result |
|---|---|
| DreamBooth (Linear Schedule) | Blurry, dark-toned outputs |
| Algorithm 1 + Linear Schedule | Marginal improvement |
| Sigmoid Schedule only | More color variety, but low fidelity |
| **Algorithm 1 + Sigmoid Schedule** | ✅ Best — clear, realistic faces |

---

## Background

DreamBooth proposes a fine-tuning method to personalize text-to-image diffusion models using only 3–5 reference images. However, when applied to **Stable Diffusion** (which omits the Super-Resolution fine-tuning stage used in the original Imagen-based DreamBooth), facial synthesis quality degrades noticeably — producing blurry or overly dark images.

This work addresses that gap by introducing two modifications to the **Forward Process** of the Noise Scheduler.

---

## Method

### 1. Zero Terminal SNR (Algorithm 1)

Standard diffusion noise schedules have a flaw: the SNR does not reach 0 at the final timestep T, causing a mismatch between training and inference. This algorithm enforces SNR = 0 at timestep T by rescaling `alphas_bar_sqrt`.

```python
def enforce_zero_terminal_snr(self, betas):
    alphas = 1.0 - betas
    alphas_bar = tf.math.cumprod(alphas)
    alphas_bar_sqrt = tf.sqrt(alphas_bar)

    alphas_bar_sqrt_0 = alphas_bar_sqrt[0]
    alphas_bar_sqrt_T = alphas_bar_sqrt[-1]

    # Shift so last timestep is zero
    alphas_bar_sqrt -= alphas_bar_sqrt_T
    # Scale so first timestep is back to old value
    alphas_bar_sqrt *= alphas_bar_sqrt_0 / (alphas_bar_sqrt_0 - alphas_bar_sqrt_T)

    alphas_bar = alphas_bar_sqrt ** 2
    alphas = alphas_bar[1:] / alphas_bar[:-1]
    alphas = tf.concat([alphas_bar[0:1], alphas], axis=0)
    betas = 1 - alphas
    return betas
```

### 2. Sigmoid Beta Schedule

Replaces the default **Linear Schedule** (β linearly increasing from 0.0001 to 0.02) with a **Sigmoid Schedule**, which transitions more smoothly.

```python
def sigmoid_schedule(self, t, start=-3, end=3, tau=0.7, clip_min=1e-9):
    v_start = tf.sigmoid(start / tau)
    v_end = tf.sigmoid(end / tau)
    output = tf.sigmoid((t * (end - start) + start) / tau)
    output = (v_end - output) / (v_end - v_start)
    return tf.clip_by_value(output, clip_min, 1.0)
```

To use the Sigmoid Schedule, initialize the `NoiseScheduler` with `beta_schedule="sigmoid"`:

```python
scheduler = NoiseScheduler(
    train_timesteps=1000,
    beta_schedule="sigmoid",   # Key change
)
```

The Zero Terminal SNR is automatically applied inside the sigmoid branch.

---

## Usage

### Prepare Instance Images

Gather **3–5 images** of the subject you want to personalize. For best results with human faces:

- Use a **meaningless unique identifier** (e.g., `sks`) to avoid conflicts with pre-trained semantics
- Prepare **200 class images** for Prior Preservation Loss using a prompt like:
  ```
  A photo of person without mustache, handsome, ultra realistic, 4k, 8k
  ```

### Fine-Tuning

```bash
python train_dreambooth.py \
  --instance_images_dir ./instance_images \
  --class_images_dir ./class_images \
  --instance_prompt "A photo of sks person" \
  --class_prompt "A photo of person" \
  --beta_schedule sigmoid \
  --use_zero_terminal_snr \
  --learning_rate 5e-6 \
  --max_train_steps 1000
```

### Inference

```bash
python inference.py \
  --prompt "A photo of sks person without mustache, handsome, ultra realistic, 4k, 8k" \
  --model_path ./output/personalized_model
```

---

## Experimental Setup

| Component | Specification |
|---|---|
| CPU | Intel Core i7-12700KF |
| GPU | NVIDIA GeForce RTX 3090 Ti 24GB |
| RAM | Samsung DDR4 32GB |
| OS | Ubuntu Linux 20.04.6 LTS |
| CUDA | 11.8.0 |
| cuDNN | 8.6.0 |
| Python | 3.10.12 |
| Deep Learning Library | TensorFlow 2.12.0 |

**Hyperparameters:**
- Final timestep T: `1000`
- Learning Rate: `5e-6`
- Class images: `200`
- Unique identifier: `sks`

---

## Key Findings

- Fine-tuning **only the U-Net** of Stable Diffusion (without the Text Encoder or other components) is sufficient for effective personalization
- **Algorithm 1 alone** (with Linear Schedule) yields minimal improvement
- **Sigmoid Schedule alone** introduces color diversity but reduces fidelity
- **Algorithm 1 + Sigmoid Schedule** together produce the best results — resolving the blurry output and dark-tone-only problem observed in baseline DreamBooth
- Using less subjective words in text prompts preserves subject-specific features more faithfully

---


## References

- Ruiz et al., "DreamBooth: Fine Tuning Text-to-Image Diffusion Models for Subject-Driven Generation", CVPR 2023
- Rombach et al., "High-Resolution Image Synthesis with Latent Diffusion Models", CVPR 2022
- Lin et al., "Common Diffusion Noise Schedules and Sample Steps are Flawed"
- Ho et al., "Denoising Diffusion Probabilistic Models", NeurIPS 2020
- Nichol & Dhariwal, "Improved Denoising Diffusion Probabilistic Models", ICML 2021

---

*This research was supported by the Korea Creative Content Agency (KOCCA) under the Ministry of Culture, Sports and Tourism (Project No. R2022020014).*

# Domain Adaptation in High-Resolution Satellite Images using Deep Learning

**Self-supervised, domain-agnostic image-to-image translation for building segmentation across satellite datasets.**

[![Status](https://img.shields.io/badge/status-work--in--progress-yellow)]()
[![Approach](https://img.shields.io/badge/approach-SS(DA)²%20%2F%20CUT-blue)]()
[![Backbone](https://img.shields.io/badge/segmentation%20backbone-ResNet50-informational)]()
[![License](https://img.shields.io/badge/license-BSD-lightgrey)]()

## Overview

Deep convolutional networks trained for semantic segmentation on Very-High-Resolution (VHR)
satellite imagery tend to degrade sharply when applied to imagery from a different
geographic region or acquisition setting — a classic **domain shift** problem. This is made
worse by the scarcity of densely annotated satellite datasets for every region/sensor
combination, and it particularly hurts detection of small, densely packed objects such as
buildings and vehicles.

This project explores a **self-supervised, domain-agnostic domain adaptation (SS(DA)²)**
approach: a GAN with a contrastive-learning component that performs image-to-image
translation between satellite image patches from two different domains, without requiring
explicit domain labels. The translated (domain-adapted) imagery is intended to be used to
improve a downstream semantic segmentation model (building / not-building) so that it
generalizes better across datasets.

The implementation is built on top of the **Contrastive Unpaired Translation (CUT / FastCUT)**
codebase by Park et al., adapted and re-trained here for satellite imagery domain adaptation
between the WHU Building Dataset and the INRIA Aerial Image Labeling Dataset.

## Problem & Approach

- **Problem**: A DCNN segmentation model trained on one satellite dataset (source domain)
  loses accuracy when evaluated on another dataset (target domain) because of differences in
  sensor, resolution, geography, and acquisition conditions.
- **Approach**: Adversarial learning + contrastive learning (PatchNCE) to translate images
  from one domain into the visual style of another while preserving structural content
  (buildings, roads, layout), rather than requiring paired/labeled cross-domain data.
- **Architecture**:
  - Encoder–decoder generator (CUT's ResNet-based encoder with an instance-normalization /
    ReLU stack) with a **UNet-style decoder** for image-to-image translation, as described in
    Zhang, Shi & Zhu's SS(DA)² formulation.
  - **PatchNCE contrastive loss** to enforce content correspondence between source patches and
    their translated counterparts, avoiding the need for paired image data.
  - Downstream **semantic segmentation head on a ResNet50 backbone** for the
    building / not-building task.

## Datasets

| Dataset | Coverage | Resolution | Notes |
|---|---|---|---|
| [INRIA Aerial Image Labeling Dataset](https://project.inria.fr/aerialimagelabeling/) | 810 km², urban areas from dense cities to small towns | 0.3 m | Binary labels: building / not-building |
| [WHU Building Dataset](http://gpcv.whu.edu.cn/data/building_dataset.html) | 220,000+ labeled buildings, 450 km² (Christchurch, NZ) + a global satellite subset | 0.075 m (aerial) | Includes a lower-resolution (2.7 m) satellite subset covering East Asia |

Dataset download/setup follows the standard CUT convention of `trainA` / `trainB` (and
`testA` / `testB`) subdirectories per domain — see `docs/datasets.md`.

## Status / Results

**This is a proposal-and-progress-stage research project, not a finished benchmark.**

Completed so far:
- Development environment and baseline CUT/FastCUT pipeline set up and verified.
- End-to-end GAN training runs executed on satellite image patches (see `train_out.txt` for
  a full training log: 80 epochs, 204 training images, `CUTModel` with a `ResnetGenerator`
  and PatchNCE loss, with GAN and NCE losses logged per iteration).
- Initial experimentation with hyperparameters (learning rate, decay schedule, epoch count)
  and with the ResNet50-backed downstream segmentation head.

**Not yet done / explicitly out of scope for this snapshot:**
- No final quantitative evaluation (no accuracy, IoU, or F1 numbers are reported anywhere in
  this repo — the training log contains only GAN/NCE training losses, not evaluation metrics).
- No side-by-side comparison against a non-adapted baseline.
- Cross-dataset (WHU → INRIA / INRIA → WHU) quantitative evaluation is planned future work.

See the [project proposal](./CS_444_Project_Report.pdf) and [progress report](./CS_444_Project_Progress_Report.pdf)
for the full proposal → progress write-up, including limitations and the planned evaluation.

## Repository Structure

```
data/          Dataset loaders (unaligned/paired image folders) — from CUT
models/        CUT / FastCUT model definitions: generator, discriminator, PatchNCE loss
options/       Command-line option parsing for train/test
experiments/   Launcher scripts for specific experiment configurations
util/         Visualization, HTML report, image pooling utilities
docs/          Dataset documentation (inherited from CUT)
train.py       Main training entrypoint
old_train.py   An earlier variant of the CUT training loop
test.py        Standard CUT test/inference driver
inference.py   Project-side inference script
train_out.txt  Console log from an actual training run (evidence of WIP, not final results)
CS_444_Project_Report.pdf           Project proposal report
CS_444_Project_Progress_Report.pdf  Project progress report
```

Most of `data/`, `models/`, `options/`, `util/`, and `docs/` are inherited from the upstream
CUT / FastCUT codebase (see Acknowledgements). The project-specific contribution is the
application of this framework to satellite imagery domain adaptation: dataset preparation,
training configuration, hyperparameter tuning, and (planned) downstream segmentation
evaluation.

## Setup & Usage

```bash
# Create the environment (Python 3.6, PyTorch 1.4)
conda env create -f environment.yml
conda activate contrastive-unpaired-translation

# or, with pip
pip install -r requirements.txt
```

Training and testing follow the standard CUT/FastCUT CLI conventions:

```bash
# Train (see options/train_options.py for all flags)
python train.py --dataroot ./datasets/<dataset_name> --name <experiment_name> --model cut

# Test / run inference on a trained checkpoint
python test.py --dataroot ./datasets/<dataset_name> --name <experiment_name> --model cut
```

Refer to `docs/datasets.md` for dataset folder conventions (`trainA`/`trainB`,
`testA`/`testB`).

`TODO:` Publish concrete `--dataroot`/`--name` values and exact hyperparameters used for the
run captured in `train_out.txt` (not recorded in the log header available in this repo).

## References

- F. Zhang, Y. Shi, X. X. Zhu. "Self-supervised domain-agnostic domain adaptation for
  satellite images."
- T. Park, A. A. Efros, R. Zhang, J.-Y. Zhu. "Contrastive Learning for Unpaired Image-to-Image
  Translation." ECCV 2020. ([code](https://github.com/taesungp/contrastive-unpaired-translation))
- Y. Koga, H. Miyazaki, R. Shibasaki. "A method for vehicle detection in high-resolution
  satellite images that uses a region-based object detector and unsupervised domain
  adaptation." *Remote Sensing*, 12(575), 2020.
- T. Postadjian, A. Le Bris, H. Sahbi, C. Mallet. "Domain adaptation for large scale
  classification of very high resolution satellite images with deep convolutional neural
  networks." IGARSS 2018.
- [INRIA Aerial Image Labeling Dataset](https://project.inria.fr/aerialimagelabeling/)
- [WHU Building Dataset](http://gpcv.whu.edu.cn/data/building_dataset.html)
- muskaan712, [Domain-Adaptable Self-Supervised Representation Learning on Remote Sensing
  Satellite Imagery](https://github.com/muskaan712/Domain-Adaptable-Self-Supervised-Representation-Learning-on-Remote-Sensing-Satellite-Imagery)
- Levantespot, [UDA_for_RS](https://github.com/Levantespot/UDA_for_RS)

## Acknowledgements

This implementation builds directly on the **CUT / FastCUT** codebase
([taesungp/contrastive-unpaired-translation](https://github.com/taesungp/contrastive-unpaired-translation)),
released under the BSD license by Taesung Park and Jun-Yan Zhu (see `LICENSE`). The `data/`,
`models/`, `options/`, `util/`, and `docs/` directories are largely inherited from that
codebase.

## Author & Collaborators

- **Vaibhav Raheja** — model training and hyperparameter tuning
- Mayank Shetty — data preprocessing and augmentation
- Viraj Shah — evaluation and documentation

Originally developed as an independent research project investigating domain adaptation
techniques for high-resolution satellite imagery.

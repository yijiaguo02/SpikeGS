# SpikeGS: Reconstruct 3D scene captured by a fast-moving bio-inspired camera
Yijia Guo*, Liwen Hu*, Yuanxi Bai, Jiawei Yao, Lei Ma, Tiejun Huang (* indicates equal contribution)<br>
| [Webpage]( https://spikegs.github.io/) | [Arxiv](https://arxiv.org/pdf/2407.03771) |  <br>
This repository contains the official authors implementation associated with the paper "SpikeGS: Reconstruct 3D scene captured by a fast-moving bio-inspired camera", which can be found [here](https://arxiv.org/pdf/2407.03771). We further provide the reference images used to create the error metrics reported in the paper, as well as recently created, pre-trained models. 

<!--<a href="https://www.inria.fr/"><img height="100" src="assets/logo_inria.png"> </a>-->
<!--<a href="https://univ-cotedazur.eu/"><img height="100" src="assets/logo_uca.png"> </a>-->
<!--<a href="https://www.mpi-inf.mpg.de"><img height="100" src="assets/logo_mpi.png"> </a> -->
<!--<a href="https://team.inria.fr/graphdeco/"> <img style="width:100%;" src="assets/logo_graphdeco.png"></a>-->

Abstract: *3D Gaussian Splatting (3DGS) has been proven to exhibit exceptional performance in reconstructing 3D scenes. However, the effectiveness of 3DGS heavily relies on sharp images, and fulfilling this requirement presents challenges in real-world scenarios particularly when utilizing fast-moving cameras. This limitation severely constrains the practical application of 3DGS and may compromise the feasibility of real-time reconstruction.  
To mitigate these challenges, we proposed Spike Gaussian Splatting (SpikeGS), the first framework that integrates the Bayer-pattern spike streams into the 3DGS
pipeline to reconstruct 3D scenes captured by a fast-moving high temporal resolution color spike camera \textbf{in one second}. 
With accumulation rasterization, interval supervision, and a special designed pipeline, SpikeGS realizes continuous spatiotemporal perception while extracts detailed structure and texture from Bayer-pattern spike stream which is unstable and lacks details. 
Extensive experiments on both synthetic and real-world datasets
demonstrate the superiority of SpikeGS compared with existing spike-based and deblurring 3D scene reconstruction methods.*




## Funding and Acknowledgments

This work was supported by National Scienceand Technology Major Project (2022ZD0116305).


## Cloning the Repository

The repository contains submodules, thus please check it out with 
```shell
# SSH
git clone git@github.com:yijiaguo02/spikegs.git --recursive
```
or
```shell
# HTTPS
git clone https://github.com/yijiaguo02/spikegs --recursive
```


## Optimizer

The optimizer uses PyTorch and CUDA extensions in a Python environment to produce trained models. 

### Hardware Requirements

- CUDA-ready GPU with Compute Capability 7.0+
- 24 GB VRAM (80 GB VRAM to train to paper evaluation quality)

### Setup

#### Local Setup

Our default, provided install method is based on Conda package and environment management:
```shell
conda env create --file environment.yml
conda activate spikegs
```

### Running

To run the optimizer, simply use

```shell
python train.py -s <path to prepared data>
```

### Evaluation
By default, the trained models use all available images in the dataset. To train them while withholding a test set for evaluation, use the ```--eval``` flag. This way, you can render training/test sets and produce error metrics as follows:
```shell
python render.py -m <path to trained model> # Generate renderings
python metrics.py -m <path to trained model> # Compute error metrics on renderings
```
### Pre-trained models
Coming soon
<!--If you want to evaluate our [pre-trained models](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/datasets/pretrained/models.zip), you will have to download the corresponding source data sets and indicate their location to ```render.py``` with an additional ```--source_path/-s``` flag. Note: The pre-trained models were created with the release codebase. This code base has been cleaned up and includes bugfixes, hence the metrics you get from evaluating them will differ from those in the paper.-->
<!--```shell-->
<!--python render.py -m <path to pre-trained model> -s <path to COLMAP dataset>-->
<!--python metrics.py -m <path to pre-trained model>-->
<!--```-->
### Datasets
Coming soon

## Processing your own Scenes

Our COLMAP loaders expect the following dataset structure in the source path location:

```
<location>
|---images
|   |---<image>
|   |---<tfp>
|   |---...
|---sparse
    |---0
        |---cameras.bin
        |---images.bin
        |---points3D.bin
```

<section class="section" id="BibTeX">
  <div class="container is-max-desktop content">
    <h2 class="title">BibTeX</h2>
    <pre><code>@article{guo2024spikegs,
  title={SpikeGS: Reconstruct 3D scene via fast-moving bio-inspired sensors},
  author={Guo, Yijia and Hu, Liwen and Ma, Lei and Huang, Tiejun},
  journal={arXiv preprint arXiv:2407.03771},
  year={2024}
}
}</code></pre>
  </div>
</section>




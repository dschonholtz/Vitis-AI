<table class="sphinxhide">
 <tr>
   <td align="center"><img src="https://raw.githubusercontent.com/Xilinx/Image-Collateral/main/xilinx-logo.png" width="30%"/><h1>Vitis AI</h1><h0>Adaptable & Real-Time AI Inference Acceleration</h0>
   </td>
 </tr>
</table>

![Release Version](https://img.shields.io/github/v/release/Xilinx/Vitis-AI)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
![GitHub Pull Requests](https://img.shields.io/github/issues-pr-raw/Xilinx/Vitis-AI)
[![Documentation](https://img.shields.io/badge/documentation-github.IO-blue.svg)](https://xilinx.github.io/Vitis-AI/)
![Repo Size](https://img.shields.io/github/repo-size/Xilinx/Vitis-AI)


<br />
Xilinx&reg; Vitis&trade; AI is an Integrated Development Environment that can be leveraged to accelerate AI inference on Xilinx platforms. Vitis AI provides optimized IP, tools, libraries, models, as well as resources, such as example designs and tutorials that aid the user throughout the development process.  It is designed with high efficiency and ease-of-use in mind, unleashing the full potential of AI acceleration on Xilinx SoCs and Alveo Data Center accelerator cards.  
<br /> <br />


<div align="center">
  <img width="100%" height="100%" src="docsrc/source/docs/reference/images/VAI_IDE.PNG">
</div>
<br />

## Current State of this Fork:

1. This is a project as part of the Northeastern University Wireless Networks and Embedded Systems Lab (Wines Lab)

1. All changes pertaining to that lab can be found in the Vitis-AI/wines dir.

1. The goal of this project is to convert a 1d CNN model to something that can be quantized and run on one of the Vitis-AI boards without degrading model performance. 

1. Currently, we see approximately a .5 to 13% change in classification liklihoods from our softmax on individual channels. 
    1. See wines/quantized_diffs_out.json for the full list of classification changes. To see the individual softmax classifications checkout the other json files in the wines dir. 

1. The next step would be to actually evaluate the model against a test set of data to see how much total classification accuracy declines along with changes in false positives and negatives.

1. The best way to do this would be to push the associated quantized model onto the DGX server, and then to run a test set for the patient the model has been trained on and compare the associated performance.

1. Then we would want to do quantized training on the model with the quantized training pipeline provided in this repo so we could rapidly quantize the model, test it, and quantize it again while leveraging the GPU there.

1. To do this we should run the provided docker container on the DGX, in order to do this though, you must agree to a license agreement found at: Vitis-AI/docker/dockerfiles/PROMPT/PROMPT_gpu.txt.

1. That license agreement should likely be reviewed by someone that has been around a bit longer than me, although that license agreement seems fine and all of the associated installs and changes to the machine should be containerized in docker.

### Makefile Usage

The wines dir has a makefile to assist with data management and model evaluation so that script/path parameters do not have to be remembered for a wide variety of python scripts.

    make test_h5

This tests the original h5 data against the original h5_model before anything is quantized. It also has a side effect of saving all of the data to a dir of numpy files which can be useful later.

    make model_2d

Make model_2d converts the existing 1d h5 model to a 2d h5 model and saves it.

    make quantize_model

This uses the vitis-AI docker container to quantize the 2d h5 model and save it.

    make test_quant

Tests the quantized model and saves the results to a json file for each channel.

    make evaluate_quant_differences

Compares the two generated json files and saves a diff of each of the first values for each softmax of the two networks. 


## Getting Started

If your visit here is accidental, but you are enthusiastic to learn more about Vitis AI, please visit the Vitis AI [homepage](https://www.xilinx.com/products/design-tools/vitis/vitis-ai.html) on Xilinx.com.

Otherwise, if your visit is deliberate and you are ready to begin, why not **[VIEW THE VITIS-AI DOCUMENTATION ON GITHUB.IO](https://xilinx.github.io/Vitis-AI/)**?

## How to Download the Repository

To get a local copy of Vitis AI, clone this repository to the local system with the following command:

```
git clone https://github.com/Xilinx/Vitis-AI
```

This command needs to be executed only once to retrieve the latest version of Vitis AI.

Optionally, configure git-lfs in order to reduce the local storage requirements. 

## Repository Branching and Tagging Strategy

To understand the branching and tagging strategy leveraged by this repository, please refer to [this page](https://xilinx.github.io/Vitis-AI/docs/install/branching_tagging_strategy.html)

## Licenses

Vitis AI License: [Apache 2.0](LICENSE)</br>
Third party: [Components](docsrc/source/docs/reference/Thirdpartysource.md)

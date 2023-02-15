# QUANTIZATION PIPELINE AS DESCRIBED HERE: 
# https://github.com/Xilinx/Vitis-AI/tree/c5d2bd43d951c174185d728b8e5bcda3869e0b39/src/vai_quantizer/vai_q_tensorflow2.x

import tensorflow as tf
from tensorflow import keras
from tensorflow_model_optimization.quantization.keras import vitis_quantize
import os
import numpy as np
import argparse


def quantize_model(model_path, quantized_model_path, numpy_dir):
    float_model = keras.models.load_model(model_path)

    # 1. Create a quantizer with proper quantize_strategy.
    # a) For DPU devices, set quantize_strategy to 'pof2s' to apply power-of-2 scale quantization. This is the default quantize_strategy.
    # b) For other devices supporting floating point arithmetic, set quantize_strategy to 'fs' to apply float scale quantization.
    quantizer = vitis_quantize.VitisQuantizer(float_model, quantize_strategy='pof2s')
    # 2. Call quantizer.quantize_model to do post training quantization.
    # Here calib_dataset is a representative dataset for calibration, you can use full or subset of eval_dataset or train_dataset.
    # See 'quantize_model method' section below for detailed options.
    # Since our model is already trained we shouldn't need calib_dataset unless we see serious accuracy drop

    # load all of the numpy data files in the calib_dataset dir into a single numpy object
    calib_dataset = None
    for f in os.listdir(numpy_dir):
        if f.endswith('.npy'):
            if calib_dataset is not None:
                calib_dataset = np.concatenate((calib_dataset, np.load(os.path.join(numpy_dir, f))))
            else:
                calib_dataset = np.load(os.path.join(numpy_dir, f))
    
    # add a dimension to the dataset in the dimension after batch size
    calib_dataset = np.expand_dims(calib_dataset, axis=1)

    quantized_model = quantizer.quantize_model(
        # output_format='onnx',
        # onnx_opset_version=11,
        output_dir='./quantize_results',
        calib_dataset=calib_dataset,
        calib_batch_size=64,
        add_shape_info=True,
        # include_fast_ft=True,
        # fast_ft_epochs=10,
    )


    # SKIP THIS AS WE ALREADY HAVE A TRAINED MODEL. it is expected to take a long time.
    # from tensorflow_model_optimization.quantization.keras import vitis_quantize
    # quantizer = vitis_quantize.VitisQuantizer(model)
    # qat_model = quantizer.get_qat_model()

    # # Then run the training process with this qat_model to get the quantize finetuned model.
    # qat_model.fit(train_dataset)

    # save the model
    quantizer.get_deploy_model(quantized_model).save(quantized_model_path)
    # quantized_model.save(quantized_model_path)

    model = keras.models.load_model(quantized_model_path)
    quantizer.VitisQuantizer.dump_model(model=quantized_model, 
                                         dataset=calib_dataset,
                                         output_dir='./quantized_dump')

    # path to test dataset
    # eval_dataset = os.path.join(local_dir_path, 'np_data')

    # quantized_model.compile(loss=your_loss, metrics=your_metrics)
    # quantized_model.evaluate(eval_dataset)


def main():
    """
    Pass in arguments with argparse input model path and output quantized model path
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', help='path to input model')
    parser.add_argument('--quantized_model_path', help='path to output quantized model')
    parser.add_argument('--numpy_dir', help='path to calibration dataset')
    args = parser.parse_args()

    quantize_model(args.model_path, args.quantized_model_path, args.numpy_dir)
    

if __name__ == '__main__':
    main()
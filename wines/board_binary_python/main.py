# import vitis_ai_library

# from ctypes import *
# from typing import List
# import cv2
import numpy as np
import vart
import os
# import pathlib
import xir
# import threading
# import time
# import sys
import argparse


divider = '---------------------------'


def get_input_data(input_dir):
    """
    Takes a dir of csvs, pulls out all of the csv files and puts them into a 2d numpy array.
    """
    input_data = []
    for file in os.listdir(input_dir):
        if file.endswith(".csv"):
            input_file = os.path.join(input_dir, file)
            input_chunk = np.loadtxt(input_file, delimiter=',', dtype='float32')
            input_data.append(input_chunk)
    return input_data


# def example():
#     # create graph runner
#     # graph = xir.Graph.deserialize(xmodel_file)
#     # runner = vitis_ai_library.GraphRunner.create_graph_runner(graph)
#     # get input and output tensor buffers
#     input_tensor_buffers = runner.get_inputs()
#     output_tensor_buffers = runner.get_outputs()
#     # run graph runner
#     v = runner.execute_async(input_tensor_buffers, output_tensor_buffers)
#     runner.wait(v)
#     output_data = np.asarray(output_tensor_buffers[0])

def app(input_dir, output_dir, model_path):
    # create graph runner
    input_data = get_input_data(input_dir)
    num_chunks = len(input_data)
    output_list = []

    xmodel_file = model_path
    graph = xir.Graph.deserialize(xmodel_file)
    print(graph)
    subgraphs = graph.get_root_subgraph().toposort_child_subgraph()
    for sub in subgraphs:
        print(f'\n\n\n{sub}')
    runner = vart.Runner.create_runner(subgraphs[1], "run")
    # populate input/output tensors
    # process fpgaOutput
    # get input and output tensor buffers
    input_tensor_buffers = runner.get_input_tensors()
    output_tensor_buffers = runner.get_output_tensors()
    output_data = []
    for output_tensor_buffer in output_tensor_buffers:
        print(output_tensor_buffer)
        output_data.append(np.zeros(output_tensor_buffer.dims, dtype='float32'))
    # run graph runner
    for i in range(num_chunks):
        # put the first input chunk into the input tensor buffer
        print(input_tensor_buffers)
        # input tensor buffer is 1 x 1 x 1024 x 1
        inputData = []
        for inputTensor in input_tensor_buffers:
            inputData.append(input_data[i].reshape(inputTensor.dims))

        jid = runner.execute_async(inputData, output_data)
        runner.wait(jid)
        # output_tensor_buffers = runner.get_outputs()
        # output_data = np.asarray(output_tensor_buffers[0])
        output_list.append(output_data)
        print(f'output_data: {output_data}\n\n\n')
    # save the output to a file
    output_file = os.path.join(output_dir, "output.npy")
    np.save(output_file, output_list)
    return output_list


def main():
    ap = argparse.ArgumentParser()  
    ap.add_argument('-d', '--input_dir', type=str, default='../csv_data', help='Path to folder of seizure data. Default is seizures')
    ap.add_argument('-o', '--output_dir', type=str, default='../results', help='Path to folder of results. Default is results')
    #   ap.add_argument('-t', '--threads',   type=int, default=1,        help='Number of threads. Default is 1')
    ap.add_argument('-m', '--model',     type=str, default='../compile_model/seiznet_tf2.xmodel', help='Path of xmodel. Default is seiznet_tf2.xmodel')

    args = ap.parse_args()  
    print(divider)
    print ('Command line options:')
    print (' --input_dir : ', args.input_dir)
    print (' --output_dir: ', args.output_dir)
    # print (' --threads   : ', args.threads)
    print (' --model     : ', args.model)
    print(divider)

    app(args.input_dir, args.output_dir, args.model)


if __name__ == "__main__":
    main()

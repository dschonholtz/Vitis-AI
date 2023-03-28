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
from utils import get_input_data


divider = '---------------------------'


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
def run_subgraph(subgraph_runner, input_data):
    input_tensor_buffers = subgraph_runner.get_input_tensors()
    output_tensor_buffers = subgraph_runner.get_output_tensors()
    output_list = []
    output_data = []
    for output_tensor_buffer in output_tensor_buffers:
        # print(output_tensor_buffer)
        output_data.append(np.zeros(output_tensor_buffer.dims, dtype='float32'))
    # run graph runner
    for i in range(len(input_data)):
        # put the first input chunk into the input tensor buffer
        # print(input_tensor_buffers)
        # input tensor buffer is 1 x 1 x 1024 x 1
        inputData = []
        # print(f'input_data: {input_data}')
        for inputTensor in input_tensor_buffers:
            inputData.append(np.array(input_data[i]).reshape(inputTensor.dims))

        jid = subgraph_runner.execute_async(inputData, output_data)
        subgraph_runner.wait(jid)
        # output_tensor_buffers = runner.get_outputs()
        # output_data = np.asarray(output_tensor_buffers[0])
        output_list.append(output_data)
        # print(f'output_data: {output_data}\n\n\n')
    return output_list


def app(input_dir, output_dir, model_path):
    # create graph runner
    input_data = get_input_data(input_dir)
    output_list = []

    xmodel_file = model_path
    graph = xir.Graph.deserialize(xmodel_file)
    print(graph)
    subgraphs = graph.get_root_subgraph().toposort_child_subgraph()
    print(len(subgraphs))
    for sub in subgraphs:
        print(f'\n\n\n{sub}')
    print(f'Creating Runner 0 {subgraphs[0].get_name()}')
    # input layer, we can skip it. It is an identity layer
    # runner0 = vart.Runner.create_runner(subgraphs[0], "run")
    print(f'Creating Runner 1 {subgraphs[1].get_name()}')
    # dpu
    dpu_runner1 = vart.Runner.create_runner(subgraphs[1], "run")
    print(f'Creating Runner 2 {subgraphs[2].get_name()}')
    cpu_runner2 = vart.Runner.create_runner(subgraphs[2], "ref")
    print('Creating Runner 3' + subgraphs[3].get_name())
    dpu_runner3 = vart.Runner.create_runner(subgraphs[3], "run")
    print('Creating Runner 4' + subgraphs[4].get_name())
    cpu_runner4 = vart.Runner.create_runner(subgraphs[4], "ref")
    print('Creating Runner 5' + subgraphs[5].get_name())
    dpu_runner5 = vart.Runner.create_runner(subgraphs[5], "run")
    print('Creating Runner 6' + subgraphs[6].get_name())
    cpu_runner6 = vart.Runner.create_runner(subgraphs[6], "ref")
    # populate input/output tensors
    # process fpgaOutput
    # get input and output tensor buffers
    # output = run_subgraph(runner0, input_data)
    output = run_subgraph(dpu_runner1, input_data)
    output = run_subgraph(cpu_runner2, output)
    output = run_subgraph(dpu_runner3, output)
    output = run_subgraph(cpu_runner4, output)
    output = run_subgraph(dpu_runner5, output)
    output = run_subgraph(cpu_runner6, output)

    print(f'output6: {output}\n\n\n')

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

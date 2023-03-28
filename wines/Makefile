# A makefile to help do various model manipulations
# ASSUMES YOU ARE RUNNING IN THE VITIS AI DOCKER CONTAINER

SHELL := /bin/bash

# get the root dir of the vectorblox repo one level up from the current dir
ROOT_DIR=$(shell dirname $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST)))))

# get the current dir and save it as a var
WINES_DIR=$(shell pwd)
NP_DIR="$(WINES_DIR)/np_data"
CSV_DIR="$(WINES_DIR)/csv_data"
H5_DATA_PATH="$(WINES_DIR)/preseiz-13089010103_41368_2_sample9.h5"
H5_MODEL_PATH="$(WINES_DIR)/pat_25302.pruneCNN12.h5"
2D_MODEL_PATH="$(WINES_DIR)/model_2d"
QUANTIZED_PATH="$(WINES_DIR)/quantized_model.h5"
ORIGINAL_RESULTS="${WINES_DIR}/h5_model_out.json"
QUANTIZED_RESULTS="${WINES_DIR}/quantized_model_out.json"
BOARD_IP="10.42.0.42"


# assumed to be the vitis-AI docker container. Also worth activating the conda env: (vitis-ai-tensorflow2)
# do this with ./docker_run.sh xilinx/vitis-ai-tensorflow2-cpu:latest
# conda activate vitis-ai-tensorflow2
is_docker:
	@echo "Checking if running in docker container"
	@if [ -f /.dockerenv ]; then \
		echo "Running in docker container"; \
	else \
		echo "Not running in docker container. Pull and run the xilinx/vitis-ai-tensorflow2-cpu:latest or equivilant model"; \
		exit 1; \
	fi

is_not_docker:
	@echo "Checking if running in docker container"
	@if [ -f /.dockerenv ]; then \
		echo "Running in docker container. Run this outside of the docker container. Using the vectorblox env works neatly."; \
		exit 1; \
	else \
		echo "Not running in docker container. Yay"; \
	fi

# do cleanup
clean:
	cd ${WINES_DIR} && bash cleanup.sh

npy_to_csv:
	cd ${WINES_DIR} && python3 np_to_csv.py --input_dir "${NP_DIR}" --output_dir "${CSV_DIR}" --replace_existing

# Test h5 has a side effect of creating the input data.
# that needs to be cleaned up, but for now, it is a dependency
# of other testing.
# Test_h5 is run with a non-quantized model which doesn't work in the docker container.
test_h5: is_not_docker
	python3 test_h5_model.py \
		--input_data "${H5_DATA_PATH}" \
		--input_npy_dir "${NP_DIR}" \
		--model "${H5_MODEL_PATH}" \
		--output_file "${ORIGINAL_RESULTS}" \
		--nuke_data FALSE

# we need a 2d model to run in the docker container
model_2d: is_not_docker
	cd ${WINES_DIR} && \
		python3 save_model.py \
			--model_1d "${H5_MODEL_PATH}" \
			--out_model_2d "${2D_MODEL_PATH}"

quantize_model: is_docker #test_h5
	cd ${WINES_DIR} && \
		python3 quantize.py \
			--model_path "${2D_MODEL_PATH}" \
			--quantized_model_path "${QUANTIZED_PATH}" \
			--numpy_dir "${NP_DIR}" 

test_quant: is_docker #quantize_model
	python3 test_h5_model.py \
		--input_data "${H5_DATA_PATH}" \
		--input_npy_dir "${NP_DIR}" \
		--model "${QUANTIZED_PATH}" \
		--output_file "${QUANTIZED_RESULTS}" \
		--nuke_data "FALSE" \
		--expand_dims "TRUE"

evaluate_quant_differences:
	python3 evaluate_quant_differences.py \
		--quantized_json "${QUANTIZED_RESULTS}" \
		--pre_quant_json "${ORIGINAL_RESULTS}" \
		--output_json "${WINES_DIR}/quantized_diffs_out.json"

compile_VART: is_docker
	vai_c_tensorflow2 -m ${QUANTIZED_PATH} \
		--arch /opt/vitis_ai/compiler/arch/DPUCZDX8G/KV260/arch.json       \
		--output_dir compile_model                                       \
		--net_name seiznet_tf2

# I haven't used this yet.
test_xmodel: is_docker
	env XLNX_ENABLE_DUMP=1 XLNX_ENABLE_DEBUG_MODE=1 XLNX_GOLDEN_DIR=./dump_gpu/dump_results_0 \
		xdputil run ./compile_model/resnet_v1_50_tf.xmodel            \
		./dump_gpu/dump_results_0/input_aquant.bin                    \
		2>result.log 1>&2


default:
	make test_quant


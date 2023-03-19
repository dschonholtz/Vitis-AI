"""
Takes all of the data in np_data and turns each npy file to a csv file and copies it into the output_dir.
"""

import os
import numpy as np
import shutil


def save_np_data_to_dir(np_data, output_dir, replace_existing=False):
    # nuke the existing data if it is there.
    if replace_existing:
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
    elif os.path.exists(output_dir):
        print("Output directory already exists.  Not replacing.")
        return
    # make the dir
    os.makedirs(output_dir, exist_ok=True)
    for i, chunk in enumerate(np_data):
        input_file = str(i) + ".csv"
        chunk = chunk.reshape(1024)
        input_chunk_name = os.path.join(output_dir, input_file)
        # add a dim to the chunk
        chunk = np.expand_dims(chunk, 0)
        np.savetxt(input_chunk_name, chunk, delimiter=",")


def read_np_data_from_dir(input_dir):
    input_data = []
    for file in os.listdir(input_dir):
        if file.endswith(".npy"):
            input_file = os.path.join(input_dir, file)
            # load npy file
            input_chunk = np.load(input_file)
            input_data.append(input_chunk)
    return input_data


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, default="np_data")
    parser.add_argument("--output_dir", type=str, default="csv_data")
    parser.add_argument("--replace_existing", action="store_true", default=True)
    args = parser.parse_args()
    np_data = read_np_data_from_dir(args.input_dir)
    save_np_data_to_dir(np_data, args.output_dir, args.replace_existing)
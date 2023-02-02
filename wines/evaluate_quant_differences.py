import json
import argparse


def json_diffs(json1, json2):
    """
    Given to jsons with the form [[[float, float], ]]
    return the differences between the two.
    """
    # load the jsons
    with open(json1, 'r') as f:
        json1 = json.load(f)
    with open(json2, 'r') as f:
        json2 = json.load(f)

    # get the differences
    diffs = []
    for i, chunk in enumerate(json1):
        for j, sample in enumerate(chunk):
            diffs.append(sample[0] - json2[i][j][0])
    
    return diffs


def main():
    """
    Evaluate the differences between the given quantized model and the given h5 model json softmax classification accuracies.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--quantized_json', help='path to quantized json file')
    parser.add_argument('--pre_quant_json', help='path to json results file before quantization')
    parser.add_argument('--output_json', help='path to output json file')
    args = parser.parse_args()

    diffs = json_diffs(args.quantized_json, args.pre_quant_json)
    max_val = max(diffs)
    min_val = min(diffs)
    json_data = {
        "max": max_val,
        "min": min_val,
        "diffs": diffs
    }
    # save the diffs list to a json file
    with open(args.output_json, 'w') as f:
        json.dump(json_data, f)


if __name__ == "__main__":
    main()
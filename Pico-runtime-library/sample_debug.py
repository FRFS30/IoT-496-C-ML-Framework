
# Testing file to be sure the bin file produced by server_csv.py properly preserved the samples
import struct

FILE = "samples.bin"

NUM_FEATURES = 24
SAMPLE_SIZE = 97


def read_sample(f):

    data = f.read(SAMPLE_SIZE)

    if len(data) != SAMPLE_SIZE:
        return None
    features = struct.unpack(f"{NUM_FEATURES}f",data[:96])
    label = struct.unpack("B",data[96:97])[0]
    return features, label


with open(FILE, "rb") as f:

    for i in range(5):

        result = read_sample(f)

        if result is None:
            break

        features, label = result

        print(f"\nSample {i}")

        print("Features:")

        for j in range(24): 
            print(features[j])

        print("Label:", label)
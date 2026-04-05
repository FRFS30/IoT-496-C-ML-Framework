#this python file converts the csv of the testing datset into a bin file the tcp server can send to the microcontroller

import csv
import struct

INPUT_CSV = "clean_dataset_new.csv"
OUTPUT_BIN = "samples.bin"


NUM_FEATURES = 24
SAMPLE_SIZE = NUM_FEATURES * 4 + 1  # 97 bytes

count = 0
skipped = 0

def safe_float(x):
    """Convert string to float safely."""
    x = x.strip()

    if x == "":
        return 0.0   # replace empty with 0

    return float(x)


with open(INPUT_CSV, newline='') as csvfile, \
     open(OUTPUT_BIN, "wb") as binfile:

    reader = csv.reader(csvfile)

    header = next(reader) #Skip header row

    print("Header columns:", len(header))

    for row_num, row in enumerate(reader, start=2):

        # Skip malformed rows
        if len(row) < NUM_FEATURES + 1:
            skipped += 1
            continue

        try:

            # Convert features safely
            features = [
                safe_float(x)
                for x in row[:NUM_FEATURES]
            ]

            # Convert label
            label_str = row[-1].strip().upper()

            if label_str == "ATTACK":
                label = 1

            elif label_str == "BENIGN":
                label = 0

            else:
                skipped += 1
                continue

            # Pack floats
            packed_features = struct.pack(f"{NUM_FEATURES}f",*features)

            # Pack label
            packed_label = struct.pack("B", label)

            # Write binary sample
            binfile.write(
                packed_features +
                packed_label
            )

            count += 1

        except Exception as e:

            print(
                f"Skipping row {row_num}: {e}"
            )

            skipped += 1


print(f"Samples written: {count}")
print(f"Rows skipped: {skipped}")
print(f"Sample size: {SAMPLE_SIZE} bytes")
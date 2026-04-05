

#OUTDATED. Make samples in the csv into a samples C header. obsoleted by the active sample stream

import csv

CSV_FILE = "clean_dataset_new.csv"
NUM_SAMPLES = 1000
OUTPUT_C_FILE = "rf_samples.h"

FEATURE_COLUMNS = [
    "Flow_Duration",
    "Fwd_Packet_Length_Mean",
    "Flow_Bytes_per_s",
    "Fwd_IAT_Mean",
    "Bwd_IAT_Mean",
    "Fwd_PSH_Flags",
    "Fwd_URG_Flags",
    "Fwd_Header_Length",
    "Bwd_Header_Length",
    "Min_Packet_Length",
    "Max_Packet_Length",
    "FIN_Flag_Count",
    "PSH_Flag_Count",
    "ACK_Flag_Count",
    "URG_Flag_Count",
    "ECE_Flag_Count",
    "Down_per_Up_Ratio",
    "Init_Win_bytes_forward",
    "Init_Win_bytes_backward",
    "Active_Mean",
    "Active_Std",
    "Active_Max",
    "Idle_Std",
    "Idle_Max"
]

LABEL_COLUMN = "Label"


def safe_float(value):
    value = value.strip()
    if value == "":
        return 0.0
    return float(value)


def label_to_int(label):
    return 0 if label.strip().upper() == "BENIGN" else 1


def csv_to_c_array(csv_file, num_samples, output_file):
    samples = []
    labels = []

    with open(csv_file, "r", newline="") as f:
        reader = csv.DictReader(f)

        for i, row in enumerate(reader):
            if i >= num_samples:
                break

            sample = [safe_float(row[col]) for col in FEATURE_COLUMNS]
            label = label_to_int(row[LABEL_COLUMN])

            samples.append(sample)
            labels.append(label)

    actual_samples = len(samples)

    with open(output_file, "w") as f:
        f.write("#ifndef RF_SAMPLES_H\n")
        f.write("#define RF_SAMPLES_H\n\n")

        f.write(f"#define NUM_SAMPLES {actual_samples}\n")
        f.write("#define NUM_FEATURES 24\n\n")

        f.write("static const float samples[NUM_SAMPLES][NUM_FEATURES] = {\n")
        for sample in samples:
            values = ", ".join(f"{x:.6f}f" for x in sample)
            f.write(f"    {{{values}}},\n")
        f.write("};\n\n")

        f.write("static const int labels[NUM_SAMPLES] = {\n    ")
        f.write(", ".join(map(str, labels)))
        f.write("\n};\n\n")

        f.write("#endif\n")

    print(f"Generated {output_file} with {actual_samples} samples.")


if __name__ == "__main__":
    csv_to_c_array(CSV_FILE, NUM_SAMPLES, OUTPUT_C_FILE)
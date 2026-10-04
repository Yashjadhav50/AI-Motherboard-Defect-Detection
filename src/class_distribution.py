import os
from collections import Counter

label_folder = "dataset/train/labels"

class_names = [
    "CPU_FAN_NO_Screws",
    "CPU_FAN_Screw_loose",
    "CPU_FAN_Screws",
    "CPU_fan",
    "CPU_fan_port",
    "CPU_fan_port_detached",
    "Incorrect_Screws",
    "Loose_Screws",
    "No_Screws",
    "Scratch",
    "Screws"
]

counter = Counter()

# Read every label file
for filename in os.listdir(label_folder):

    if filename.endswith(".txt"):

        file_path = os.path.join(label_folder, filename)

        with open(file_path, "r") as file:

            for line in file:

                values = line.strip().split()

                if values:
                    class_id = int(values[0])
                    counter[class_id] += 1


print("\n===== CLASS DISTRIBUTION =====\n")

for class_id, class_name in enumerate(class_names):

    count = counter[class_id]

    print(f"{class_id:2} | {class_name:25} | {count}")

print("\nTotal annotations:", sum(counter.values()))
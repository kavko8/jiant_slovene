import json
import os


for mode in ["machine_translation"]:
    tasks_list = ["boolq", "cb", "copa", "multirc", "rte"]
    for task in tasks_list:
        if not os.path.isdir(f"{os.getcwd()}/tasks/configs/{mode}"):
            os.makedirs(f"{os.getcwd()}/tasks/configs/{mode}")
        json_file = {
            "task": f"{task}",
            "paths": {
                "train": f"{os.getcwd()}/tasks/data/{mode}/{task}/train.jsonl",
                "val": f"{os.getcwd()}/tasks/data/{mode}/{task}/val.jsonl",
                "test": f"{os.getcwd()}/tasks/data/{mode}/{task}/test.jsonl",
                "val_test": f"{os.getcwd()}/tasks/data/{mode}/{task}/val_test.jsonl",
            },
            "name": f"{task}"
        }
        with open(f"{os.getcwd()}/tasks/configs/{mode}/{task}_config.json", "w") as f:
            json.dump(json_file, f)

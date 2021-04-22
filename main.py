import jiant.proj.main.tokenize_and_cache as tokenize_and_cache
import jiant.proj.main.export_model as export_model
import jiant.proj.main.scripts.configurator as configurator
import jiant.proj.main.runscript as main_runscript
import jiant.shared.caching as caching
import jiant.utils.python.io as py_io
import jiant.utils.display as display
import os
import benchmark_submission_formatter

pretrained_models = {
    "sloberta": "./models/pretrained/sloberta",
    "crosloengual": "EMBEDDIA/crosloengual-bert",
    "multilingual": "bert-base-multilingual-cased",
    "roberta": "roberta-base",
}

pretrained = pretrained_models["crosloengual"]
output_name = list(pretrained_models.keys())[list(pretrained_models.values()).index(pretrained)]
human_translation = True
name = "human_translation" if human_translation else "machine_translation"
tasks = ["record_lemma", "record"]  # ["cb", "copa", "multirc", "record", "rte", "wsc", "boolq"]

train_batch_size = 1
eval_batch_size = 1
epochs = 0.005
num_gpus = 1
max_seq_length = 256
learning_rate = 1e-5
optimizer_type = "adam"  # radam
adam_epsilon = 1e-8
max_grad_norm = 1.0
eval_every_steps = 500
no_improvements_for_n_evals = 0
eval_subset_num = None
model_load_mode = "from_transformers"

do_train = True
do_val = True
validate_test = True
force_overwrite = True
write_test_preds = True
write_val_preds = True
write_val_test_preds = True
do_save = False
do_save_best = True

do_save_last = False
smart_truncate = True
do_iter = True
load_best_model = True

phases = []
if do_train:
    phases.append("train")
if do_val:
    phases.append("val")
if write_test_preds:
    phases.append("test")
if validate_test:
    phases.append("val_test")


export_model.export_model(
    hf_pretrained_model_name_or_path=pretrained,
    output_base_path=f"./models/{output_name}",
)


# Tokenize and cache each task
for task_name in tasks:
    tokenize_and_cache.main(tokenize_and_cache.RunConfiguration(
        task_config_path=f"./tasks/{name}/configs/{task_name}_config.json",
        hf_pretrained_model_name_or_path=pretrained,
        output_dir=f"./cache/{task_name}",
        phases=phases,
        do_iter=do_iter,
        smart_truncate=smart_truncate,
        max_seq_length=max_seq_length,
    ))

for task_name in tasks:
    no_tokens = False
    row = caching.ChunkedFilesDataCache(f"./cache/{task_name}/train").load_chunk(0)[0]["data_row"]
    print(row.input_ids)
    try:
        print(row.tokens)
    except AttributeError:
        no_tokens = True
        print("Tokens not in row. Trying row.tokens_list")
    if no_tokens:
        try:
            print(row.tokens_list)
        except AttributeError:
            print(f"Tokens_list not in row. ({task_name.upper()})")


jiant_run_config = configurator.SimpleAPIMultiTaskConfigurator(
    task_config_base_path=f"./tasks/{name}/configs",
    task_cache_base_path="./cache",
    train_task_name_list=tasks,
    val_task_name_list=tasks,
    test_task_name_list=tasks,
    val_test_task_name_list=tasks,
    train_batch_size=train_batch_size,
    eval_batch_size=eval_batch_size,
    epochs=epochs,
    num_gpus=num_gpus,
    eval_subset_num=eval_subset_num,
).create_config()


os.makedirs("./run_configs/", exist_ok=True)
os.makedirs(f"./runs/{output_name}/{name}", exist_ok=True)
py_io.write_json(jiant_run_config, "./run_configs/jiant_run_config.json")
display.show_json(jiant_run_config)

run_args = main_runscript.RunConfiguration(
    jiant_task_container_config_path="./run_configs/jiant_run_config.json",
    output_dir=f"./runs/{output_name}/{name}",
    model_load_mode=model_load_mode,
    hf_pretrained_model_name_or_path=pretrained,
    #  model_path="./runs/sloberta/human_translation/best_model.p",
    model_path=f"./models/{output_name}/model/model.p",
    model_config_path=f"./models/{output_name}/model/config.json",
    learning_rate=learning_rate,
    eval_every_steps=eval_every_steps,
    do_train=do_train,
    do_val=do_val,
    do_val_test=validate_test,
    force_overwrite=force_overwrite,
    write_test_preds=write_test_preds,
    write_val_preds=write_val_preds,
    write_val_test_preds=write_val_test_preds,
    do_save_best=do_save_best,
    do_save_last=do_save_last,
    do_save=do_save,
    no_cuda=True if not num_gpus else False,
    no_improvements_for_n_evals=no_improvements_for_n_evals,
    adam_epsilon=adam_epsilon,
    max_grad_norm=max_grad_norm,
    optimizer_type=optimizer_type,
    load_best_model=load_best_model,
)

main_runscript.run_loop(run_args)

benchmark_submission_formatter.results(
    benchmark="SUPERGLUE",
    input_base_path=f"./runs/{output_name}/{name}",
    output_path=f"./runs/{output_name}/{name}",
    task_names=tasks,
    preds="test_preds.p",
    regime="test",
)

benchmark_submission_formatter.results(
    benchmark="SUPERGLUE",
    input_base_path=f"./runs/{output_name}/{name}",
    output_path=f"./runs/{output_name}/{name}",
    task_names=tasks,
    preds="val_test_preds.p",
    regime="val_test",
)

benchmark_submission_formatter.results(
    benchmark="SUPERGLUE",
    input_base_path=f"./runs/{output_name}/{name}",
    output_path=f"./runs/{output_name}/{name}",
    task_names=tasks,
    preds="val_preds.p",
    regime="val",
)
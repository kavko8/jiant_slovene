import os

import jiant.proj.main.preprocessing as preprocessing
import jiant.shared.caching as shared_caching
import jiant.shared.model_resolution as model_resolution
import jiant.shared.model_setup as model_setup
import jiant.tasks as tasks
import jiant.tasks.evaluate as evaluate
import jiant.utils.zconf as zconf
import jiant.utils.python.io as py_io
from jiant.shared.constants import PHASE


@zconf.run_config
class RunConfiguration(zconf.RunConfig):
    # === Required parameters === #
    task_config_path = zconf.attr(type=str, required=True)
    model_type = zconf.attr(type=str, required=True)
    model_tokenizer_path = zconf.attr(type=str, required=True)
    output_dir = zconf.attr(type=str, required=True)

    # === Optional parameters === #
    phases = zconf.attr(default="train,val", type=str)
    max_seq_length = zconf.attr(default=128, type=int)
    chunk_size = zconf.attr(default=10000, type=int)
    smart_truncate = zconf.attr(action="store_true")
    do_iter = zconf.attr(action="store_true")
    skip_write_output_paths = zconf.attr(action="store_true")


def chunk_and_save(task, phase, examples, feat_spec, tokenizer, args: RunConfiguration):
    """Convert Examples to DataRows, optionally truncate sequences if possible, and save to disk.

    Note:
        If args.do_iter is True, processes data without loading whole dataset into memory.

    Args:
        task: Task object
        phase (str): string identifying the data subset (e.g., train, val or test).
        examples (list[Example]): list of task Examples.
        feat_spec: (FeaturizationSpec): Tokenization-related metadata.
        tokenizer: TODO  (issue #1188)
        args (RunConfiguration): run configuration object.

    """
    if args.do_iter:
        iter_chunk_and_save(
            task=task,
            phase=phase,
            examples=examples,
            feat_spec=feat_spec,
            tokenizer=tokenizer,
            args=args,
        )
    else:
        full_chunk_and_save(
            task=task,
            phase=phase,
            examples=examples,
            feat_spec=feat_spec,
            tokenizer=tokenizer,
            args=args,
        )


def full_chunk_and_save(task, phase, examples, feat_spec, tokenizer, args: RunConfiguration):
    """Convert Examples to ListDataset, optionally truncate sequences if possible, and save to disk.

    Args:
        task: Task object
        phase (str): string identifying the data subset (e.g., train, val or test).
        examples (list[Example]): list of task Examples.
        feat_spec: (FeaturizationSpec): Tokenization-related metadata.
        tokenizer: TODO  (issue #1188)
        args (RunConfiguration): run configuration object.

    """
    dataset = preprocessing.convert_examples_to_dataset(
        task=task,
        examples=examples,
        feat_spec=feat_spec,
        tokenizer=tokenizer,
        phase=phase,
        verbose=True,
    )
    if args.smart_truncate:
        dataset, length = preprocessing.smart_truncate(
            dataset=dataset, max_seq_length=args.max_seq_length, verbose=True,
        )
        os.makedirs(os.path.join(args.output_dir, phase), exist_ok=True)
        py_io.write_json(
            data={"truncated_to": int(length)},
            path=os.path.join(args.output_dir, phase, "smart_truncate.json"),
        )
    shared_caching.chunk_and_save(
        data=dataset.data,
        chunk_size=args.chunk_size,
        data_args=args.to_dict(),
        output_dir=os.path.join(args.output_dir, phase),
    )


def iter_chunk_and_save(task, phase, examples, feat_spec, tokenizer, args: RunConfiguration):
    """Convert Examples to DataRows, optionally truncate sequences if possible, stream to disk.

    Args:
        task: Task object
        phase (str): string identifying the data subset (e.g., train, val or test).
        examples (list[Example]): list of task Examples.
        feat_spec: (FeaturizationSpec): Tokenization-related metadata.
        tokenizer: TODO  (issue #1188)
        args (RunConfiguration): run configuration object.

    """
    dataset_generator = preprocessing.iter_chunk_convert_examples_to_dataset(
        task=task,
        examples=examples,
        feat_spec=feat_spec,
        tokenizer=tokenizer,
        phase=phase,
        verbose=True,
    )
    max_valid_length_recorder = preprocessing.MaxValidLengthRecorder(args.max_seq_length)
    shared_caching.iter_chunk_and_save(
        data=dataset_generator,
        chunk_size=args.chunk_size,
        data_args=args.to_dict(),
        output_dir=os.path.join(args.output_dir, phase),
        recorder_callback=max_valid_length_recorder,
    )
    if args.smart_truncate:
        preprocessing.smart_truncate_cache(
            cache=shared_caching.ChunkedFilesDataCache(os.path.join(args.output_dir, phase)),
            max_seq_length=args.max_seq_length,
            max_valid_length=max_valid_length_recorder.max_valid_length,
            verbose=True,
        )
        py_io.write_json(
            data={"truncated_to": int(max_valid_length_recorder.max_valid_length)},
            path=os.path.join(args.output_dir, phase, "smart_truncate.json"),
        )


def main(args: RunConfiguration):
    task = tasks.create_task_from_config_path(config_path=args.task_config_path, verbose=True)
    if isinstance(args.phases, str):
        phases = args.phases.split(",")
    else:
        phases = args.phases
    assert set(phases) <= {PHASE.TRAIN, PHASE.VAL, PHASE.TEST}

    paths_dict = {}
    os.makedirs(args.output_dir, exist_ok=True)

    if PHASE.TEST in phases:
        val_ex = task.get_val_examples()
        test_examples = task.get_test_examples()
    task_type = args.output_dir.rsplit('/',1)[1].strip()
    test_ex_dict = {}
    for idx, example in enumerate(test_examples):
        if task_type != "mctaco":
            example.guid = task_type + "_" + str(idx)
            test_ex_dict[example.guid] = example.__dict__
        else:
            #test-q1330-9437
            guid = example.guid.split('-')[1][1:]
            #example.guid = guid
            if guid not in test_ex_dict:
                test_ex_dict[guid] = []
            test_ex_dict[guid].append(example.__dict__)

    if not args.skip_write_output_paths:
        fname = task_type+ "_" +"raw_test_examples.json"
        py_io.write_json(data=test_ex_dict, path=os.path.join('/scratch/pmh330/nyu-mll-jiant/raw_examples',fname)) 

if __name__ == "__main__":
    main(args=RunConfiguration.run_cli_json_prepend())

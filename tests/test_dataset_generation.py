import json

import pytest

from AlgorithmsBlock import get_template, list_algorithms
from scripts.generate_sft_dataset import (
    ALGORITHMS,
    generate_dataset,
    gen_cipherspec_example,
    gen_improvement_example,
)


@pytest.mark.parametrize("algo_name", list_algorithms())
def test_spec_examples_match_reference_templates(algo_name):
    example = gen_cipherspec_example(algo_name, seed=123)
    assistant = json.loads(example["messages"][2]["content"])
    assert assistant["components"] == get_template(algo_name, seed=123).components


@pytest.mark.parametrize("algo_name", list_algorithms())
def test_patch_examples_restore_reference_components(algo_name):
    example = gen_improvement_example(algo_name, seed=321)
    patch = json.loads(example["messages"][2]["content"])
    reference = get_template(algo_name, seed=321).components

    assert patch["replace_components"]
    assert patch["new_rounds"] in ALGORITHMS[algo_name]["rounds"]
    for name, value in patch["replace_components"].items():
        assert reference[name] == value


def test_generate_dataset_respects_requested_sizes():
    train, valid = generate_dataset(n_train=20, n_valid=10, seed=7)
    assert len(train) == 20
    assert len(valid) == 10

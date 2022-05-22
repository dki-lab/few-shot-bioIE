# coding=utf-8
# Copyright 2020 HuggingFace Datasets Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as: python3
"""NCBI disease corpus: a resource for disease name recognition and concept normalization"""

import datasets
import wandb
logger = datasets.logging.get_logger(__name__)

class NERSingleConfig(datasets.BuilderConfig):
    """BuilderConfig for NERSingle"""

    def __init__(self, **kwargs):
        """BuilderConfig for NERSingle.
        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(NERSingleConfig, self).__init__(**kwargs)


class NERSingleWandB(datasets.GeneratorBasedBuilder):
    """NERSingle dataset."""

    BUILDER_CONFIGS = [
        NERSingleConfig(name=wandb.config.data_name, version=datasets.Version('1.0.{}'.format(wandb.config.subset_num)), description="NERSingle dataset"),
    ]

    def _info(self):
        return datasets.DatasetInfo(
            features=datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "tokens": datasets.Sequence(datasets.Value("string")),
                    "ner_tags": datasets.Sequence(
                        datasets.features.ClassLabel(
                            names=[
                                "O",
                                "B",
                                "I",
                            ]
                        )
                    ),
                }
            ),
            supervised_keys=None
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""

        return [
            datasets.SplitGenerator(name=datasets.Split.TRAIN, gen_kwargs={"filepath": wandb.config.training_file}),
            datasets.SplitGenerator(name=datasets.Split.VALIDATION, gen_kwargs={"filepath": wandb.config.dev_file}),
            datasets.SplitGenerator(name=datasets.Split.TEST, gen_kwargs={"filepath": wandb.config.test_file}),
        ]

    def _generate_examples(self, filepath):
        logger.info("⏳ Generating examples from = %s", filepath)
        with open(filepath, encoding="utf-8") as f:
            guid = 0
            line_id = 0
            tokens = []
            ner_tags = []
            for line in f:
                if line_id == 0:
                    line_id += 1
                    continue

                if line == "" or line == "\n":
                    if tokens:
                        yield guid, {
                            "id": str(guid),
                            "tokens": tokens,
                            "ner_tags": ner_tags,
                        }
                        guid += 1
                        tokens = []
                        ner_tags = []
                else:
                    # tokens are tab separated
                    splits = line.split("\t")
                    tokens.append(splits[0])
                    ner_tags.append(splits[1].rstrip())

            if len(tokens) > 0:
                # last example
                yield guid, {
                    "id": str(guid),
                    "tokens": tokens,
                    "ner_tags": ner_tags,
                }
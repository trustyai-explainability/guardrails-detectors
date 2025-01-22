from .base_dataset_config import BaseDatasetConfig


class MMLUDatasetConfig(BaseDatasetConfig):
    """Config for defining text and label pairs from MMLU"""

    def __init__(self):
        super().__init__()

    def get_text(self, docs):
        """Define a function to extract the "text" from each row of the dataset."""
        qs = docs['question']
        ans = docs['answer']
        cs = docs['choices']
        return ["{}\n\n{}".format(qs[i], cs[i][ans[i]]) for i in range(len(docs))]

    def get_label(self, docs):
        """Define a function to extract the label from each row of the dataset."""
        return docs['subject']
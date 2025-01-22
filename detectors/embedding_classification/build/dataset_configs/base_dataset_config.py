class BaseDatasetConfig():
    """Base Config for defining text and label pairs from a Huggingface text dataset"""

    def __init__(self):
        pass

    def get_text(self, docs):
        """Define a function to extract the "text" from each row of the dataset."""
        raise NotImplementedError

    def get_label(self, docs):
        """Define a function to extract the label from each row of the dataset."""
        raise NotImplementedError
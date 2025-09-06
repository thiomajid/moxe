import grain.python as grain
from transformers.data.data_collator import DataCollatorForLanguageModeling


class CollateForLanguageModeling(grain.MapTransform):
    """
    Applies a collator to a dataset element and converts the specified columns to **JAX** arrays.
    This transform uses a Hugging Face **DataCollatorForLanguageModeling** to process a dataset element,
    then converts the specified columns to **JAX** arrays, removing any other columns.

    Attributes:
        collator: A Hugging Face DataCollatorForLanguageModeling instance.
        target_columns: A list of strings representing the columns to keep and convert to JAX arrays.
    """

    def __init__(
        self,
        collator: DataCollatorForLanguageModeling,
        target_columns: list[str],
    ):
        super().__init__()

        self.collator = collator
        self.target_columns = target_columns

    def map(self, element):
        if not isinstance(element, list):
            element = [element]

        return self.collator(element)

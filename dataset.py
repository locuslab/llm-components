import random
import numpy as np
from tqdm import tqdm
import re

from datasets import load_dataset


class TwoDigitArithmeticDataset:
    def __init__(self, arithmetic_type, modeltype, include_reversed=False):
        """
        Initializes the dataset.

        Args:
            include_reversed (bool, optional): If True, ensures that for each pair
                                               (a, b) where a != b, both the prompts
                                               "a × b=" and "b × a=" are included
                                               in the dataset (each exactly once).
                                               Defaults to False, which includes only
                                               "a × b=" for each pair encountered.
        """
        self.include_reversed = include_reversed

        temp_data = []
        pairs_set = set() # Use a set to track added pairs for uniqueness

        for i in range(100):
            for j in range(100):
                pair = (i, j)
                reversed_pair = (j, i)

                # Add the primary pair (i, j) if it hasn't been added
                if pair not in pairs_set:
                    temp_data.append(pair)
                    pairs_set.add(pair)

                # If reversed is requested and i!=j, ensure the reversed pair is also added
                if include_reversed and i != j:
                    if reversed_pair not in pairs_set:
                        temp_data.append(reversed_pair)
                        pairs_set.add(reversed_pair)

        self.data = temp_data # Assign the unique pairs list
        # import pdb; pdb.set_trace()
        max_token = 1000
        if arithmetic_type == "multiplication":
            self.prompts = [f"{a} x {b} =" for a, b in self.data if a * b < max_token]
            self.answers = [str(a * b) for a, b in self.data if a * b < max_token]
        elif arithmetic_type == "addition":
            self.prompts = [f"{a} + {b} =" for a, b in self.data]
            self.answers = [str(a + b) for a, b in self.data]
        elif arithmetic_type == "subtraction":
            self.prompts = [f"{b} - {a} =" for a, b in self.data]
            self.answers = [str(b - a) for a, b in self.data]
        elif arithmetic_type == "division":
            products = [(a, b, a * b) for a, b in self.data if a * b < max_token]
            products = [p for p in products if p[2] != 0]
            self.prompts = [f"{c} / {a} = " for a, b, c in products]
            self.answers = [str(b) for a, b, c in products]
        elif arithmetic_type == "all":
            random.shuffle(self.data)
            subset_size = len(self.data) // 4
            add_set = self.data[0:subset_size]
            sub_set = self.data[subset_size:subset_size*2]
            mul_set = self.data[subset_size*2:subset_size*3]
            div_set = self.data[subset_size*3:]
            add_prompts = [f"{a} + {b} =" for a, b in add_set]
            add_answers = [str(a + b) for a, b in add_set]
            sub_prompts = [f"{b} - {a} =" for a, b in sub_set]
            sub_answers = [str(b - a) for a, b in sub_set]
            mul_prompts = [f"{a} x {b} =" for a, b in mul_set if a * b < max_token]
            mul_answers = [str(a * b) for a, b in mul_set if a * b < max_token]
            products = [(a, b, a * b) for a, b in div_set if a * b < max_token]
            products = [p for p in products if p[2] != 0]
            div_prompts = [f"{c} / {a} = " for a, b, c in products]
            div_answers = [str(b) for a, b, c in products]
            self.prompts = add_prompts + sub_prompts + mul_prompts + div_prompts
            self.answers = add_answers + sub_answers + mul_answers + div_answers
        else:
            raise ValueError(f"Invalid arithmetic type: {arithmetic_type}")

        print("Prompts and answers generated.")

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        try:
            return {
                "input": self.prompts[idx],
                "target": self.answers[idx]
            }
        except IndexError as e:
             raise IndexError(f"Dataset index {idx} out of range for size {len(self.prompts)}") from e




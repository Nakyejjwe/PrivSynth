"""Data module: loading files and describing their privacy schema."""

from privsynth.data.dataset import Dataset
from privsynth.data.loaders import AdultDataLoader, CSVLoader, DataLoader

__all__ = ["AdultDataLoader", "CSVLoader", "DataLoader", "Dataset"]

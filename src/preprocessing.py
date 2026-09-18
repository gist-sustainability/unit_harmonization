"""Feature engineering and data splitting for unit normalization."""

import re

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder


def extract_quantity_keyword(text):
	"""Extract the quantity marker used by the unit normalization model."""
	text = text.lower()
	if re.search(
		r"\b(tsd\.?|thousand|1,?000|10.?3t?|kilo|kt|tt|million kg|tausend|ktons|kiloton(ne)?s?)\b|\bkt(?=[A-Za-z])",
		text,
	):
		return "kt"
	if re.search(
		r"\b(million(en)?|mio\.?|1,?000,?000|mega|megaton(ne)?s?n?|mm|mn|mmt(on)?(ne)?s?)\b",
		text,
	):
		return "Mt"
	if re.search(r"\b(kg|kilograms?)\b|\bkg(?=[A-Za-z])", text):
		return "kg"
	if re.search(r"\b(us[\s-]?ton(ne)?s?|ust|short tons?)\b", text):
		return "USt"
	return ""


def clean_unit(text):
	"""Remove numeric and connector tokens from a raw unit string."""
	text = re.sub(r"[0-9\.,]+", "", text)
	text = re.sub(r"[^\w]+", " ", text)
	text = re.sub(r"\bof\b", "", text)
	return text.strip().lower()


def create_features_and_labels(
	unit_dict_new,
	extract_quantity_keyword,
	clean_unit,
	unit_col="unit",
	target_col="normalized_unit",
):
	"""Create model features and encoded target labels."""
	raw_units = unit_dict_new[unit_col]
	labels = unit_dict_new[target_col]
	cleaned_text = raw_units.str.lower().str.replace(
		r"\d+[,\.]?\d*", "NUMBER", regex=True
	)
	quantity = raw_units.apply(extract_quantity_keyword)
	unit_base = raw_units.apply(clean_unit)
	features = pd.DataFrame(
		{
			"cleaned_text": cleaned_text,
			"quantity": quantity,
			"unit_base": unit_base,
			"length": raw_units.str.len(),
		}
	)
	features["text"] = (
		features["cleaned_text"]
		+ " "
		+ features["quantity"]
		+ " "
		+ features["unit_base"]
	)
	features["orig_index"] = features.index

	label_encoder = LabelEncoder()
	encoded_labels = label_encoder.fit_transform(labels)
	return features[["text", "length", "orig_index"]], encoded_labels, label_encoder


def stratified_split_by_count(
	unit_dict_new, X, y_encoded, max_count=2, include_artificial=False, random_state=42
):
	"""Split each target class while prioritizing rare units for testing."""
	labels = pd.Series(y_encoded, index=X["orig_index"])
	rng = np.random.RandomState(random_state)
	train_indices = []
	test_indices = []

	for _, group in unit_dict_new.groupby("normalized_unit"):
		group_indices = group.index.values
		candidate_counts = list(range(1, max_count + 1))
		if include_artificial:
			candidate_counts.append(1.5)
		candidates = group[group["count"].isin(candidate_counts)].sort_values("count")
		candidate_indices = candidates.index.values
		test_limit = int(np.floor(0.2 * len(group_indices)))

		if len(candidate_indices) > test_limit:
			selected_indices = []
			for count in sorted(candidates["count"].unique()):
				current = candidates[candidates["count"] == count].index.tolist()
				if len(selected_indices) + len(current) < test_limit:
					selected_indices.extend(current)
				else:
					remaining = test_limit - len(selected_indices)
					selected_indices.extend(
						rng.choice(current, size=remaining, replace=False)
					)
					break
			test_sample = selected_indices
		else:
			test_sample = candidate_indices

		test_indices.extend(test_sample)
		train_indices.extend(np.setdiff1d(group_indices, test_sample))

	train_indices = rng.permutation(train_indices)
	test_indices = rng.permutation(test_indices)
	X_train = X.loc[train_indices].reset_index(drop=True)
	y_train = labels.loc[train_indices].values
	X_test = X.loc[test_indices].reset_index(drop=True)
	y_test = labels.loc[test_indices].values
	return X_train, y_train, X_test, y_test

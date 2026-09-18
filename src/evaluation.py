"""Evaluation metrics and prediction reports for unit normalization."""

import numpy as np
import pandas as pd
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import (
	accuracy_score,
	f1_score,
	precision_recall_fscore_support,
	precision_score,
	recall_score,
)


def evaluate_pipeline(
	pipeline,
	X_test,
	y_test,
	le,
	unit_dict_new,
	extract_quantity_keyword,
	clean_unit,
	new_units=None,
):
	"""Evaluate predictions and return metrics, errors, and new-unit predictions."""
	y_pred = pipeline.predict(X_test)
	accuracy = accuracy_score(y_test, y_pred)
	f1_weighted = f1_score(y_test, y_pred, average="weighted")
	f1_macro = f1_score(y_test, y_pred, average="macro")
	f1_micro = f1_score(y_test, y_pred, average="micro")
	print("=== MAIN METRICS ===")
	print(f"Accuracy:          {accuracy:.2%}")
	print(f"F1-score weighted: {f1_weighted:.2%}")
	print(f"F1-score macro:    {f1_macro:.2%}\n")
	print(f"F1-score micro:    {f1_micro:.2%}\n")

	precision = precision_score(y_test, y_pred, average="weighted")
	recall = recall_score(y_test, y_pred, average="weighted")
	print("=== ADDITIONAL METRICS (weighted) ===")
	print(f"Precision:         {precision:.2%}")
	print(f"Recall:            {recall:.2%}")
	print(f"F1-score weighted: {f1_weighted:.2%}\n")

	true_labels = le.inverse_transform(y_test)
	predicted_labels = le.inverse_transform(y_pred)
	try:
		probabilities = pipeline.predict_proba(X_test)
	except Exception:
		probabilities = None
	if probabilities is not None:
		max_probability = probabilities.max(axis=1)
		classes = sorted(set(y_test))
		class_labels = le.inverse_transform(classes)
		label_to_column = {label: index for index, label in enumerate(class_labels)}
		true_class_probability = np.array(
			[probabilities[i, label_to_column[label]] for i, label in enumerate(true_labels)]
		)
	else:
		max_probability = np.full(len(X_test), np.nan)
		true_class_probability = np.full(len(X_test), np.nan)

	results_df = pd.DataFrame(
		{
			"true_label": true_labels,
			"predicted_label": predicted_labels,
			"orig_index": X_test["orig_index"].values,
			"unit": unit_dict_new.loc[X_test["orig_index"], "unit"].values,
			"pred_max_proba": max_probability,
			"true_class_proba": true_class_probability,
		}
	)
	results_df["is_correct"] = results_df["true_label"] == results_df["predicted_label"]
	error_stats = (
		results_df.assign(error=~results_df["is_correct"])
		.groupby("true_label")
		.agg(mean_error=("error", "mean"), count=("error", "size"))
	)
	precision_by_class, recall_by_class, f1_by_class, support_by_class = (
		precision_recall_fscore_support(y_test, y_pred, average=None)
	)
	classes = sorted(set(y_test))
	per_class = pd.DataFrame(
		{
			"true_label": le.inverse_transform(classes),
			"precision": precision_by_class,
			"recall": recall_by_class,
			"f1": f1_by_class,
			"support": support_by_class,
		}
	).set_index("true_label")
	paper_class_order = [
		"t CO2e", "kt CO2e", "Mt CO2e", "kg CO2e",
		"USt CO2e", "10 kt CO2e", "MM Mt CO2e", "Other",
	]
	combined_stats = error_stats.join(per_class, how="left")
	combined_stats = combined_stats.reindex(
		[label for label in paper_class_order if label in combined_stats.index]
	)
	print("=== ERROR STATISTICS AND PER-CLASS METRICS ===")
	print(combined_stats.round(4))

	confusion_df = pd.crosstab(
		results_df["true_label"], results_df["predicted_label"],
		rownames=["True Label"], colnames=["Predicted Label"],
	)

	desired_order = [
		"kg CO2e", "USt CO2e", "t CO2e", "kt CO2e",
		"10 kt CO2e", "Mt CO2e", "MM Mt CO2e", "Other",
	]
	ordered_confusion = confusion_df.reindex(
		index=desired_order, columns=desired_order, fill_value=0
	).astype(float)
	row_sums = ordered_confusion.sum(axis=1)
	normalized_confusion = ordered_confusion.div(
		row_sums.replace(0, np.nan), axis=0
	).fillna(0.0) * 100
	color_map = LinearSegmentedColormap.from_list(
		"unit_blue", ["#ffffff", "#156082"]
	)
	plt.figure(figsize=(10, 8))
	ax = sns.heatmap(
		normalized_confusion,
		annot=True,
		fmt=".1f",
		cmap=color_map,
		norm=mcolors.PowerNorm(gamma=0.5, vmin=0.0, vmax=100.0),
		linewidths=0.5,
		linecolor="grey",
	)
	plt.title("Row-normalized confusion matrix of emission unit categories")
	plt.ylabel("True unit category")
	plt.xlabel("Predicted unit category")
	colorbar = ax.collections[-1].colorbar
	colorbar.set_label("Share of predictions per true unit category [%]", rotation=90)
	colorbar.set_ticks(np.arange(0, 101, 20))
	colorbar.set_ticklabels([str(value) for value in range(0, 101, 20)])
	plt.tight_layout()
	plt.show()

	print("\nMost frequent confusions:\n", confusion_df)
	errors = results_df[~results_df["is_correct"]].copy()
	if not errors.empty:
		errors["unit"] = unit_dict_new.loc[errors["orig_index"], "unit"].values

	if new_units is None:
		new_units = ["Tonnen Co2 / verkauftem Produkt", "million CO2e", "kwh"]
	new_units_series = pd.Series(new_units)
	new_text = new_units_series.str.lower().str.replace(
		r"\d+[,\.]?\d*", "NUMBER", regex=True
	)
	new_features = pd.DataFrame(
		{
			"text": new_text
			+ " "
			+ new_units_series.apply(extract_quantity_keyword)
			+ " "
			+ new_units_series.apply(clean_unit),
			"length": new_units_series.str.len(),
		}
	)
	predictions = le.inverse_transform(pipeline.predict(new_features))
	print("Predictions for new units:", predictions)
	return {
		"accuracy": accuracy,
		"precision": precision,
		"recall": recall,
		"f1_weighted": f1_weighted,
		"f1_macro": f1_macro,
		"results_df": results_df,
		"error_stats": combined_stats,
		"confusion_df": confusion_df,
		"abweichungen": errors,
		"new_units": new_units,
		"new_predictions": predictions,
	}

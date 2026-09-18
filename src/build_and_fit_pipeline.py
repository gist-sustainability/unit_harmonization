"""Model construction, training, validation, and hyperparameter search."""

import numpy as np
import lightgbm as lgb
import xgboost as xgb
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (
	GridSearchCV,
	RandomizedSearchCV,
	StratifiedKFold,
	cross_validate,
	train_test_split,
)
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from pathlib import Path

from evaluation import evaluate_pipeline
from preprocessing import clean_unit, create_features_and_labels, extract_quantity_keyword


def build_and_train_pipeline(
	X_train, y_train, ml_model, voting_models=None, mode="build_and_train"
):
	"""Build a configured pipeline and optionally fit it."""
	preprocessor = ColumnTransformer(
		transformers=[
			(
				"text",
				TfidfVectorizer(
					analyzer="char_wb",
					max_features=1000,
					ngram_range=(1, 3),
					min_df=3,
					max_df=0.95,
					sublinear_tf=True,
				),
				"text",
			),
			("length", StandardScaler(), ["length"]),
		]
	)
	classifiers = {
		"xgb": xgb.XGBClassifier(
			colsample_bytree=0.8, subsample=1.0, n_estimators=100, max_depth=4,
			learning_rate=0.2, gamma=None, reg_alpha=None, reg_lambda=1.5,
			random_state=42, eval_metric="mlogloss",
		),
		"lightgbm": lgb.LGBMClassifier(
			n_estimators=600, max_depth=-1, learning_rate=0.1, bagging_freq=0,
			feature_fraction=1.0, num_leaves=63, reg_lambda=0.0, random_state=42,
			objective="multiclass", metric="multi_logloss", verbosity=-1,
		),
		"logistic_regression": LogisticRegression(C=100, max_iter=1000, solver="newton-cg"),
		"random_forest": RandomForestClassifier(
			n_estimators=500, max_depth=20, bootstrap=False, class_weight=None,
			max_features="sqrt", min_samples_leaf=1, min_samples_split=10,
			random_state=42, n_jobs=-1,
		),
		"svm": SVC(
			kernel="linear", C=10, class_weight=None, gamma="scale",
			probability=True, random_state=42,
		),
		"mlp": MLPClassifier(
			hidden_layer_sizes=(200, 100), max_iter=500, activation="tanh",
			solver="lbfgs", alpha=0.001, learning_rate_init=0.001,
			random_state=42, early_stopping=True,
		),
	}
	registry = {
		name: (short_name, classifier)
		for name, short_name, classifier in [
			("xgb", "xgb", classifiers["xgb"]),
			("lightgbm", "lgb", classifiers["lightgbm"]),
			("logistic_regression", "logistic", classifiers["logistic_regression"]),
			("random_forest", "rf", classifiers["random_forest"]),
			("svm", "svm", classifiers["svm"]),
			("mlp", "mlp", classifiers["mlp"]),
		]
	}
	if ml_model == "voting_classifier":
		voting_models = voting_models or ["svm", "mlp", "logistic_regression"]
		unknown = [name for name in voting_models if name not in registry]
		if unknown:
			raise ValueError(
				f"Unknown model in voting_models: '{unknown[0]}'. "
				f"Allowed models are: {list(registry)}"
			)
		classifier = VotingClassifier(
			estimators=[registry[name] for name in voting_models], voting="hard"
		)
	elif ml_model in classifiers:
		classifier = classifiers[ml_model]
	else:
		raise ValueError(
			"ml_model must be one of: xgb, lightgbm, logistic_regression, "
			"random_forest, svm, mlp, voting_classifier"
		)

	pipeline = Pipeline([("preprocessor", preprocessor), ("clf", classifier)])
	if mode == "build_and_train":
		pipeline.fit(X_train, y_train)
	return pipeline


def full_pipeline(
	unit_dict_new=None,
	extract_quantity_keyword=extract_quantity_keyword,
	clean_unit=clean_unit,
	split_by_count=False,
	ml_model="logistic_regression",
	voting_models=None,
	max_count=2,
	include_artificial=True,
	new_units=None,
	random_state=42,
):
	"""Run feature engineering, splitting, training, and evaluation."""
	if unit_dict_new is None:
		data_path = Path(__file__).resolve().parents[1] / "data" / "unit_normalization_dict.csv"
		unit_dict_new = pd.read_csv(data_path)
	X, y_encoded, label_encoder = create_features_and_labels(
		unit_dict_new, extract_quantity_keyword, clean_unit
	)
	if split_by_count:
		from preprocessing import stratified_split_by_count
		X_train, y_train, X_test, y_test = stratified_split_by_count(
			unit_dict_new, X, y_encoded, max_count, include_artificial, random_state
		)
	else:
		X_train, X_test, y_train, y_test = train_test_split(
			X, y_encoded, test_size=0.2, stratify=y_encoded, random_state=random_state
		)
	pipeline = build_and_train_pipeline(X_train, y_train, ml_model, voting_models)
	evaluation = evaluate_pipeline(
		pipeline, X_test, y_test, label_encoder, unit_dict_new,
		extract_quantity_keyword, clean_unit, new_units,
	)
	return {
		"pipeline": pipeline,
		"label_encoder": label_encoder,
		"split": {"X_train": X_train, "y_train": y_train, "X_test": X_test, "y_test": y_test},
		"evaluation": evaluation,
	}


def crossvalidate_full_pipeline(
	unit_dict_new=None, ml_model="voting_classifier", voting_models=None,
	cv_folds=10, random_state=42,
):
	"""Evaluate a complete pipeline with stratified cross-validation."""
	if unit_dict_new is None:
		data_path = Path(__file__).resolve().parents[1] / "data" / "unit_normalization_dict.csv"
		unit_dict_new = pd.read_csv(data_path)
	X, y_encoded, _ = create_features_and_labels(
		unit_dict_new, extract_quantity_keyword, clean_unit
	)
	pipeline = build_and_train_pipeline(X, y_encoded, ml_model, voting_models, mode="build_only")
	cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
	scoring = {
		"f1_weighted": "f1_weighted", "f1_macro": "f1_macro",
		"f1_micro": "f1_micro", "accuracy": "accuracy",
	}
	cv_results = cross_validate(
		pipeline, X, y_encoded, cv=cv, scoring=scoring, n_jobs=-1,
		return_train_score=False,
	)
	print(
		"Mean CV scores (+/- std):\n"
		f"  Accuracy:      {np.mean(cv_results['test_accuracy']):.2%} +/- {np.std(cv_results['test_accuracy']):.2%}\n"
		f"  F1 (weighted): {np.mean(cv_results['test_f1_weighted']):.2%} +/- {np.std(cv_results['test_f1_weighted']):.2%}\n"
		f"  F1 (macro):    {np.mean(cv_results['test_f1_macro']):.2%} +/- {np.std(cv_results['test_f1_macro']):.2%}\n"
		f"  F1 (micro):    {np.mean(cv_results['test_f1_micro']):.2%} +/- {np.std(cv_results['test_f1_micro']):.2%}"
	)
	return cv_results


def search_pipeline(
	param_space, unit_dict_new=None, ml_model="voting_classifier", voting_models=None,
	mode="grid", scoring=None, refit="f1", n_iter=100, cv=5, n_jobs=-1,
	random_state=42, verbose=1,
):
	"""Run a grid or randomized hyperparameter search and evaluate its model."""
	if unit_dict_new is None:
		data_path = Path(__file__).resolve().parents[1] / "data" / "unit_normalization_dict.csv"
		unit_dict_new = pd.read_csv(data_path)
	scoring = scoring or {
		"accuracy": "accuracy", "f1": "f1_weighted", "logloss": "neg_log_loss"
	}
	X, y_encoded, label_encoder = create_features_and_labels(
		unit_dict_new, extract_quantity_keyword, clean_unit
	)
	X_train, X_test, y_train, y_test = train_test_split(
		X, y_encoded, test_size=0.2, stratify=y_encoded, random_state=random_state
	)
	pipeline = build_and_train_pipeline(X_train, y_train, ml_model, voting_models, mode="build_only")
	search_class = GridSearchCV if mode == "grid" else RandomizedSearchCV
	search_kwargs = {
		"cv": cv, "scoring": scoring, "refit": refit, "n_jobs": n_jobs, "verbose": verbose
	}
	if mode == "grid":
		search = search_class(pipeline, param_grid=param_space, **search_kwargs)
	elif mode == "random":
		search = search_class(
			pipeline, param_distributions=param_space, n_iter=n_iter,
			random_state=random_state, **search_kwargs,
		)
	else:
		raise ValueError("mode must be either 'grid' or 'random'.")
	search.fit(X_train, y_train)
	print(f"Best parameters ({mode.title()}Search):", search.best_params_)
	for metric in scoring if isinstance(scoring, dict) else [scoring]:
		value = search.cv_results_[f"mean_test_{metric}"][search.best_index_]
		print(f"Best {metric}: {value:.4f}")
	pipeline.set_params(**search.best_params_)
	pipeline.fit(X_train, y_train)
	stats = evaluate_pipeline(
		pipeline, X_test, y_test, label_encoder, unit_dict_new,
		extract_quantity_keyword, clean_unit,
	)
	return search, stats

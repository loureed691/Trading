"""ML-based forecasting with cross-validation and leakage prevention."""

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class ForecastResult:
    """Forecast result with confidence intervals."""

    prediction: float
    confidence: float
    lower_bound: float
    upper_bound: float
    direction: str  # "up", "down", "neutral"
    cv_score: float
    is_reliable: bool


class MLForecaster:
    """Simple ML-based forecaster with robust validation.
    
    Uses basic statistical methods that are more robust than complex ML
    while still providing useful forecasts with proper validation.
    """

    def __init__(self, config: dict[str, Any] | None = None):
        config = config or {}
        self.lookback = config.get("lookback", 100)
        self.forecast_horizon = config.get("forecast_horizon", 5)
        self.min_cv_score = config.get("min_cv_score", 0.51)
        self.n_cv_folds = config.get("n_cv_folds", 5)
        self.min_training_samples = config.get("min_training_samples", 200)
        self.regularization = config.get("regularization", 0.01)
        
        # Feature cache
        self._feature_cache: dict[str, Any] = {}
        self._cv_scores: dict[str, float] = {}

    def _create_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create features from OHLCV data with leakage prevention.
        
        Only uses past data to create features (no future information).
        """
        df = data.copy()
        
        # Price-based features (all lagged)
        df["return_1"] = df["close"].pct_change(1).shift(1)
        df["return_5"] = df["close"].pct_change(5).shift(1)
        df["return_10"] = df["close"].pct_change(10).shift(1)
        
        # Volatility features
        df["volatility_10"] = df["close"].pct_change().rolling(10).std().shift(1)
        df["volatility_20"] = df["close"].pct_change().rolling(20).std().shift(1)
        
        # Momentum features
        df["momentum_5"] = (df["close"].shift(1) / df["close"].shift(6) - 1)
        df["momentum_10"] = (df["close"].shift(1) / df["close"].shift(11) - 1)
        
        # Mean reversion features
        df["zscore_20"] = (
            (df["close"].shift(1) - df["close"].rolling(20).mean().shift(1))
            / df["close"].rolling(20).std().shift(1)
        )
        
        # Volume features
        df["volume_ratio"] = (
            df["volume"].shift(1) / df["volume"].rolling(10).mean().shift(1)
        )
        
        # Range features
        df["range_pct"] = ((df["high"] - df["low"]) / df["close"]).shift(1)
        
        # Target (future return) - only for training
        df["target"] = df["close"].pct_change(self.forecast_horizon).shift(-self.forecast_horizon)
        
        return df

    def _purged_cv_split(
        self, n_samples: int, n_folds: int, gap: int = 5
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        """Create purged time-series cross-validation splits.
        
        Includes gap between train and test to prevent leakage.
        """
        fold_size = n_samples // n_folds
        splits = []
        
        for i in range(n_folds):
            test_start = i * fold_size
            test_end = (i + 1) * fold_size if i < n_folds - 1 else n_samples
            
            # Training uses all data before test minus gap
            train_end = max(0, test_start - gap)
            if train_end < self.min_training_samples // 2:
                continue
            
            train_idx = np.arange(0, train_end)
            test_idx = np.arange(test_start, test_end)
            
            splits.append((train_idx, test_idx))
        
        return splits

    def _simple_regression_predict(
        self, X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray
    ) -> np.ndarray:
        """Simple linear regression prediction without external libraries.
        
        Uses normal equation: beta = (X'X)^-1 X'y
        """
        # Add intercept
        X_train_bias = np.column_stack([np.ones(len(X_train)), X_train])
        X_test_bias = np.column_stack([np.ones(len(X_test)), X_test])
        
        try:
            # Solve normal equation with regularization (ridge regression)
            XtX = X_train_bias.T @ X_train_bias
            XtX_reg = XtX + self.regularization * np.eye(XtX.shape[0])
            Xty = X_train_bias.T @ y_train
            
            beta = np.linalg.solve(XtX_reg, Xty)
            predictions = X_test_bias @ beta
            
            return predictions
        except Exception:
            # Fallback to mean prediction
            return np.full(len(X_test), np.mean(y_train))

    def cross_validate(self, data: pd.DataFrame) -> float:
        """Run cross-validation to assess model reliability.
        
        Returns: CV score (directional accuracy)
        """
        if len(data) < self.min_training_samples:
            return 0.0
        
        df = self._create_features(data)
        
        # Feature columns (exclude target and any future-looking)
        feature_cols = [
            "return_1", "return_5", "return_10",
            "volatility_10", "volatility_20",
            "momentum_5", "momentum_10",
            "zscore_20", "volume_ratio", "range_pct",
        ]
        
        # Drop rows with NaN
        df_clean = df.dropna(subset=feature_cols + ["target"])
        
        if len(df_clean) < self.min_training_samples:
            return 0.0
        
        X = df_clean[feature_cols].values
        y = df_clean["target"].values
        
        # Purged CV
        splits = self._purged_cv_split(len(X), self.n_cv_folds)
        
        if len(splits) < 2:
            return 0.0
        
        accuracies = []
        for train_idx, test_idx in splits:
            X_train, y_train = X[train_idx], y[train_idx]
            X_test, y_test = X[test_idx], y[test_idx]
            
            if len(X_train) < 50 or len(X_test) < 10:
                continue
            
            predictions = self._simple_regression_predict(X_train, y_train, X_test)
            
            # Directional accuracy
            correct = np.sum(np.sign(predictions) == np.sign(y_test))
            accuracy = correct / len(y_test)
            accuracies.append(accuracy)
        
        if not accuracies:
            return 0.0
        
        cv_score = np.mean(accuracies)
        symbol = data.attrs.get("symbol", "unknown")
        self._cv_scores[symbol] = cv_score
        
        logger.info(f"CV Score for {symbol}: {cv_score:.4f}")
        return cv_score

    def forecast(self, data: pd.DataFrame) -> ForecastResult:
        """Generate forecast with confidence bounds.
        
        Only produces reliable forecasts if CV score is above threshold.
        """
        symbol = data.attrs.get("symbol", "unknown")
        
        # Check if we have a valid CV score
        if symbol not in self._cv_scores:
            cv_score = self.cross_validate(data)
        else:
            cv_score = self._cv_scores[symbol]
        
        is_reliable = cv_score >= self.min_cv_score
        
        if not is_reliable:
            # Return neutral forecast
            return ForecastResult(
                prediction=0.0,
                confidence=0.0,
                lower_bound=0.0,
                upper_bound=0.0,
                direction="neutral",
                cv_score=cv_score,
                is_reliable=False,
            )
        
        # Create features
        df = self._create_features(data)
        feature_cols = [
            "return_1", "return_5", "return_10",
            "volatility_10", "volatility_20",
            "momentum_5", "momentum_10",
            "zscore_20", "volume_ratio", "range_pct",
        ]
        
        df_clean = df.dropna(subset=feature_cols)
        
        if len(df_clean) < self.min_training_samples:
            return ForecastResult(
                prediction=0.0,
                confidence=0.0,
                lower_bound=0.0,
                upper_bound=0.0,
                direction="neutral",
                cv_score=cv_score,
                is_reliable=False,
            )
        
        # Train on all available data (excluding last row which is prediction target)
        train_df = df_clean[:-1].dropna(subset=["target"])
        
        if len(train_df) < self.min_training_samples // 2:
            return ForecastResult(
                prediction=0.0,
                confidence=0.0,
                lower_bound=0.0,
                upper_bound=0.0,
                direction="neutral",
                cv_score=cv_score,
                is_reliable=False,
            )
        
        X_train = train_df[feature_cols].values
        y_train = train_df["target"].values
        X_pred = df_clean[feature_cols].iloc[-1:].values
        
        prediction = self._simple_regression_predict(X_train, y_train, X_pred)[0]
        
        # Estimate prediction uncertainty
        residuals = y_train - self._simple_regression_predict(X_train, y_train, X_train)
        std_residual = np.std(residuals)
        
        # Confidence based on CV score and prediction magnitude
        confidence = min(1.0, (cv_score - 0.5) * 2)
        
        # Direction
        if abs(prediction) < std_residual * 0.5:
            direction = "neutral"
        elif prediction > 0:
            direction = "up"
        else:
            direction = "down"
        
        return ForecastResult(
            prediction=prediction,
            confidence=confidence,
            lower_bound=prediction - 2 * std_residual,
            upper_bound=prediction + 2 * std_residual,
            direction=direction,
            cv_score=cv_score,
            is_reliable=is_reliable,
        )

    def check_data_leakage(self, data: pd.DataFrame) -> list[str]:
        """Check for common data leakage issues.
        
        Returns list of warnings about potential leakage.
        """
        warnings = []
        
        # Check for future-looking values in features
        df = self._create_features(data)
        
        # Check if any features have values that shouldn't exist yet
        feature_cols = [
            "return_1", "return_5", "return_10",
            "volatility_10", "volatility_20",
        ]
        
        for col in feature_cols:
            if col not in df.columns:
                continue
            
            # First N values should be NaN (where N depends on lookback)
            expected_nan_count = int(col.split("_")[-1]) if "_" in col else 1
            actual_nan_count = df[col].isna().sum()
            
            if actual_nan_count < expected_nan_count:
                warnings.append(
                    f"Feature '{col}' may have leakage: "
                    f"expected at least {expected_nan_count} NaN values, "
                    f"found {actual_nan_count}"
                )
        
        return warnings

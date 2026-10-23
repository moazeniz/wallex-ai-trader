"""
Feature Preprocessor

Handles feature normalization, scaling, and preprocessing:
- Normalization (min-max, z-score)
- Missing value handling
- Feature selection
- Dimensionality reduction
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.impute import SimpleImputer


class FeaturePreprocessor:
    """
    Feature preprocessing pipeline.

    Handles normalization, missing values, and feature selection.
    """

    def __init__(
        self,
        scaling_method: str = "robust",  # "standard", "minmax", "robust"
        handle_missing: str = "median",  # "mean", "median", "zero", "drop"
    ):
        """
        Initialize the preprocessor.

        Args:
            scaling_method: Method for scaling features
            handle_missing: How to handle missing values
        """
        self.scaling_method = scaling_method
        self.handle_missing = handle_missing

        # Scalers
        self._scalers: Dict[str, Any] = {}
        self._imputers: Dict[str, SimpleImputer] = {}

        # Feature info
        self._feature_columns: List[str] = []
        self._fitted = False

    def fit(
        self,
        df: pd.DataFrame,
        feature_columns: Optional[List[str]] = None,
    ) -> "FeaturePreprocessor":
        """
        Fit the preprocessor to the data.

        Args:
            df: DataFrame with features
            feature_columns: Columns to process (uses all numeric if None)

        Returns:
            Self for chaining
        """
        # Select feature columns
        if feature_columns is None:
            feature_columns = df.select_dtypes(include=[np.number]).columns.tolist()

        self._feature_columns = feature_columns

        # Create scaler
        if self.scaling_method == "standard":
            scaler = StandardScaler()
        elif self.scaling_method == "minmax":
            scaler = MinMaxScaler()
        elif self.scaling_method == "robust":
            scaler = RobustScaler()
        else:
            raise ValueError(f"Unknown scaling method: {self.scaling_method}")

        # Handle missing values first
        df_clean = self._handle_missing_fit(df[feature_columns])

        # Fit scaler
        scaler.fit(df_clean)
        self._scalers["main"] = scaler

        self._fitted = True
        return self

    def transform(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Transform data using fitted preprocessor.

        Args:
            df: DataFrame to transform

        Returns:
            Transformed DataFrame
        """
        if not self._fitted:
            raise RuntimeError("Preprocessor not fitted. Call fit() first.")

        # Get feature columns
        df_features = df[self._feature_columns].copy()

        # Handle missing values
        df_clean = self._handle_missing_transform(df_features)

        # Scale
        scaled = self._scalers["main"].transform(df_clean)

        # Create output DataFrame
        result = df.copy()
        result[self._feature_columns] = scaled

        return result

    def fit_transform(
        self,
        df: pd.DataFrame,
        feature_columns: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """Fit and transform in one step."""
        self.fit(df, feature_columns)
        return self.transform(df)

    def _handle_missing_fit(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Fit missing value handling and transform."""
        if self.handle_missing == "drop":
            return df.dropna()

        # Create imputer
        if self.handle_missing == "mean":
            strategy = "mean"
        elif self.handle_missing == "median":
            strategy = "median"
        elif self.handle_missing == "zero":
            strategy = "constant"
        else:
            strategy = "median"

        imputer = SimpleImputer(
            strategy=strategy,
            fill_value=0 if strategy == "constant" else None
        )

        # Fit and transform
        result = pd.DataFrame(
            imputer.fit_transform(df),
            columns=df.columns,
            index=df.index
        )

        self._imputers["main"] = imputer
        return result

    def _handle_missing_transform(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Transform missing values using fitted imputer."""
        if self.handle_missing == "drop":
            return df.dropna()

        imputer = self._imputers.get("main")
        if imputer is None:
            return df.fillna(0)

        return pd.DataFrame(
            imputer.transform(df),
            columns=df.columns,
            index=df.index
        )

    # =========================================================================
    # Feature Selection
    # =========================================================================

    def remove_low_variance(
        self,
        df: pd.DataFrame,
        threshold: float = 0.01,
    ) -> Tuple[pd.DataFrame, List[str]]:
        """
        Remove features with low variance.

        Args:
            df: DataFrame with features
            threshold: Variance threshold

        Returns:
            Tuple of (filtered DataFrame, removed columns)
        """
        variances = df.var()
        low_var_cols = variances[variances < threshold].index.tolist()

        return df.drop(columns=low_var_cols), low_var_cols

    def remove_correlated(
        self,
        df: pd.DataFrame,
        threshold: float = 0.95,
    ) -> Tuple[pd.DataFrame, List[str]]:
        """
        Remove highly correlated features.

        Args:
            df: DataFrame with features
            threshold: Correlation threshold

        Returns:
            Tuple of (filtered DataFrame, removed columns)
        """
        corr_matrix = df.corr().abs()

        # Upper triangle
        upper = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )

        # Find columns to drop
        to_drop = [col for col in upper.columns if any(upper[col] > threshold)]

        return df.drop(columns=to_drop), to_drop

    def get_feature_importance(
        self,
        df: pd.DataFrame,
        target_col: str,
    ) -> pd.Series:
        """
        Calculate feature importance based on correlation with target.

        Args:
            df: DataFrame with features and target
            target_col: Target column name

        Returns:
            Series with feature importances
        """
        correlations = df.corr()[target_col].abs()
        correlations = correlations.drop(target_col)
        return correlations.sort_values(ascending=False)

    # =========================================================================
    # Utility Methods
    # =========================================================================

    def get_feature_stats(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Get statistics for all features."""
        stats = df.describe().T
        stats["missing"] = df.isnull().sum()
        stats["missing_pct"] = (df.isnull().sum() / len(df)) * 100
        stats["unique"] = df.nunique()
        return stats

    def clip_outliers(
        self,
        df: pd.DataFrame,
        columns: Optional[List[str]] = None,
        lower_percentile: float = 1,
        upper_percentile: float = 99,
    ) -> pd.DataFrame:
        """
        Clip outliers to percentile bounds.

        Args:
            df: DataFrame to process
            columns: Columns to clip (None for all numeric)
            lower_percentile: Lower bound percentile
            upper_percentile: Upper bound percentile

        Returns:
            DataFrame with clipped values
        """
        result = df.copy()

        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()

        for col in columns:
            lower = df[col].quantile(lower_percentile / 100)
            upper = df[col].quantile(upper_percentile / 100)
            result[col] = df[col].clip(lower, upper)

        return result

    def add_lag_features(
        self,
        df: pd.DataFrame,
        columns: List[str],
        lags: List[int] = [1, 2, 3, 5, 10],
    ) -> pd.DataFrame:
        """
        Add lagged versions of features.

        Args:
            df: DataFrame
            columns: Columns to create lags for
            lags: List of lag periods

        Returns:
            DataFrame with lag features added
        """
        result = df.copy()

        for col in columns:
            for lag in lags:
                result[f"{col}_lag_{lag}"] = df[col].shift(lag)

        return result

    def add_rolling_features(
        self,
        df: pd.DataFrame,
        columns: List[str],
        windows: List[int] = [5, 10, 20],
    ) -> pd.DataFrame:
        """
        Add rolling statistics.

        Args:
            df: DataFrame
            columns: Columns to create rolling stats for
            windows: List of window sizes

        Returns:
            DataFrame with rolling features added
        """
        result = df.copy()

        for col in columns:
            for window in windows:
                result[f"{col}_roll_mean_{window}"] = df[col].rolling(window).mean()
                result[f"{col}_roll_std_{window}"] = df[col].rolling(window).std()
                result[f"{col}_roll_min_{window}"] = df[col].rolling(window).min()
                result[f"{col}_roll_max_{window}"] = df[col].rolling(window).max()

        return result


def create_preprocessor(
    scaling_method: str = "robust",
    handle_missing: str = "median",
) -> FeaturePreprocessor:
    """Create a FeaturePreprocessor instance."""
    return FeaturePreprocessor(
        scaling_method=scaling_method,
        handle_missing=handle_missing,
    )


if __name__ == "__main__":
    print("Testing FeaturePreprocessor...")

    # Create sample data
    np.random.seed(42)
    n = 100

    df = pd.DataFrame({
        "feature1": np.random.randn(n) * 10 + 50,
        "feature2": np.random.randn(n) * 5 + 20,
        "feature3": np.random.randn(n) * 2 + 10,
        "constant": np.ones(n),
        "target": np.random.randint(0, 2, n),
    })

    # Add some missing values
    df.loc[10:15, "feature1"] = np.nan

    print("Original stats:")
    print(df.describe())

    # Preprocess
    pp = FeaturePreprocessor(scaling_method="robust", handle_missing="median")
    df_processed = pp.fit_transform(df, feature_columns=["feature1", "feature2", "feature3"])

    print("\nProcessed stats:")
    print(df_processed[["feature1", "feature2", "feature3"]].describe())

    # Feature selection
    df_filtered, removed = pp.remove_low_variance(df, threshold=0.01)
    print(f"\nRemoved low variance: {removed}")

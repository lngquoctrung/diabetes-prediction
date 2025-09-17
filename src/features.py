import sys
from pathlib import Path
# Add the project path into the python path
root_dir = str(Path(__file__).parent.parent.parent.absolute())
if not root_dir in sys.path:
    sys.path.insert(0, root_dir)

import logging
import os

from sklearn.preprocessing import LabelEncoder
from src.config import LOG_FORMAT
from src.utils import make_dirs

class DiabetesFeatureEngineering:
    """Class for comprehensive feature engineering on diabetes dataset"""

    def __init__(self, log_file: str | None = None, log_format: str | None = LOG_FORMAT):
        """
        Constructor of DiabetesFeatureEngineering class

        Parameters
        ----------
            log_file: str or None
                The log filename
            log_format: str
                The log format
        """
        # Log configuration
        self.logger = logging.getLogger(name=__name__)
        self.logger.setLevel(level=logging.DEBUG)
        log_formatter = logging.Formatter(fmt=log_format)
        # Clear existing handlers to avoid duplicate logs
        self.logger.handlers.clear()
        # Handler log to file
        if log_file:
            # Create a log directory
            make_dirs(path=os.path.dirname(log_file))
            log_file_handler = logging.FileHandler(filename=log_file)
            log_file_handler.setLevel(level=logging.INFO)
            log_file_handler.setFormatter(fmt=log_formatter)
            self.logger.addHandler(hdlr=log_file_handler)
        # Handler log to console
        log_console_handler = logging.StreamHandler()
        log_console_handler.setLevel(level=logging.INFO)
        log_console_handler.setFormatter(fmt=log_formatter)
        self.logger.addHandler(hdlr=log_console_handler)

        self.logger.info(msg="DiabetesFeatureEngineering initialized successfully")

    def remove_null_values(self, df):
        """
        Remove null values from the dataframe

        Parameters
        ----------
            df: pandas.DataFrame
                Input dataframe

        Returns
        -------
            pandas.DataFrame
                Cleaned dataframe without null values
        """
        self.logger.info(msg="Starting null values removal process...")

        initial_rows = len(df)
        null_counts = df.isnull().sum()
        total_nulls = null_counts.sum()

        if total_nulls > 0:
            self.logger.info(msg=f"Found {total_nulls} null values across {(null_counts > 0).sum()} columns")
            for col in null_counts[null_counts > 0].index:
                self.logger.info(msg=f"  - {col}: {null_counts[col]} null values")
        else:
            self.logger.info(msg="No null values found in the dataset")

        df_clean = df.copy()
        df_clean.dropna(inplace=True)

        final_rows = len(df_clean)
        removed_rows = initial_rows - final_rows

        self.logger.info(
            msg=f"Null values removal completed. Removed {removed_rows} rows ({removed_rows / initial_rows * 100:.2f}%)"
        )
        self.logger.info(msg=f"Dataset shape: {initial_rows} -> {final_rows} rows")

        return df_clean

    def remove_duplicates(self, df):
        """
        Remove duplicate rows from the dataframe

        Parameters
        ----------
            df: pandas.DataFrame
                Input dataframe

        Returns
        -------
            pandas.DataFrame
                The Dataframe without duplicate rows
        """
        self.logger.info(msg="Starting duplicate removal process...")

        initial_rows = len(df)
        duplicate_count = df.duplicated().sum()

        if duplicate_count > 0:
            self.logger.info(msg=f"Found {duplicate_count} duplicate rows")

            # Check duplicates by label
            for label in df['Diabetes'].unique():
                label_duplicates = df[df['Diabetes'] == label].duplicated().sum()
                self.logger.info(f"  - Label {label}: {label_duplicates} duplicate rows")
        else:
            self.logger.info("No duplicate rows found")

        df_clean = df.copy()
        df_clean.drop_duplicates(inplace=True)

        final_rows = len(df_clean)
        removed_rows = initial_rows - final_rows

        self.logger.info(
            msg=f"Duplicate removal completed. Removed {removed_rows} rows ({removed_rows / initial_rows * 100:.2f}%)"
        )
        self.logger.info(msg=f"Dataset shape: {initial_rows} -> {final_rows} rows")

        return df_clean

    def handle_outliers(self, df, columns_to_clean=None):
        """
        Handle outliers for specified columns within each Diabetes group using the IQR method

        Parameters
        ----------
            df: pandas.DataFrame
                Input dataframe
            columns_to_clean: list
                List of column names to handle outliers for

        Returns
        -------
            pandas.DataFrame
                The Dataframe with outliers handled
        """
        if columns_to_clean is None:
            columns_to_clean = ['BMI', 'MentHlth', 'PhysHlth']
        self.logger.info(msg="Starting outlier handling process...")
        self.logger.info(msg=f"Columns to process: {columns_to_clean}")

        df_clean = df.copy()
        outliers_handled = {}

        for col in columns_to_clean:
            if col not in df.columns:
                self.logger.warning(msg=f"Column '{col}' not found in dataframe. Skipping...")
                continue

            column_outliers = 0
            self.logger.info(msg=f"Processing outliers for column: {col}")

            # Calculate Q1, Q3 and IQR for each Diabetes group
            for diabetes_type in df['Diabetes'].unique():
                mask = df['Diabetes'] == diabetes_type

                Q1 = df[mask][col].quantile(0.25)
                Q3 = df[mask][col].quantile(0.75)
                IQR = Q3 - Q1

                # Define outlier bounds
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR

                # Count outliers before handling
                outliers_lower = (df[mask][col] < lower_bound).sum()
                outliers_upper = (df[mask][col] > upper_bound).sum()
                group_outliers = outliers_lower + outliers_upper

                if group_outliers > 0:
                    self.logger.info(
                        msg=f"  - Diabetes group {diabetes_type}: {group_outliers} outliers (Lower: {outliers_lower}, Upper: {outliers_upper})"
                    )
                    column_outliers += group_outliers

                # Handle outliers by capping to the bounds
                df_clean.loc[mask & (df[col] < lower_bound), col] = lower_bound
                df_clean.loc[mask & (df[col] > upper_bound), col] = upper_bound

            outliers_handled[col] = column_outliers

        total_outliers = sum(outliers_handled.values())
        self.logger.info(msg=f"Outlier handling completed. Total outliers handled: {total_outliers}")

        return df_clean

    def create_health_score(self, df):
        """
        Create health score feature based on GenHlth, MentHlth, and PhysHlth

        Parameters
        ----------
            df: pandas.DataFrame
                Input dataframe

        Returns
        -------
            pandas.DataFrame
                Dataframe with HlthScore feature added
        """
        self.logger.info(msg="Creating health score feature...")

        df_new = df.copy()

        required_cols = ['GenHlth', 'MentHlth', 'PhysHlth']
        missing_cols = [col for col in required_cols if col not in df.columns]

        if missing_cols:
            self.logger.warning(msg=f"Missing columns for health score: {missing_cols}. Skipping health score creation.")
            return df_new

        # HlthScore = GenHlth/5 + MentHlth/30 + PhysHlth/30
        df_new['HlthScore'] = (df_new['GenHlth'] / 5.0) + \
                              (df_new['MentHlth'] / 30.0) + \
                              (df_new['PhysHlth'] / 30.0)

        self.logger.info(
            msg=f"Health score created successfully. Range: [{df_new['HlthScore'].min():.3f}, {df_new['HlthScore'].max():.3f}]"
        )

        return df_new

    def create_risk_score(self, df):
        """
        Create a risk score feature based on HighBP, HighChol, HeartDiseaseorAttack, and Stroke

        Parameters
        ----------
            df: pandas.DataFrame
                Input dataframe

        Returns
        -------
            pandas.DataFrame
                The Dataframe with RiskScore feature added
        """
        self.logger.info(msg="Creating risk score feature...")

        df_new = df.copy()

        required_cols = ['HighBP', 'HighChol', 'HeartDiseaseorAttack', 'Stroke']
        missing_cols = [col for col in required_cols if col not in df.columns]

        if missing_cols:
            self.logger.warning(msg=f"Missing columns for risk score: {missing_cols}. Using available columns.")
            available_cols = [col for col in required_cols if col in df.columns]
        else:
            available_cols = required_cols

        if available_cols:
            df_new['RiskScore'] = df_new[available_cols].sum(axis=1)
            self.logger.info(
                msg=f"Risk score created successfully using {len(available_cols)} features. Range: [{df_new['RiskScore'].min()}, {df_new['RiskScore'].max()}]"
            )
        else:
            self.logger.warning(msg="No columns available for risk score creation.")

        return df_new

    def create_lifestyle_score(self, df):
        """
        Create a lifestyle score feature based on PhysActivity and other lifestyle factors

        Parameters
        ----------
            df: pandas.DataFrame
                Input dataframe

        Returns
        -------
            pandas.DataFrame
                The Dataframe with LifestyleScore feature added
        """
        self.logger.info(msg="Creating lifestyle score feature...")

        df_new = df.copy()

        # Base lifestyle factors (excluding Fruits and Veggies as requested)
        base_factors = ['PhysActivity']
        available_factors = [col for col in base_factors if col in df.columns]

        # Additional positive lifestyle factors you might have
        additional_positive = []  # Can add more columns here based on your dataset

        # Negative lifestyle factors
        negative_factors = ['HvyAlcoholConsump', 'Smoker']
        available_negative = [col for col in negative_factors if col in df.columns]

        if available_factors:
            # Calculate positive lifestyle score
            positive_score = df_new[available_factors + additional_positive].sum(axis=1) if (
                        available_factors + additional_positive) else 0

            # Calculate negative lifestyle score
            negative_score = df_new[available_negative].sum(axis=1) if available_negative else 0

            # Final lifestyle score
            df_new['LifestyleScore'] = positive_score - negative_score

            self.logger.info(
                msg=f"Lifestyle score created successfully using {len(available_factors + additional_positive)} positive and {len(available_negative)} negative factors"
            )
            self.logger.info(msg=f"Range: [{df_new['LifestyleScore'].min()}, {df_new['LifestyleScore'].max()}]")
        else:
            self.logger.warning(msg="No columns available for lifestyle score creation.")

        return df_new

    def create_cardio_risk(self, df):
        """
        Create cardiovascular risk feature based on HighBP, HighChol, and BMI obesity

        Parameters
        ----------
            df: pandas.DataFrame
                Input dataframe

        Returns
        -------
            pandas.DataFrame
                Dataframe with CardioRisk feature added
        """
        self.logger.info(msg="Creating cardiovascular risk feature...")

        df_new = df.copy()

        required_cols = ['HighBP', 'HighChol', 'BMI']
        missing_cols = [col for col in required_cols if col not in df.columns]

        if missing_cols:
            self.logger.warning(msg=f"Missing columns for cardio risk: {missing_cols}. Using available columns.")

        cardio_risk = 0
        factors_used = []

        if 'HighBP' in df.columns:
            cardio_risk += df_new['HighBP']
            factors_used.append('HighBP')

        if 'HighChol' in df.columns:
            cardio_risk += df_new['HighChol']
            factors_used.append('HighChol')

        if 'BMI' in df.columns:
            obesity_indicator = (df_new['BMI'] > 30).astype(int)
            cardio_risk += obesity_indicator
            factors_used.append('BMI_Obesity')

        if factors_used:
            df_new['CardioRisk'] = cardio_risk
            self.logger.info(msg=f"Cardiovascular risk created successfully using: {factors_used}")
            self.logger.info(msg=f"Range: [{df_new['CardioRisk'].min()}, {df_new['CardioRisk'].max()}]")
        else:
            self.logger.warning(msg="No columns available for cardiovascular risk creation.")

        return df_new

    def create_bmi_category(self, df):
        """
        Create a BMI category feature based on WHO standards

        Parameters
        ----------
            df: pandas.DataFrame
                Input dataframe

        Returns
        -------
            pandas.DataFrame
                The Dataframe with BMICategory feature added
        """
        self.logger.info(msg="Creating BMI category feature...")

        df_new = df.copy()

        if 'BMI' not in df.columns:
            self.logger.warning(msg="BMI column not found. Skipping BMI category creation.")
            return df_new

        def bmi_category(bmi):
            if bmi < 18.5:
                return 'Underweight'
            elif bmi <= 24.9:
                return 'Normal weight'
            elif bmi <= 29.9:
                return 'Pre-obesity'
            elif bmi <= 34.9:
                return 'Obesity class I'
            elif bmi <= 39.9:
                return 'Obesity class II'
            else:
                return 'Obesity class III'

        df_new['BMICategory'] = df_new['BMI'].apply(bmi_category)

        # Log category distribution
        category_counts = df_new['BMICategory'].value_counts()
        self.logger.info("BMI category distribution:")
        for category, count in category_counts.items():
            self.logger.info(msg=f"  - {category}: {count} ({count / len(df_new) * 100:.2f}%)")

        return df_new

    def create_age_group(self, df):
        """
        Create an age group feature

        Parameters
        ----------
            df: pandas.DataFrame
                Input dataframe

        Returns
        -------
            pandas.DataFrame
                The Dataframe with AgeGroup feature added
        """
        self.logger.info(msg="Creating age group feature...")

        df_new = df.copy()

        if 'Age' not in df.columns:
            self.logger.warning(msg="Age column not found. Skipping age group creation.")
            return df_new

        def age_group(age):
            if age <= 3:
                return 'Young'
            elif age <= 8:
                return 'Middle'
            else:
                return 'Senior'

        df_new['AgeGroup'] = df_new['Age'].apply(age_group)

        # Log age group distribution
        age_counts = df_new['AgeGroup'].value_counts()
        self.logger.info(msg="Age group distribution:")
        for group, count in age_counts.items():
            self.logger.info(msg=f"  - {group}: {count} ({count / len(df_new) * 100:.2f}%)")

        return df_new

    def encode_categorical_features(self, df):
        """
        Encode categorical features using Label Encoder

        Parameters
        ----------
            df: pandas.DataFrame
                Input dataframe

        Returns
        -------
            tuple: (pandas.DataFrame, dict)
                Dataframe with encoded categorical features and dictionary of encoders
        """
        self.logger.info(msg="Starting categorical feature encoding...")

        df_new = df.copy()
        encoders = {}

        # Find categorical columns
        categorical_cols = df_new.select_dtypes(include=['object']).columns.tolist()

        if not categorical_cols:
            self.logger.info(msg="No categorical columns found to encode.")
            return df_new, encoders

        self.logger.info(msg=f"Found {len(categorical_cols)} categorical columns to encode: {categorical_cols}")

        for col in categorical_cols:
            self.logger.info(msg=f"Encoding column: {col}")
            encoder = LabelEncoder()
            df_new[col] = encoder.fit_transform(y=df_new[col])
            encoders[col] = encoder

            # Log encoding mapping
            encoding_dict = dict(zip(encoder.classes_, range(len(encoder.classes_))))
            self.logger.info(msg=f"  - Encoding mapping: {encoding_dict}")

        self.logger.info(msg="Categorical feature encoding completed")

        return df_new, encoders

    def feature_selection(self, df, target_col='Diabetes', corr_threshold=0.1):
        """
        Select features based on correlation with the target variable

        Parameters
        ----------
            df: pandas.DataFrame
                Input dataframe
            target_col: str
                Name of target column
            corr_threshold: float
                The correlation threshold for feature selection

        Returns
        -------
            list
                The list of selected feature names (including target column)
        """
        self.logger.info(msg="Starting feature selection process...")
        self.logger.info(msg=f"Target column: {target_col}")
        self.logger.info(msg=f"Correlation threshold: {corr_threshold}")

        if target_col not in df.columns:
            self.logger.error(msg=f"Target column '{target_col}' not found in dataframe")
            return df.columns.tolist()

        # Calculate correlations with a target
        correlations = df.corr()[target_col]
        correlations = correlations.drop(target_col)

        # Get absolute correlations
        abs_correlations = abs(correlations)

        # Select features above a threshold
        selected_features = abs_correlations[abs_correlations > corr_threshold].index.tolist()
        selected_features.append(target_col)  # Add the target column back

        self.logger.info(msg=f"Selected {len(selected_features) - 1} features (+ target column)")
        self.logger.info(msg="Selected features and their correlations:")

        for feature in selected_features[:-1]:  # Exclude target column from logging
            corr_value = correlations[feature]
            self.logger.info(msg=f"  - {feature}: {corr_value:.4f}")

        return selected_features

    def process_all(self, df, target_col='Diabetes', corr_threshold=0.1,
                    columns_to_clean=None):
        """
        Execute all feature engineering steps

        Parameters
        ----------
            df: pandas.DataFrame
                Input dataframe
            target_col: str
                Name of target column
            corr_threshold: float
                The correlation threshold for feature selection
            columns_to_clean: list
                List of columns to handle outliers for

        Returns
        -------
            tuple: (pandas.DataFrame, dict, list)
                Processed dataframe, encoder dictionary, and selected features list
        """
        if columns_to_clean is None:
            columns_to_clean = ['BMI', 'MentHlth', 'PhysHlth']
        self.logger.info(msg="=" * 60)
        self.logger.info(msg="STARTING COMPLETE FEATURE ENGINEERING PIPELINE")
        self.logger.info(msg="=" * 60)

        initial_shape = df.shape
        self.logger.info(msg=f"Initial dataset shape: {initial_shape}")

        # Step 1: Remove null values
        df = self.remove_null_values(df=df)
        self.logger.info(msg=f"After null removal: {df.shape}")

        # Step 2: Remove duplicates
        df = self.remove_duplicates(df=df)
        self.logger.info(msg=f"After duplicate removal: {df.shape}")

        # Step 3: Handle outliers
        df = self.handle_outliers(df=df, columns_to_clean=columns_to_clean)

        # Step 4: Create new features
        df = self.create_health_score(df=df)
        df = self.create_risk_score(df=df)
        df = self.create_lifestyle_score(df=df)
        df = self.create_cardio_risk(df=df)
        df = self.create_bmi_category(df=df)
        df = self.create_age_group(df=df)

        self.logger.info(msg=f"After feature creation: {df.shape}")

        # Step 5: Encode categorical features
        df, encoders = self.encode_categorical_features(df=df)
        self.logger.info(msg=f"After categorical encoding: {df.shape}")

        # Step 6: Feature selection
        selected_features = self.feature_selection(
            df=df,
            target_col=target_col,
            corr_threshold=corr_threshold
        )
        df_final = df[selected_features]

        final_shape = df_final.shape
        self.logger.info(msg=f"Final dataset shape: {final_shape}")

        # Summary
        self.logger.info(msg="=" * 60)
        self.logger.info(msg="FEATURE ENGINEERING PIPELINE COMPLETED")
        self.logger.info(msg="=" * 60)
        self.logger.info(msg=f"Dataset transformation: {initial_shape} -> {final_shape}")
        self.logger.info(
            msg=f"Rows change: {initial_shape[0]} -> {final_shape[0]} ({(final_shape[0] / initial_shape[0] - 1) * 100:+.2f}%)"
        )
        self.logger.info(
            msg=f"Features change: {initial_shape[1]} -> {final_shape[1]} ({(final_shape[1] / initial_shape[1] - 1) * 100:+.2f}%)"
        )

        return df_final, encoders, selected_features
    
    def process_for_inference(self, df, selected_features, encoders=None, columns_to_clean=None):
        """
        Process data for inference using pre-selected features
        
        Parameters
        ----------
        df : pandas.DataFrame
            Input dataframe
        selected_features : list
            Pre-selected feature names from training
        encoders : dict
            Pre-fitted encoders from training
        columns_to_clean : list
            List of columns to handle outliers for
        
        Returns
        -------
        pandas.DataFrame
            Processed dataframe with selected features only
        """
        if columns_to_clean is None:
            columns_to_clean = ['BMI', 'MentHlth', 'PhysHlth']
        
        self.logger.info(msg="="*60)
        self.logger.info(msg="STARTING INFERENCE FEATURE ENGINEERING PIPELINE")
        self.logger.info(msg="="*60)
        
        initial_shape = df.shape
        self.logger.info(msg=f"Initial dataset shape: {initial_shape}")
        
        # Steps 1-4: Same preprocessing steps
        df = self.remove_null_values(df=df)
        df = self.remove_duplicates(df=df)
        df = self.handle_outliers(df=df, columns_to_clean=columns_to_clean)
        
        # Create new features
        df = self.create_health_score(df=df)
        df = self.create_risk_score(df=df)
        df = self.create_lifestyle_score(df=df)
        df = self.create_cardio_risk(df=df)
        df = self.create_bmi_category(df=df)
        df = self.create_age_group(df=df)
        
        # Encode categorical features using existing encoders if provided
        if encoders:
            for col, encoder in encoders.items():
                if col in df.columns:
                    df[col] = encoder.transform(df[col])
        else:
            df, _ = self.encode_categorical_features(df=df)
        
        # Apply pre-selected features
        available_features = [f for f in selected_features if f in df.columns]
        missing_features = set(selected_features) - set(available_features)
        
        if missing_features:
            self.logger.warning(msg=f"Missing features in inference data: {missing_features}")
        
        df_final = df[available_features]
        
        self.logger.info(msg=f"Final dataset shape: {df_final.shape}")
        self.logger.info(msg="="*60)
        self.logger.info(msg="INFERENCE FEATURE ENGINEERING PIPELINE COMPLETED")
        self.logger.info(msg="="*60)
        
        return df_final


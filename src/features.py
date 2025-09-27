import sys
from pathlib import Path
# Add the project path into the python path
root_dir = str(Path(__file__).parent.parent.parent.absolute())
if not root_dir in sys.path:
    sys.path.insert(0, root_dir)

import os
from sklearn.preprocessing import LabelEncoder
from src.config import LOG_FORMAT
from src.utils import get_configured_logger

class DiabetesFeatureEngineering:
    """Enhanced class for comprehensive feature engineering on diabetes dataset with 33+ features"""

    def __init__(self, logger_name: str | None = __name__, log_file: str | None = None, log_format: str | None = LOG_FORMAT):
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
        self.logger = get_configured_logger(
            name=logger_name, 
            log_file=log_file, 
            log_format=log_format
        )

        self.logger.info(msg="Enhanced DiabetesFeatureEngineering initialized successfully")

    def remove_null_values(self, df):
        """Remove null values from the dataframe"""
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
        """Remove duplicate rows from the dataframe"""
        self.logger.info(msg="Starting duplicate removal process...")

        initial_rows = len(df)
        duplicate_count = df.duplicated().sum()

        if duplicate_count > 0:
            self.logger.info(msg=f"Found {duplicate_count} duplicate rows")

            # Check duplicates by label
            if 'Diabetes' in df.columns:
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
        Enhanced outlier handling for expanded feature set
        """
        if columns_to_clean is None:
            # Updated to include new continuous features
            columns_to_clean = ['BMI', 'MentHlth', 'PhysHlth', 'AlcoholDays']
        
        self.logger.info(msg="Starting enhanced outlier handling process...")
        self.logger.info(msg=f"Columns to process: {columns_to_clean}")

        df_clean = df.copy()
        outliers_handled = {}

        for col in columns_to_clean:
            if col not in df.columns:
                self.logger.warning(msg=f"Column '{col}' not found in dataframe. Skipping...")
                continue

            column_outliers = 0
            self.logger.info(msg=f"Processing outliers for column: {col}")

            # Calculate Q1, Q3 and IQR for each Diabetes group if Diabetes exists
            if 'Diabetes' in df.columns:
                for diabetes_type in df['Diabetes'].unique():
                    mask = df['Diabetes'] == diabetes_type
                    if mask.sum() == 0:
                        continue

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
            else:
                # Global outlier handling if no Diabetes column
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outliers_lower = (df[col] < lower_bound).sum()
                outliers_upper = (df[col] > upper_bound).sum()
                column_outliers = outliers_lower + outliers_upper
                
                df_clean.loc[df[col] < lower_bound, col] = lower_bound
                df_clean.loc[df[col] > upper_bound, col] = upper_bound

            outliers_handled[col] = column_outliers

        total_outliers = sum(outliers_handled.values())
        self.logger.info(msg=f"Enhanced outlier handling completed. Total outliers handled: {total_outliers}")

        return df_clean

    def create_health_score(self, df):
        """
        Enhanced health score including difficulty walking
        """
        self.logger.info(msg="Creating enhanced health score feature...")

        df_new = df.copy()

        # Updated to include DiffWalk
        required_cols = ['GenHlth', 'MentHlth', 'PhysHlth', 'DiffWalk']
        available_cols = [col for col in required_cols if col in df.columns]

        if not available_cols:
            self.logger.warning(msg="No health-related columns available for health score creation.")
            return df_new

        self.logger.info(msg=f"Using columns for health score: {available_cols}")

        # Enhanced health score calculation
        health_score = 0
        
        if 'GenHlth' in available_cols:
            health_score += df_new['GenHlth'] / 5.0
            
        if 'MentHlth' in available_cols:
            health_score += df_new['MentHlth'] / 30.0
            
        if 'PhysHlth' in available_cols:
            health_score += df_new['PhysHlth'] / 30.0
            
        if 'DiffWalk' in available_cols:
            health_score += df_new['DiffWalk'] * 0.5  # Walking difficulty penalty

        df_new['HlthScore'] = health_score

        self.logger.info(
            msg=f"Enhanced health score created using {len(available_cols)} features. Range: [{df_new['HlthScore'].min():.3f}, {df_new['HlthScore'].max():.3f}]"
        )

        return df_new

    def create_risk_score(self, df):
        """
        Enhanced comprehensive health risk score
        """
        self.logger.info(msg="Creating enhanced comprehensive risk score feature...")

        df_new = df.copy()

        # Expanded risk factors
        risk_factors = [
            'HighBP', 'HighChol', 'HeartDiseaseorAttack', 'Stroke',
            'DiagnosedHeartAttack', 'CoronaryHeartDisease', 'COPD', 'KidneyDisease'
        ]
        
        available_factors = [col for col in risk_factors if col in df.columns]

        if not available_factors:
            self.logger.warning(msg="No risk factors available for risk score creation.")
            return df_new

        self.logger.info(msg=f"Using {len(available_factors)} risk factors: {available_factors}")

        df_new['RiskScore'] = df_new[available_factors].sum(axis=1)

        self.logger.info(
            msg=f"Enhanced risk score created successfully. Range: [{df_new['RiskScore'].min()}, {df_new['RiskScore'].max()}]"
        )

        return df_new

    def create_lifestyle_score(self, df):
        """
        Enhanced lifestyle score with granular alcohol consumption
        """
        self.logger.info(msg="Creating enhanced lifestyle score feature...")

        df_new = df.copy()

        # Positive lifestyle factors
        positive_factors = ['PhysActivity']
        available_positive = [col for col in positive_factors if col in df.columns]

        # Negative lifestyle factors - prioritize AlcoholDays over HvyAlcoholConsump
        negative_factors = []
        if 'AlcoholDays' in df.columns:
            negative_factors.append('AlcoholDays')
        elif 'HvyAlcoholConsump' in df.columns:
            negative_factors.append('HvyAlcoholConsump')
            
        if 'Smoker' in df.columns:
            negative_factors.append('Smoker')

        available_negative = [col for col in negative_factors if col in df.columns]

        if not available_positive and not available_negative:
            self.logger.warning(msg="No lifestyle factors available for score creation.")
            return df_new

        # Calculate scores
        positive_score = df_new[available_positive].sum(axis=1) if available_positive else 0

        negative_score = 0
        if 'AlcoholDays' in available_negative:
            # Normalize alcohol days (0-30 range) to 0-1 scale
            negative_score += df_new['AlcoholDays'] / 30.0
        if 'HvyAlcoholConsump' in available_negative:
            negative_score += df_new['HvyAlcoholConsump']
        if 'Smoker' in available_negative:
            negative_score += df_new['Smoker']

        # Final lifestyle score
        df_new['LifestyleScore'] = positive_score - negative_score

        self.logger.info(
            msg=f"Enhanced lifestyle score created using {len(available_positive)} positive and {len(available_negative)} negative factors"
        )
        self.logger.info(msg=f"Range: [{df_new['LifestyleScore'].min():.3f}, {df_new['LifestyleScore'].max():.3f}]")

        return df_new

    def create_cardio_risk(self, df):
        """
        Enhanced cardiovascular risk with diagnosed conditions
        """
        self.logger.info(msg="Creating enhanced cardiovascular risk feature...")

        df_new = df.copy()

        cardio_factors = ['HighBP', 'HighChol', 'DiagnosedHeartAttack', 'CoronaryHeartDisease']
        available_factors = [col for col in cardio_factors if col in df.columns]

        cardio_risk = 0
        factors_used = []

        for factor in available_factors:
            cardio_risk += df_new[factor]
            factors_used.append(factor)

        # Add BMI obesity indicator
        if 'BMI' in df.columns:
            obesity_indicator = (df_new['BMI'] > 30).astype(int)
            cardio_risk += obesity_indicator
            factors_used.append('BMI_Obesity')

        if factors_used:
            df_new['CardioRisk'] = cardio_risk
            self.logger.info(msg=f"Enhanced cardiovascular risk created using: {factors_used}")
            self.logger.info(msg=f"Range: [{df_new['CardioRisk'].min()}, {df_new['CardioRisk'].max()}]")
        else:
            self.logger.warning(msg="No factors available for cardiovascular risk creation.")

        return df_new

    def create_healthcare_access_score(self, df):
        """
        NEW: Create healthcare access and utilization score
        """
        self.logger.info(msg="Creating healthcare access score feature...")

        df_new = df.copy()

        # Healthcare access factors
        access_factors = {
            'positive': ['AnyHealthcare', 'HasPersonalDoctor'],
            'negative': ['NoDocbcCost', 'CannotAffordDoctor'],
            'utilization': ['LastCheckup']  # Higher values = more recent = better
        }

        available_positive = [col for col in access_factors['positive'] if col in df.columns]
        available_negative = [col for col in access_factors['negative'] if col in df.columns]
        available_utilization = [col for col in access_factors['utilization'] if col in df.columns]

        if not (available_positive or available_negative or available_utilization):
            self.logger.warning(msg="No healthcare access factors available.")
            return df_new

        # Calculate healthcare access score
        access_score = 0

        # Positive factors (having insurance, having doctor)
        if available_positive:
            access_score += df_new[available_positive].sum(axis=1)

        # Negative factors (cost barriers)
        if available_negative:
            access_score -= df_new[available_negative].sum(axis=1)

        # Utilization factors (recent checkups are positive)
        if 'LastCheckup' in available_utilization:
            # Invert LastCheckup so recent visits (lower values) contribute positively
            max_checkup = df_new['LastCheckup'].max()
            access_score += (max_checkup - df_new['LastCheckup']) / max_checkup

        df_new['HealthcareAccessScore'] = access_score

        factors_used = available_positive + available_negative + available_utilization
        self.logger.info(msg=f"Healthcare access score created using: {factors_used}")
        self.logger.info(msg=f"Range: [{df_new['HealthcareAccessScore'].min():.3f}, {df_new['HealthcareAccessScore'].max():.3f}]")

        return df_new

    def create_socioeconomic_score(self, df):
        """
        NEW: Create socioeconomic status score
        """
        self.logger.info(msg="Creating socioeconomic score feature...")

        df_new = df.copy()

        socio_factors = ['Education', 'Income', 'EmploymentStatus', 'MaritalStatus']
        available_factors = [col for col in socio_factors if col in df.columns]

        if not available_factors:
            self.logger.warning(msg="No socioeconomic factors available.")
            return df_new

        # Normalize and combine socioeconomic factors
        socio_score = 0
        factors_used = []

        if 'Education' in available_factors:
            socio_score += df_new['Education'] / df_new['Education'].max()
            factors_used.append('Education')

        if 'Income' in available_factors:
            socio_score += df_new['Income'] / df_new['Income'].max()
            factors_used.append('Income')

        if 'EmploymentStatus' in available_factors:
            socio_score += df_new['EmploymentStatus'] / df_new['EmploymentStatus'].max()
            factors_used.append('EmploymentStatus')

        if 'MaritalStatus' in available_factors:
            # Reverse the scale: married = highest socioeconomic score
            socio_score += (7 - df_new['MaritalStatus']) / 6.0
            factors_used.append('MaritalStatus')


        df_new['SocioeconomicScore'] = socio_score

        self.logger.info(msg=f"Socioeconomic score created using: {factors_used}")
        self.logger.info(msg=f"Range: [{df_new['SocioeconomicScore'].min():.3f}, {df_new['SocioeconomicScore'].max():.3f}]")

        return df_new

    def create_cholesterol_management_score(self, df):
        """
        NEW: Create cholesterol management score
        """
        self.logger.info(msg="Creating cholesterol management score feature...")

        df_new = df.copy()

        chol_factors = ['CholCheck', 'HighChol', 'CholesterolMeds']
        available_factors = [col for col in chol_factors if col in df.columns]

        if not available_factors:
            self.logger.warning(msg="No cholesterol management factors available.")
            return df_new

        # Calculate cholesterol management score
        management_score = 0

        if 'CholCheck' in available_factors:
            management_score += df_new['CholCheck']  # Regular checking is positive

        if 'CholesterolMeds' in available_factors:
            management_score += df_new['CholesterolMeds']  # Taking meds when needed is positive

        # High cholesterol is a risk factor, but managed high cholesterol is better than unmanaged
        if 'HighChol' in available_factors and 'CholesterolMeds' in available_factors:
            # Penalty for high cholesterol, but reduced penalty if taking meds
            high_chol_penalty = df_new['HighChol'] * (1 - 0.5 * df_new['CholesterolMeds'])
            management_score -= high_chol_penalty
        elif 'HighChol' in available_factors:
            management_score -= df_new['HighChol']

        df_new['CholesterolManagementScore'] = management_score

        self.logger.info(msg=f"Cholesterol management score created using: {available_factors}")
        self.logger.info(msg=f"Range: [{df_new['CholesterolManagementScore'].min():.3f}, {df_new['CholesterolManagementScore'].max():.3f}]")

        return df_new

    def create_mental_health_score(self, df):
        """
        NEW: Create comprehensive mental health score
        """
        self.logger.info(msg="Creating comprehensive mental health score feature...")

        df_new = df.copy()

        mental_factors = ['Depression', 'CognitiveIssues', 'MentHlth']
        available_factors = [col for col in mental_factors if col in df.columns]

        if not available_factors:
            self.logger.warning(msg="No mental health factors available.")
            return df_new

        # Calculate comprehensive mental health score
        mental_score = 0

        if 'Depression' in available_factors:
            mental_score -= df_new['Depression']  # Depression is negative

        if 'CognitiveIssues' in available_factors:
            mental_score -= df_new['CognitiveIssues']  # Cognitive issues are negative

        if 'MentHlth' in available_factors:
            mental_score -= df_new['MentHlth'] / 30.0  # Normalize days of poor mental health

        df_new['MentalHealthScore'] = mental_score

        self.logger.info(msg=f"Comprehensive mental health score created using: {available_factors}")
        self.logger.info(msg=f"Range: [{df_new['MentalHealthScore'].min():.3f}, {df_new['MentalHealthScore'].max():.3f}]")

        return df_new

    def create_bmi_category(self, df):
        """Create BMI category feature based on WHO standards"""
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
        """Enhanced age group creation"""
        self.logger.info(msg="Creating enhanced age group feature...")

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

        # Enhanced analysis with sex interaction if available
        if 'Sex' in df.columns:
            self.logger.info("Age-Sex distribution analysis:")
            cross_tab = df_new.groupby(['AgeGroup', 'Sex']).size().unstack(fill_value=0)
            for age_group in cross_tab.index:
                for sex in cross_tab.columns:
                    count = cross_tab.loc[age_group, sex]
                    self.logger.info(f"  - {age_group}-Sex{sex}: {count}")

        # Log age group distribution
        age_counts = df_new['AgeGroup'].value_counts()
        self.logger.info(msg="Age group distribution:")
        for group, count in age_counts.items():
            self.logger.info(msg=f"  - {group}: {count} ({count / len(df_new) * 100:.2f}%)")

        return df_new

    def encode_categorical_features(self, df):
        """Enhanced categorical encoding for expanded feature set"""
        self.logger.info(msg="Starting enhanced categorical feature encoding...")

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
            
            try:
                df_new[col] = encoder.fit_transform(y=df_new[col])
                encoders[col] = encoder

                # Log encoding mapping
                encoding_dict = dict(zip(encoder.classes_, range(len(encoder.classes_))))
                self.logger.info(msg=f"  - Encoding mapping: {encoding_dict}")
                
            except Exception as e:
                self.logger.error(f"Error encoding column {col}: {str(e)}")
                continue

        self.logger.info(msg=f"Categorical feature encoding completed for {len(encoders)} columns")

        return df_new, encoders

    def feature_selection(self, df, target_col='Diabetes', corr_threshold=0.05):
        """
        Enhanced feature selection for expanded feature set
        """
        self.logger.info(msg="Starting enhanced feature selection process...")
        self.logger.info(msg=f"Target column: {target_col}")
        self.logger.info(msg=f"Correlation threshold: {corr_threshold}")

        if target_col not in df.columns:
            self.logger.error(msg=f"Target column '{target_col}' not found in dataframe")
            return df.columns.tolist()

        # Calculate correlations with target
        correlations = df.corr()[target_col]
        correlations = correlations.drop(target_col)

        # Get absolute correlations
        abs_correlations = abs(correlations)

        # Select features above threshold
        selected_features = abs_correlations[abs_correlations > corr_threshold].index.tolist()
        selected_features.append(target_col)  # Add target column back

        # Log detailed correlation analysis
        self.logger.info(f"Selected {len(selected_features) - 1} features (+ target column) out of {len(correlations)} available")
        self.logger.info("Top 10 selected features by correlation:")

        # Sort by absolute correlation for top features display
        top_features = abs_correlations.sort_values(ascending=False).head(10)
        for feature, abs_corr in top_features.items():
            if feature in selected_features[:-1]:  # Exclude target
                actual_corr = correlations[feature]
                self.logger.info(f"  - {feature}: {actual_corr:.4f} (|{abs_corr:.4f}|)")

        return selected_features

    def process_all(self, df, target_col='Diabetes', corr_threshold=0.05, columns_to_clean=None):
        """
        Enhanced processing pipeline for expanded feature set
        """
        if columns_to_clean is None:
            columns_to_clean = ['BMI', 'MentHlth', 'PhysHlth', 'AlcoholDays']
            
        self.logger.info(msg="=" * 80)
        self.logger.info(msg="STARTING ENHANCED FEATURE ENGINEERING PIPELINE")
        self.logger.info(msg="=" * 80)

        initial_shape = df.shape
        self.logger.info(msg=f"Initial dataset shape: {initial_shape}")

        # Step 1-3: Basic preprocessing
        df = self.remove_null_values(df=df)
        self.logger.info(msg=f"After null removal: {df.shape}")

        df = self.remove_duplicates(df=df)
        self.logger.info(msg=f"After duplicate removal: {df.shape}")

        df = self.handle_outliers(df=df, columns_to_clean=columns_to_clean)

        # Step 4: Enhanced feature creation
        self.logger.info(msg="Creating enhanced feature set...")
        
        df = self.create_health_score(df=df)
        df = self.create_risk_score(df=df)
        df = self.create_lifestyle_score(df=df)
        df = self.create_cardio_risk(df=df)
        df = self.create_bmi_category(df=df)
        df = self.create_age_group(df=df)
        df = self.create_healthcare_access_score(df=df)
        df = self.create_socioeconomic_score(df=df)
        df = self.create_cholesterol_management_score(df=df)
        df = self.create_mental_health_score(df=df)

        self.logger.info(msg=f"After enhanced feature creation: {df.shape}")

        # Step 5: Encode categorical features
        df, encoders = self.encode_categorical_features(df=df)
        self.logger.info(msg=f"After categorical encoding: {df.shape}")

        # Step 6: Enhanced feature selection
        selected_features = self.feature_selection(
            df=df,
            target_col=target_col,
            corr_threshold=corr_threshold
        )
        df_final = df[selected_features]

        final_shape = df_final.shape
        self.logger.info(msg=f"Final dataset shape: {final_shape}")

        # Enhanced summary
        self.logger.info(msg="=" * 80)
        self.logger.info(msg="ENHANCED FEATURE ENGINEERING PIPELINE COMPLETED")
        self.logger.info(msg="=" * 80)
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
        Enhanced inference processing for expanded feature set
        """
        if columns_to_clean is None:
            columns_to_clean = ['BMI', 'MentHlth', 'PhysHlth', 'AlcoholDays']
        
        self.logger.info(msg="="*80)
        self.logger.info(msg="STARTING ENHANCED INFERENCE FEATURE ENGINEERING PIPELINE")
        self.logger.info(msg="="*80)
        
        initial_shape = df.shape
        self.logger.info(msg=f"Initial dataset shape: {initial_shape}")
        
        # Steps 1-3: Same preprocessing steps
        df = self.remove_null_values(df=df)
        df = self.remove_duplicates(df=df)
        df = self.handle_outliers(df=df, columns_to_clean=columns_to_clean)
        
        # Enhanced feature creation
        df = self.create_health_score(df=df)
        df = self.create_risk_score(df=df)
        df = self.create_lifestyle_score(df=df)
        df = self.create_cardio_risk(df=df)
        df = self.create_bmi_category(df=df)
        df = self.create_age_group(df=df)
        
        # New features
        df = self.create_healthcare_access_score(df=df)
        df = self.create_socioeconomic_score(df=df)
        df = self.create_cholesterol_management_score(df=df)
        df = self.create_mental_health_score(df=df)
        
        # Encode categorical features using existing encoders if provided
        if encoders:
            for col, encoder in encoders.items():
                if col in df.columns:
                    try:
                        df[col] = encoder.transform(df[col])
                    except ValueError as e:
                        self.logger.warning(f"Could not transform column {col}: {str(e)}")
        else:
            df, _ = self.encode_categorical_features(df=df)
        
        # Apply pre-selected features
        available_features = [f for f in selected_features if f in df.columns]
        missing_features = set(selected_features) - set(available_features)
        
        if missing_features:
            self.logger.warning(msg=f"Missing features in inference data: {missing_features}")
        
        df_final = df[available_features]
        
        self.logger.info(msg=f"Final dataset shape: {df_final.shape}")
        self.logger.info(msg="="*80)
        self.logger.info(msg="ENHANCED INFERENCE FEATURE ENGINEERING PIPELINE COMPLETED")
        self.logger.info(msg="="*80)
        
        return df_final

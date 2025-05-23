import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from scipy.stats import entropy

class TransactionFeatureEngineering:
    def __init__(self, parquet_file_path, top_n_categories=20):
        """
        Initialize the feature engineering class with the parquet file path.
        
        Args:
            parquet_file_path (str): Path to the parquet file containing transaction data
            top_n_categories (int): Number of top categories to keep for high-cardinality features
        """
        self.parquet_file_path = parquet_file_path
        self.top_n_categories = top_n_categories
        self.df_transactions = None
        self.df_user_features = None
        self.df_scaled_user_features = None
        self.df_clustering_ready = None
        self.scaler = None
        self.categorical_encoder = None
        self.reference_date_for_recency = None
        self.top_mccs = None
        self.top_transaction_types = None
        
    def load_data(self):
        """Load transaction data from parquet file."""
        try:
            self.df_transactions = pd.read_parquet(self.parquet_file_path)
            print(f"Successfully loaded {len(self.df_transactions)} transactions")
            return True
        except Exception as e:
            print(f"Error loading Parquet file: {e}")
            return False
    
    def clean_and_prepare_data(self):
        """Clean and prepare the transaction data."""
        if self.df_transactions is None or self.df_transactions.empty:
            print("No data to clean. Please load data first.")
            return False
            
        # Convert timestamp to datetime objects
        self.df_transactions['transaction_timestamp'] = pd.to_datetime(
            self.df_transactions['transaction_timestamp']
        )
        
        # Sort by card_id and timestamp
        self.df_transactions = self.df_transactions.sort_values(
            by=['card_id', 'transaction_timestamp']
        )

        # Create a flag for whether a currency conversion originally took place
        self.df_transactions['had_currency_conversion'] = self.df_transactions['original_amount'].notna().astype(int)

        # Fill original_amount where transaction_currency is KZT and original_amount is null.
        self.df_transactions.loc[
            (self.df_transactions['transaction_currency'] == 'KZT') &
            self.df_transactions['original_amount'].isna(),
            'original_amount'
        ] = self.df_transactions['transaction_amount_kzt']
        
        # Fill any remaining NaNs in 'original_amount'.
        self.df_transactions['original_amount'] = self.df_transactions['original_amount'].fillna(-1)
        
        # Handle missing values
        self.df_transactions['merchant_mcc'] = self.df_transactions['merchant_mcc'].astype(str).fillna('UNKNOWN_MCC')
        self.df_transactions['merchant_id'] = self.df_transactions['merchant_id'].astype(str).fillna('UNKNOWN_MERCHANT')
        self.df_transactions['merchant_city'] = self.df_transactions['merchant_city'].fillna('UNKNOWN_CITY')
        self.df_transactions['pos_entry_mode'] = self.df_transactions['pos_entry_mode'].fillna('NOT_APPLICABLE')
        self.df_transactions['wallet_type'] = self.df_transactions['wallet_type'].fillna('REGULAR_CARD')
        
        # Calculate reference date for recency
        overall_last_transaction_date = self.df_transactions['transaction_timestamp'].max()
        self.reference_date_for_recency = overall_last_transaction_date + pd.Timedelta(days=1)
        
        # Identify top categories for frequency-based encoding
        self._identify_top_categories()
        
        print("Data cleaning and preparation completed")
        return True
    
    def _identify_top_categories(self):
        """Identify top N categories for MCC and transaction types."""
        # Top MCCs
        mcc_counts = self.df_transactions['merchant_mcc'].value_counts()
        self.top_mccs = mcc_counts.head(self.top_n_categories).index.tolist()
        
        # Top transaction types
        txn_type_counts = self.df_transactions['transaction_type'].value_counts()
        self.top_transaction_types = txn_type_counts.head(self.top_n_categories).index.tolist()
        
        print(f"Identified top {len(self.top_mccs)} MCCs and {len(self.top_transaction_types)} transaction types")
    
    def _calculate_entropy(self, series):
        """Calculate entropy for a series to measure diversity."""
        if len(series) <= 1:
            return 0
        value_counts = series.value_counts(normalize=True)
        return entropy(value_counts.values)
    
    def _calculate_gini_coefficient(self, series):
        """Calculate Gini coefficient for a series to measure concentration."""
        if len(series) <= 1:
            return 0
        
        values = series.sort_values().values
        n = len(values)
        index = np.arange(1, n + 1)
        return (2 * np.sum(index * values)) / (n * np.sum(values)) - (n + 1) / n
    
    def engineer_features(self):
        """Engineer user-level features from transaction data."""
        if self.df_transactions is None or self.df_transactions.empty:
            print("No transaction data available for feature engineering")
            return False
            
        user_features_list = []
        grouped_by_card = self.df_transactions.groupby('card_id')
        
        for card, group in grouped_by_card:
            user_data = {'card_id': card}
            
            # Basic Aggregations
            user_data['total_transactions'] = group.shape[0]
            user_data['total_spend_kzt'] = group['transaction_amount_kzt'].sum()
            user_data['avg_spend_kzt'] = group['transaction_amount_kzt'].mean()
            user_data['median_spend_kzt'] = group['transaction_amount_kzt'].median()
            
            # Handle standard deviation for single transactions
            if group.shape[0] > 1:
                user_data['std_spend_kzt'] = group['transaction_amount_kzt'].std(ddof=0)
            else:
                user_data['std_spend_kzt'] = 0
                
            user_data['min_spend_kzt'] = group['transaction_amount_kzt'].min()
            user_data['max_spend_kzt'] = group['transaction_amount_kzt'].max()
            
            # Financial Health Features
            if user_data['avg_spend_kzt'] > 0:
                user_data['coefficient_of_variation_kzt'] = user_data['std_spend_kzt'] / user_data['avg_spend_kzt']
            else:
                user_data['coefficient_of_variation_kzt'] = 0
                
            user_data['max_to_total_spend_ratio'] = user_data['max_spend_kzt'] / user_data['total_spend_kzt']
            
            # Time-based Features
            first_transaction_date = group['transaction_timestamp'].min()
            last_transaction_date = group['transaction_timestamp'].max()
            
            user_data['days_since_first_transaction'] = (
                self.reference_date_for_recency - first_transaction_date
            ).days
            user_data['days_since_last_transaction'] = (
                self.reference_date_for_recency - last_transaction_date
            ).days
            
            # Transaction frequency
            total_days_active = user_data['days_since_first_transaction']
            if total_days_active > 0:
                user_data['transactions_per_day'] = user_data['total_transactions'] / total_days_active
            else:
                user_data['transactions_per_day'] = 0
            
            # Enhanced Temporal Features
            user_data['transactions_morning'] = group[
                group['transaction_timestamp'].dt.hour.between(6, 11)
            ].shape[0]
            user_data['transactions_afternoon'] = group[
                group['transaction_timestamp'].dt.hour.between(12, 17)
            ].shape[0]
            user_data['transactions_evening'] = group[
                group['transaction_timestamp'].dt.hour.between(18, 23)
            ].shape[0]
            user_data['transactions_night'] = group[
                group['transaction_timestamp'].dt.hour.between(0, 5)
            ].shape[0]
            
            # Proportional temporal features
            total_txns = user_data['total_transactions']
            user_data['prop_transactions_morning'] = user_data['transactions_morning'] / total_txns
            user_data['prop_transactions_afternoon'] = user_data['transactions_afternoon'] / total_txns
            user_data['prop_transactions_evening'] = user_data['transactions_evening'] / total_txns
            user_data['prop_transactions_night'] = user_data['transactions_night'] / total_txns
            
            user_data['transactions_weekdays'] = group[
                group['transaction_timestamp'].dt.dayofweek < 5
            ].shape[0]
            user_data['transactions_weekends'] = group[
                group['transaction_timestamp'].dt.dayofweek >= 5
            ].shape[0]
            
            user_data['prop_transactions_weekdays'] = user_data['transactions_weekdays'] / total_txns
            user_data['prop_transactions_weekends'] = user_data['transactions_weekends'] / total_txns
            
            # Quarterly features (if data spans multiple quarters)
            group['quarter'] = group['transaction_timestamp'].dt.quarter
            for quarter in [1, 2, 3, 4]:
                quarter_txns = group[group['quarter'] == quarter].shape[0]
                user_data[f'prop_transactions_Q{quarter}'] = quarter_txns / total_txns
            
            # Monthly transaction average
            group['year_month'] = group['transaction_timestamp'].dt.to_period('M')
            unique_months = group['year_month'].nunique()
            if unique_months > 0:
                user_data['avg_monthly_transactions'] = total_txns / unique_months
            else:
                user_data['avg_monthly_transactions'] = 0
            
            # Average time between transactions
            if group.shape[0] > 1:
                time_diffs = group['transaction_timestamp'].diff().dropna()
                user_data['avg_days_between_transactions'] = time_diffs.mean().total_seconds() / (60*60*24)
            else:
                user_data['avg_days_between_transactions'] = 0
                
            # Merchant & Category Features with Diversity Metrics
            user_data['num_unique_merchants'] = group['merchant_id'].nunique()
            user_data['num_unique_mcc'] = group['merchant_mcc'].nunique()
            user_data['most_frequent_mcc'] = (
                group['merchant_mcc'].mode()[0] 
                if not group['merchant_mcc'].mode().empty 
                else 'UNKNOWN_MCC'
            )
            
            # Entropy and Gini coefficient for spending diversity
            user_data['mcc_entropy'] = self._calculate_entropy(group['merchant_mcc'])
            user_data['merchant_entropy'] = self._calculate_entropy(group['merchant_id'])
            user_data['spend_gini_coefficient'] = self._calculate_gini_coefficient(group['transaction_amount_kzt'])
            
            # Top MCC proportions (frequency-based encoding)
            mcc_counts = group['merchant_mcc'].value_counts(normalize=True)
            for mcc in self.top_mccs:
                user_data[f'prop_mcc_{mcc}'] = mcc_counts.get(mcc, 0)
            
            # Other MCC proportion
            other_mcc_prop = mcc_counts[~mcc_counts.index.isin(self.top_mccs)].sum()
            user_data['prop_mcc_other'] = other_mcc_prop
                
            # Top Transaction Type proportions
            transaction_type_counts = group['transaction_type'].value_counts(normalize=True)
            for t_type in self.top_transaction_types:
                user_data[f'prop_type_{t_type}'] = transaction_type_counts.get(t_type, 0)
            
            # Other transaction type proportion
            other_type_prop = transaction_type_counts[~transaction_type_counts.index.isin(self.top_transaction_types)].sum()
            user_data['prop_type_other'] = other_type_prop
            
            user_data['most_frequent_txn_type'] = (
                group['transaction_type'].mode()[0] 
                if not group['transaction_type'].mode().empty 
                else 'UNKNOWN_TYPE'
            )
            
            # Geographic Features
            user_data['num_unique_merchant_cities'] = group['merchant_city'].nunique()
            user_data['most_frequent_merchant_city'] = (
                group['merchant_city'].mode()[0] 
                if not group['merchant_city'].mode().empty 
                else 'UNKNOWN_CITY'
            )
            
            # International Activity
            user_data['num_foreign_currency_txns'] = group[
                group['transaction_currency'] != 'KZT'
            ].shape[0]
            user_data['num_foreign_acquirer_txns'] = group[
                group['acquirer_country_iso'] != 'KAZ'
            ].shape[0]
            user_data['uses_foreign_services'] = 1 if (
                user_data['num_foreign_currency_txns'] > 0 or 
                user_data['num_foreign_acquirer_txns'] > 0
            ) else 0

            # Proportion of transactions with currency conversion
            user_data['prop_currency_conversion_txns'] = group['had_currency_conversion'].mean()
            
            # POS Entry & Wallet Features
            pos_entry_mode_counts = group['pos_entry_mode'].value_counts(normalize=True)
            for mode, prop in pos_entry_mode_counts.items():
                if mode != 'NOT_APPLICABLE':
                    user_data[f'prop_pos_{mode}'] = prop
                    
            wallet_type_counts = group['wallet_type'].value_counts(normalize=True)
            for wallet, prop in wallet_type_counts.items():
                if wallet != 'REGULAR_CARD':
                    user_data[f'prop_wallet_{wallet}'] = prop
            user_data['uses_any_digital_wallet'] = 1 if group[
                group['wallet_type'] != 'REGULAR_CARD'
            ].shape[0] > 0 else 0
            
            user_features_list.append(user_data)
        
        # Create DataFrame
        self.df_user_features = pd.DataFrame(user_features_list)
        self.df_user_features = self.df_user_features.fillna(0)
        self.df_user_features = self.df_user_features.set_index('card_id')
        
        print(f"Feature engineering completed. Shape: {self.df_user_features.shape}")
        return True
    
    def scale_features(self):
        """Scale numerical features for clustering."""
        if self.df_user_features is None or self.df_user_features.empty:
            print("No user features available for scaling")
            return False
            
        # Identify categorical columns to exclude from scaling
        categorical_columns = [
            'most_frequent_mcc', 
            'most_frequent_txn_type', 
            'most_frequent_merchant_city'
        ]
        
        # Get numerical features for scaling
        features_to_scale_df = self.df_user_features.drop(
            columns=categorical_columns, 
            errors='ignore'
        )
        
        if not features_to_scale_df.empty:
            self.scaler = StandardScaler()
            scaled_features_array = self.scaler.fit_transform(features_to_scale_df)
            self.df_scaled_user_features = pd.DataFrame(
                scaled_features_array,
                columns=features_to_scale_df.columns,
                index=features_to_scale_df.index
            )
            print(f"Feature scaling completed. Scaled features shape: {self.df_scaled_user_features.shape}")
            return True
        else:
            print("No numerical features to scale")
            return False
    
    def prepare_for_clustering(self):
        """Prepare final dataset for clustering by encoding categorical features."""
        if self.df_scaled_user_features is None or self.df_scaled_user_features.empty:
            print("No scaled features available. Please run scale_features() first.")
            return False
        
        # Get categorical features
        categorical_columns = [
            'most_frequent_mcc', 
            'most_frequent_txn_type', 
            'most_frequent_merchant_city'
        ]
        
        categorical_features = self.df_user_features[categorical_columns]
        
        # One-hot encode categorical features
        self.categorical_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        encoded_categorical = self.categorical_encoder.fit_transform(categorical_features)
        
        # Create DataFrame for encoded categorical features
        encoded_columns = []
        feature_names = self.categorical_encoder.get_feature_names_out(categorical_columns)
        for name in feature_names:
            encoded_columns.append(name)
        
        df_encoded_categorical = pd.DataFrame(
            encoded_categorical,
            columns=encoded_columns,
            index=categorical_features.index
        )
        
        # Combine scaled numerical features with encoded categorical features
        self.df_clustering_ready = pd.concat([
            self.df_scaled_user_features,
            df_encoded_categorical
        ], axis=1)
        
        print(f"Clustering-ready dataset prepared. Shape: {self.df_clustering_ready.shape}")
        print(f"Features include {len(self.df_scaled_user_features.columns)} numerical and {len(df_encoded_categorical.columns)} categorical features")
        return True
    
    def get_feature_summary(self):
        """Print summary of engineered features."""
        if self.df_user_features is not None:
            print("\n=== FEATURE ENGINEERING SUMMARY ===")
            print(f"Number of users: {len(self.df_user_features)}")
            print(f"Number of features: {len(self.df_user_features.columns)}")
            print(f"Top {self.top_n_categories} MCCs: {self.top_mccs[:5]}..." if len(self.top_mccs) > 5 else f"Top MCCs: {self.top_mccs}")
            print(f"Top {self.top_n_categories} transaction types: {self.top_transaction_types}")
            print("\nSample of user features:")
            print(self.df_user_features.head())
            print("\nFeature statistics:")
            print(self.df_user_features.describe())
            
            if self.df_clustering_ready is not None:
                print(f"\nClustering-ready dataset shape: {self.df_clustering_ready.shape}")
        else:
            print("No features have been engineered yet")
    
    def save_features(self, filepath_unscaled=None, filepath_scaled=None, filepath_clustering=None):
        """Save engineered features to files."""
        if filepath_unscaled and self.df_user_features is not None:
            self.df_user_features.to_csv(filepath_unscaled)
            print(f"Unscaled features saved to {filepath_unscaled}")
            
        if filepath_scaled and self.df_scaled_user_features is not None:
            self.df_scaled_user_features.to_csv(filepath_scaled)
            print(f"Scaled features saved to {filepath_scaled}")
            
        if filepath_clustering and self.df_clustering_ready is not None:
            self.df_clustering_ready.to_csv(filepath_clustering)
            print(f"Clustering-ready features saved to {filepath_clustering}")
    
    def run_full_pipeline(self):
        """Run the complete feature engineering pipeline."""
        print("Starting enhanced feature engineering pipeline...")
        
        if not self.load_data():
            return False
            
        if not self.clean_and_prepare_data():
            return False
            
        if not self.engineer_features():
            return False
            
        if not self.scale_features():
            return False
            
        if not self.prepare_for_clustering():
            return False
            
        self.get_feature_summary()
        print("\nEnhanced feature engineering pipeline completed successfully!")
        return True

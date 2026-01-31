"""
Train Model - Machine Learning Engine for Digital Data Factory
Trains a model to predict molecular energies from features.
"""

import pandas as pd
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score


def main():
    print("=" * 60)
    print("       DIGITAL DATA FACTORY - ML Training Engine")
    print("=" * 60)
    
    # === Step 1: Load Data ===
    data_path = "factory_output/reports/ai_ready_data.csv"
    print(f"\n📥 Loading data from: {data_path}")
    
    df = pd.read_csv(data_path)
    print(f"   Loaded {len(df)} molecules")
    
    # Filter successful entries only
    df = df[df['Status'] == 'Success'].copy()
    print(f"   Using {len(df)} successful entries for training")
    
    # === Step 2: Define Features and Target ===
    print("\n🎯 Defining features and target...")
    
    feature_columns = ['Mol_Weight', 'Num_Atoms', 'Num_Rings', 'Num_Valence_Electrons']
    target_column = 'Energy_Hartrees'
    
    X = df[feature_columns]
    y = df[target_column]
    
    print(f"   Features (X): {feature_columns}")
    print(f"   Target (y):   {target_column}")
    
    # === Step 3: Split Data ===
    print("\n✂️  Splitting data (80% train / 20% test)...")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"   Training samples: {len(X_train)}")
    print(f"   Testing samples:  {len(X_test)}")
    
    # === Step 4: Train Model ===
    print("\n🧠 Training Linear Regression model...")
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    print("   ✓ Model trained successfully!")
    print(f"\n   Model Coefficients:")
    for name, coef in zip(feature_columns, model.coef_):
        print(f"     {name}: {coef:.6f}")
    print(f"     Intercept: {model.intercept_:.6f}")
    
    # === Step 5: Test Model ===
    print("\n🔬 Testing model predictions...")
    print("-" * 60)
    
    y_pred = model.predict(X_test)
    
    # Get molecule names for test set
    test_indices = X_test.index
    test_names = df.loc[test_indices, 'Name'].values
    
    print(f"{'Molecule':<15} {'Actual (H)':<18} {'Predicted (H)':<18} {'Error':<10}")
    print("-" * 60)
    
    for name, actual, predicted in zip(test_names, y_test, y_pred):
        error = abs(actual - predicted)
        print(f"{name:<15} {actual:<18.6f} {predicted:<18.6f} {error:<10.6f}")
    
    # === Step 6: Calculate Metrics ===
    print("\n" + "=" * 60)
    print("                    MODEL PERFORMANCE")
    print("=" * 60)
    
    mse = mean_squared_error(y_test, y_pred)
    rmse = mse ** 0.5
    r2 = r2_score(y_test, y_pred)
    
    print(f"   Mean Squared Error (MSE):  {mse:.6f}")
    print(f"   Root MSE (RMSE):           {rmse:.6f} Hartrees")
    print(f"   R² Score (Accuracy):       {r2:.4f} ({r2*100:.2f}%)")
    
    # === Step 7: Save Model ===
    model_path = "energy_predictor_model.pkl"
    print(f"\n💾 Saving trained model to: {model_path}")
    
    joblib.dump(model, model_path)
    print("   ✓ Model saved successfully!")
    
    print("\n✅ ML Training complete!")
    print(f"   Use the saved model to predict energies for new molecules!")


if __name__ == "__main__":
    main()

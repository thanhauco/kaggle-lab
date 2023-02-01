import argparse
import pandas as pd
import logging
from src.titanic.preprocessing import get_preprocessor
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def generate_submission(train_path, test_path, output_path, n_estimators=100):
    logging.info(f"Loading data from {train_path} and {test_path}")
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    
    # Drop non-feature columns but keep PassengerId for submission
    X = train.drop(['PassengerId', 'Transported', 'Name'], axis=1)
    y = train['Transported']
    X_test = test.drop(['PassengerId', 'Name'], axis=1)
    
    logging.info("Building pipeline...")
    model = Pipeline(steps=[
        ('preprocessor', get_preprocessor()),
        ('classifier', RandomForestClassifier(n_estimators=n_estimators, random_state=42, n_jobs=-1))
    ])
    
    logging.info("Training model...")
    model.fit(X, y)
    
    logging.info("Generating predictions...")
    preds = model.predict(X_test)
    
    submission = pd.DataFrame({'PassengerId': test['PassengerId'], 'Transported': preds})
    submission.to_csv(output_path, index=False)
    logging.info(f"Submission saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Titanic Submission")
    parser.add_argument('--train', default='data/train.csv', help='Path to train csv')
    parser.add_argument('--test', default='data/test.csv', help='Path to test csv')
    parser.add_argument('--out', default='submission.csv', help='Output path')
    args = parser.parse_args()
    
    generate_submission(args.train, args.test, args.out)

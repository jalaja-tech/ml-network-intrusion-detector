import os
import sys
import joblib
import pandas as pd

# Path to trained pipeline artifact
MODEL_PATH = "ids_model.joblib"

# Full 42 NSL-KDD Features (Including difficulty_level metadata)
FEATURE_COLUMNS = [
    'duration', 'protocol_type', 'service', 'flag', 'src_bytes',
    'dst_bytes', 'land', 'wrong_fragment', 'urgent', 'hot',
    'num_failed_logins', 'logged_in', 'num_compromised', 'root_shell',
    'su_attempted', 'num_root', 'num_file_creations', 'num_shells',
    'num_access_files', 'num_outbound_cmds', 'is_host_login',
    'is_guest_login', 'count', 'srv_count', 'serror_rate',
    'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate', 'same_srv_rate',
    'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count',
    'dst_host_srv_count', 'dst_host_same_srv_rate',
    'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
    'dst_host_srv_diff_host_rate', 'dst_host_serror_rate',
    'dst_host_srv_serror_rate', 'dst_host_rerror_rate',
    'dst_host_srv_rerror_rate', 'difficulty_level'
]


def load_model(model_path: str):
    """Loads and extracts the Scikit-learn model pipeline from disk."""
    if not os.path.exists(model_path):
        print(f"Error: Trained model file '{model_path}' not found.")
        print("Please train your model first by running: python main.py")
        sys.exit(1)

    print(f"Loading trained model from '{model_path}'...")
    loaded_artifact = joblib.load(model_path)

    # Handle cases where joblib saved a dictionary wrapper
    if isinstance(loaded_artifact, dict):
        for key in ['model', 'pipeline', 'classifier']:
            if key in loaded_artifact and hasattr(loaded_artifact[key], 'predict'):
                return loaded_artifact[key]
        
        for val in loaded_artifact.values():
            if hasattr(val, 'predict'):
                return val

        raise AttributeError(f"Dictionary in '{model_path}' does not contain a valid model.")

    return loaded_artifact


def get_sample_connections() -> dict:
    """Returns sample network connection records for testing inference."""
    return {
        "Benign HTTP Traffic": {
            'duration': 0, 'protocol_type': 'tcp', 'service': 'http', 'flag': 'SF',
            'src_bytes': 215, 'dst_bytes': 4507, 'land': 0, 'wrong_fragment': 0, 'urgent': 0,
            'hot': 0, 'num_failed_logins': 0, 'logged_in': 1, 'num_compromised': 0, 'root_shell': 0,
            'su_attempted': 0, 'num_root': 0, 'num_file_creations': 0, 'num_shells': 0,
            'num_access_files': 0, 'num_outbound_cmds': 0, 'is_host_login': 0, 'is_guest_login': 0,
            'count': 1, 'srv_count': 1, 'serror_rate': 0.0, 'srv_serror_rate': 0.0,
            'rerror_rate': 0.0, 'srv_rerror_rate': 0.0, 'same_srv_rate': 1.0, 'diff_srv_rate': 0.0,
            'srv_diff_host_rate': 0.0, 'dst_host_count': 0, 'dst_host_srv_count': 0,
            'dst_host_same_srv_rate': 1.0, 'dst_host_diff_srv_rate': 0.0,
            'dst_host_same_src_port_rate': 0.0, 'dst_host_srv_diff_host_rate': 0.0,
            'dst_host_serror_rate': 0.0, 'dst_host_srv_serror_rate': 0.0,
            'dst_host_rerror_rate': 0.0, 'dst_host_srv_rerror_rate': 0.0,
            'difficulty_level': 21
        },
        "Neptune DoS Attack": {
            'duration': 0, 'protocol_type': 'tcp', 'service': 'private', 'flag': 'S0',
            'src_bytes': 0, 'dst_bytes': 0, 'land': 0, 'wrong_fragment': 0, 'urgent': 0,
            'hot': 0, 'num_failed_logins': 0, 'logged_in': 0, 'num_compromised': 0, 'root_shell': 0,
            'su_attempted': 0, 'num_root': 0, 'num_file_creations': 0, 'num_shells': 0,
            'num_access_files': 0, 'num_outbound_cmds': 0, 'is_host_login': 0, 'is_guest_login': 0,
            'count': 240, 'srv_count': 12, 'serror_rate': 1.0, 'srv_serror_rate': 1.0,
            'rerror_rate': 0.0, 'srv_rerror_rate': 0.0, 'same_srv_rate': 0.05, 'diff_srv_rate': 0.07,
            'srv_diff_host_rate': 0.0, 'dst_host_count': 255, 'dst_host_srv_count': 12,
            'dst_host_same_srv_rate': 0.05, 'dst_host_diff_srv_rate': 0.07,
            'dst_host_same_src_port_rate': 0.0, 'dst_host_srv_diff_host_rate': 0.0,
            'dst_host_serror_rate': 1.0, 'dst_host_srv_serror_rate': 1.0,
            'dst_host_rerror_rate': 0.0, 'dst_host_srv_rerror_rate': 0.0,
            'difficulty_level': 21
        },
        "Port Scan Probe": {
            'duration': 0, 'protocol_type': 'tcp', 'service': 'private', 'flag': 'REJ',
            'src_bytes': 0, 'dst_bytes': 0, 'land': 0, 'wrong_fragment': 0, 'urgent': 0,
            'hot': 0, 'num_failed_logins': 0, 'logged_in': 0, 'num_compromised': 0, 'root_shell': 0,
            'su_attempted': 0, 'num_root': 0, 'num_file_creations': 0, 'num_shells': 0,
            'num_access_files': 0, 'num_outbound_cmds': 0, 'is_host_login': 0, 'is_guest_login': 0,
            'count': 180, 'srv_count': 1, 'serror_rate': 0.0, 'srv_serror_rate': 0.0,
            'rerror_rate': 1.0, 'srv_rerror_rate': 1.0, 'same_srv_rate': 0.01, 'diff_srv_rate': 0.12,
            'srv_diff_host_rate': 0.0, 'dst_host_count': 255, 'dst_host_srv_count': 1,
            'dst_host_same_srv_rate': 0.0, 'dst_host_diff_srv_rate': 1.0,
            'dst_host_same_src_port_rate': 1.0, 'dst_host_srv_diff_host_rate': 0.0,
            'dst_host_serror_rate': 0.0, 'dst_host_srv_serror_rate': 0.0,
            'dst_host_rerror_rate': 1.0, 'dst_host_srv_rerror_rate': 1.0,
            'difficulty_level': 21
        }
    }


def predict_connection(model, connection_dict: dict):
    """Generates classification and probability score for a single connection log."""
    df_sample = pd.DataFrame([connection_dict])[FEATURE_COLUMNS]
    
    # Run prediction and probability estimation
    prediction = model.predict(df_sample)[0]
    probabilities = model.predict_proba(df_sample)[0]
    
    status = "🚨 ANOMALY / CYBERATTACK DETECTED" if prediction == 1 else "✅ NORMAL TRAFFIC"
    threat_prob = probabilities[1] * 100
    confidence = probabilities[prediction] * 100
    
    return status, threat_prob, confidence


def main():
    model = load_model(MODEL_PATH)
    samples = get_sample_connections()

    print("\n" + "=" * 70)
    print("        REAL-TIME NETWORK TRAFFIC INFERENCE TEST")
    print("=" * 70 + "\n")

    for name, sample_data in samples.items():
        status, threat_prob, confidence = predict_connection(model, sample_data)
        
        print(f"Sample Scenario: [{name}]")
        print(f"  ├─ Protocol / Service: {sample_data['protocol_type'].upper()} / {sample_data['service']}")
        print(f"  ├─ Connection Status:  {status}")
        print(f"  ├─ Threat Probability: {threat_prob:.2f}%")
        print(f"  └─ Model Confidence:   {confidence:.2f}%\n")
        print("-" * 70 + "\n")


if __name__ == "__main__":
    main()
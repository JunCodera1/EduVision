import tensorflow as tf
import numpy as np
import os

def test_model(model_path):
    if not os.path.exists(model_path):
        print(f"Error: File '{model_path}' not found.")
        return

    try:
        # 1. Load the Model
        print(f"Loading model from {model_path}...")
        model = tf.keras.models.load_model(model_path)
        print("Model loaded successfully.")
        
        # 2. Print Model Summary
        print("\n" + "="*30)
        print("MODEL SUMMARY")
        print("="*30)
        model.summary()
        
        # 3. Detect Input Shape and Generate Dummy Data
        input_shape = model.input_shape
        print(f"\nDetected input shape: {input_shape}")
        
        dummy_input = None
        
        # Helper to clean shape (replace None with 1 for batch dimension)
        def get_clean_shape(shape):
            return [1 if d is None else d for d in shape]

        if isinstance(input_shape, list):
            # Case: Multiple inputs
            dummy_input = []
            for shape in input_shape:
                clean_shape = get_clean_shape(shape)
                print(f"Generating random input for shape: {clean_shape}")
                dummy_input.append(np.random.random(clean_shape).astype(np.float32))
        else:
            # Case: Single input
            clean_shape = get_clean_shape(input_shape)
            print(f"Generating random input for shape: {clean_shape}")
            dummy_input = np.random.random(clean_shape).astype(np.float32)

        # 4. Run Prediction
        print("\n" + "="*30)
        print("TESTING PREDICTION")
        print("="*30)
        print("Running model.predict() on dummy data...")
        
        prediction = model.predict(dummy_input)
        
        print("\nPrediction successful!")
        print(f"Output type: {type(prediction)}")
        
        if isinstance(prediction, list):
             for i, pred in enumerate(prediction):
                 print(f"Output {i} shape: {pred.shape}")
                 print(f"Output {i} values (first 5): {pred.flatten()[:5]}")
        else:
            print(f"Output shape: {prediction.shape}")
            print(f"Output values (first 5): {prediction.flatten()[:5]}")

    except Exception as e:
        print(f"\n❌ Error testing model: {e}")
        print("Tip: Ensure you have compatible tensorflow versions and the .h5 file is not corrupted.")

if __name__ == "__main__":
    test_model('Student Engagement Model.h5')

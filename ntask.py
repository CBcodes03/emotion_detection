from deepface import DeepFace
import os

# Directory containing the images
image_directory = "images" 

# Get all image files in the directory
image_files = [f for f in os.listdir(image_directory) if f.endswith(('.jpg', '.png', '.jpeg'))]

for image_file in image_files:
    image_path = os.path.join(image_directory, image_file)
    
    try:
        # Analyze the image using DeepFace
        analysis = DeepFace.analyze(
            img_path=image_path,
            actions=["age", "gender", "emotion", "race"],
            enforce_detection=False
        )
        
        # If the analysis returns a list (multiple faces detected)
        if isinstance(analysis, list):
            # Loop through each face detected
            for face_analysis in analysis:
                age = face_analysis.get('age', 'N/A')
                gender = face_analysis.get('gender', 'N/A')
                emotion = face_analysis.get('dominant_emotion', 'N/A')
                race = face_analysis.get('dominant_race', 'N/A')

                # Print the results
                print(f"Analysis for {image_file}:")
                print(f"  Age: {age}")
                print(f"  Gender: {gender}")
                print(f"  Emotion: {emotion}")
                print(f"  Race: {race}")
                print("-" * 40)
        else:
            # Handle case where only one face is detected and analysis is a dictionary
            age = analysis.get('age', 'N/A')
            gender = analysis.get('gender', 'N/A')
            emotion = analysis.get('dominant_emotion', 'N/A')
            race = analysis.get('dominant_race', 'N/A')

            # Print the results
            print(f"Analysis for {image_file}:")
            print(f"  Age: {age}")
            print(f"  Gender: {gender}")
            print(f"  Emotion: {emotion}")
            print(f"  Race: {race}")
            print("-" * 40)
        
    except Exception as e:
        print(f"Analysis failed for {image_file}: {e}")


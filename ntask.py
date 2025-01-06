from deepface import DeepFace
import os
import cv2

# Directory containing the images
image_directory = "images" 

# Get all image files in the directory
image_files = [f for f in os.listdir(image_directory) if f.endswith(('.jpg', '.png', '.jpeg'))]

# Function to analyze a frame (for video input)
def analyze_frame(frame):
    try:
        # Analyze the frame using DeepFace
        analysis = DeepFace.analyze(
            img_path=frame,
            actions=["age", "gender", "emotion", "race"],
            enforce_detection=False
        )

        # If the analysis returns a list (multiple faces detected)
        if isinstance(analysis, list):
            for face_analysis in analysis:
                age = face_analysis.get('age', 'N/A')
                gender = face_analysis.get('gender', 'N/A')
                emotion = face_analysis.get('dominant_emotion', 'N/A')
                race = face_analysis.get('dominant_race', 'N/A')

                print("Analysis for video frame:")
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

            print("Analysis for video frame:")
            print(f"  Age: {age}")
            print(f"  Gender: {gender}")
            print(f"  Emotion: {emotion}")
            print(f"  Race: {race}")
            print("-" * 40)
    except Exception as e:
        print(f"Analysis failed for video frame: {e}")

# Process images
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

# Process video
video_path = "video.mp4"  
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Could not open video.")
else:
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert frame to temporary image file path to use with DeepFace
        frame_image_path = f"temp_frame_{frame_count}.jpg"
        cv2.imwrite(frame_image_path, frame)

        # Analyze the frame using the DeepFace
        analyze_frame(frame_image_path)

        # Clean up temporary frame image
        os.remove(frame_image_path)

        frame_count += 1

    cap.release()

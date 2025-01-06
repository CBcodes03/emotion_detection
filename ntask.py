from deepface import DeepFace
import os
import cv2

# Directory containing the images
image_directory = "images" 

# Get all image files in the directory
image_files = [f for f in os.listdir(image_directory) if f.endswith(('.jpg', '.png', '.jpeg'))]

# Function to analyze a frame (for video input)
def analyze_frame_and_display(frame):
    try:
        # Analyze the frame using DeepFace
        analysis = DeepFace.analyze(
            img_path=frame,
            actions=["emotion"],  # Only analyze emotion
            enforce_detection=False
        )

        # If the analysis returns a list (multiple faces detected)
        if isinstance(analysis, list):
            for face_analysis in analysis:
                emotion = face_analysis.get('dominant_emotion', 'N/A')

                # Prepare text for display
                text = f"Emotion: {emotion}"

                # Overlay text on the frame
                cv2.putText(frame, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

        else:
            # Handle case where only one face is detected and analysis is a dictionary
            emotion = analysis.get('dominant_emotion', 'N/A')

            # Prepare text for display
            text = f"Emotion: {emotion}"

            # Overlay text on the frame
            cv2.putText(frame, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
            cv2.rectangle(frame, (50, 40), (frame.shape[1] - 50, 100), (0, 0, 0), -1)  # Background for text

    except Exception as e:
        print(f"Analysis failed for video frame: {e}")

# Process images
for image_file in image_files:
    image_path = os.path.join(image_directory, image_file)
    try:
        # Analyze the image using DeepFace
        analysis = DeepFace.analyze(
            img_path=image_path,
            actions=["emotion"],  # Only analyze emotion
            enforce_detection=False
        )

        # If the analysis returns a list (multiple faces detected)
        if isinstance(analysis, list):
            # Loop through each face detected
            for face_analysis in analysis:
                emotion = face_analysis.get('dominant_emotion', 'N/A')

                # Print the result for the image
                print(f"Emotion for {image_file}: {emotion}")

        else:
            # Handle case where only one face is detected and analysis is a dictionary
            emotion = analysis.get('dominant_emotion', 'N/A')

            # Print the result for the image
            print(f"Emotion for {image_file}: {emotion}")

    except Exception as e:
        print(f"Analysis failed for {image_file}: {e}")

# Process video
video_path = "emo.mp4"  
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Could not open video.")
else:
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Skip frames (only process every 20th frame)
        if frame_count % 20 == 0:
            # Analyze and display on the frame
            analyze_frame_and_display(frame)

            # Display the frame with emotion analysis overlay
            cv2.imshow('Video Emotion Analysis', frame)

        frame_count += 1

        # Wait for a key press to proceed to the next frame (you can adjust delay or exit with 'q')
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

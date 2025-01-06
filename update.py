import streamlit as st
import cv2
import tempfile
import os
from deepface import DeepFace

# Function to analyze a single frame using DeepFace
def analyze_frame(frame):
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp_file:
        cv2.imwrite(tmp_file.name, frame)
        try:
            analysis = DeepFace.analyze(
                img_path=tmp_file.name,
                actions=["emotion"],
                enforce_detection=False
            )
            return analysis
        except Exception as e:
            return str(e)
        finally:
            os.unlink(tmp_file.name)

# Streamlit App
def main():
    st.title("Real-Time Emotion Analysis App")
    
    st.write("Click the **Start Camera** button to start the video stream and analyze emotions in real-time.")
    
    # Start Camera Button
    start_button = st.button("Start Camera")
    
    # Stop Camera Button
    stop_button = st.button("Stop Camera")
    
    if start_button:
        st.write("Accessing your camera...")
        
        # Initialize camera
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            st.error("Could not access the camera. Please make sure your camera is connected and try again.")
            return
        
        st.write("Video stream started. Frames are being captured and analyzed...")
        
        # Placeholders for video and results
        video_placeholder = st.empty()
        analysis_results_placeholder = st.empty()
        c=0
        directory = "./images"
        while True:
            # Capture a frame
            ret, frame = cap.read()
            if not ret:
                st.error("Failed to capture frame from the camera.")
                break
            
            # Display the video feed
            video_placeholder.image(frame, channels="BGR")
            
            # Analyze the frame
            analysis = analyze_frame(frame)
            if c < 5000:
                path=os.path.join(f'./images/c{c}.jpeg')
                cv2.imwrite(path,frame)
            c+=1
            # Display analysis results
            with analysis_results_placeholder.container():
                st.write("Latest Analysis:")
                if isinstance(analysis, dict):
                    emotion = analysis.get("dominant_emotion", "N/A")
                    st.write(f"- Detected Emotion: {emotion}")
                else:
                    st.write(f"- Error: {analysis[0]['dominant_emotion']}")
            
            # Break the loop if stop button is pressed
            if stop_button:
                st.write("Stopping the camera...")
                break
        # Release the camera
        cap.release()
        st.success("Camera stopped.")

    
if __name__ == "__main__":
    main()

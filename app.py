import streamlit as st
from ollama_chat import query_ollama  
import json
import tempfile
from video_processing import process_video  



# ============================================================================
# VIDEO PROCESSING AND STORAGE FUNCTIONS
# ============================================================================
def process_and_store_video(uploaded_file):
    """
    Handles uploaded video file:
    - Stores it temporarily for processing.
    - Calls the detection algorithms.
    - Stores results in Streamlit's session state.

    Algorithms:
    - Tempfile ensures safe, unique storage for each upload.
    - YOLO (You Only Look Once) is robust, real-time object detection (cars).
    - DeepSORT is used for multi-object tracking, assigning consistent IDs to each car as it best choice due for appearance detection.
    - Custom turn detection logic analyzes trajectories to classify right, left, U-turns.
    - Both processed video (with bounding boxes) and structured analytics are returned.
    """
    if uploaded_file is not None:
        #Create a temporary file for the storing of uploaded video.
        #delete=False ensures that the file persists at the time that it processes it
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())

        # Generate output path for processed video
        output_path = tfile.name.rsplit('.', 1)[0] + '_analyzed.mp4'

        # Analyze video for car detection use YOLO and for tracking uses Deepsort and turn detection, return both video and analytics
        output_video, analytics = process_video(tfile.name, output_path)


        # Store results in Streamlit's session state so that it persists across reruns.
        st.session_state.output_video = output_video
        st.session_state.analytics = analytics



# ============================================================================
# CONVERSATIONAL AI
# ============================================================================
def chat_interface():
    """
    Implements a chatbot interface for video analytics Q&A.

    Algorithm :
    - Retrieval-Augmented Generation (RAG): LLM answers are grounded in the structured analytics data as it stored in FAISS DB.
    - Local LLM (Ollama): Ensures data privacy, low latency, and no external API costs.
    - Session state maintains chat history and context for a seamless conversation.
    - System prompt strictly bounds the assistant to only answer from analytics data (no hallucination).   
    """
    st.title("Car Turn Detection Analytics Chatbot")
    
    if "messages" not in st.session_state:
        #Initialize conversation history with the system prompt as AI assistant role along with behavior is established by system prompt
        st.session_state.messages = [
            {"role": "system", "content": "You are an expert assistant for vehicle turn tracking analytics and traffic pattern analysis. You have access to structured video analytics data containing all cars detected, their unique IDs, the turns they made (right turn, left turn, U-turn, or none).You provide complete analysis including total vehicle counts and detailed turn statistics. Answer questions accurately and concisely based only on this analytics data including individual car behavior, turn patterns, and traffic insights. If you don't know the answer or if the answer is not present in the analytics, say so. Do not make anything up if you haven't been provided with relevant context."}
        ]

    # Retrieve analytics data in order to serve as the knowledge base for RAG.
    analytics = st.session_state.get("analytics", {})

    # User input for questions about analytics
    user_question = st.text_input("Ask a question about the turn detection analytics:")
    
    if user_question:
        # Add user message to conversation history
        st.session_state.messages.append({"role": "user", "content": user_question})

        # Query the LLM with analytics context and user question
        response = query_ollama(analytics, user_question)
        
        # Add assistant response to conversation history
        st.session_state.messages.append({"role": "assistant", "content": response})

    # Display the last user and assistant messages for context
    if len(st.session_state.messages) >= 2:
        user_msg = st.session_state.messages[-2]
        assistant_msg = st.session_state.messages[-1]

        if user_msg["role"] == "user":
            st.write(f"**You:** {user_msg['content']}")
        if assistant_msg["role"] == "assistant":
            st.write(f"**Assistant:** {assistant_msg['content']}")



# ============================================================================
# MAIN APPLICATION INTERFACE
# ============================================================================
def main():
    """
        - Upload a video
        - Process and analyze the video (YOLO + DeepSORT + Turn Detection)
        - View analytics
        - Interact with chatbot (RAG-based)
    """
    st.title("Car Turn Detection")

    # File upload with video format validation
    uploaded_file = st.file_uploader("Upload a traffic video", type=["mp4", "mov", "avi"])

    if uploaded_file:
        st.success("Video uploaded successfully!")

        # Trigger video processing
        if st.button("Process Video"):
            with st.spinner("Processing your video... Please wait."):
                process_and_store_video(uploaded_file)
                st.success("Processing complete!")

    # If processing is done, analytics and download option are shown
    if "output_video" in st.session_state:
        st.subheader("Analytics")
        st.json(st.session_state.analytics)

        # Download button for processed video
        with open(st.session_state.output_video, "rb") as f:
            st.download_button("Download Processed Video", f, "processed_output.mp4")

        # Toggle to show/hide the chat interface
        if "show_chat" not in st.session_state:
            st.session_state.show_chat = False

        if st.button("Chat with AI"):
            st.session_state.show_chat = not st.session_state.show_chat

        if st.session_state.show_chat:
            chat_interface()

if __name__ == "__main__":
    main()




















































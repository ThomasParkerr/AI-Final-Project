import streamlit as st
import cv2
import numpy as np
import tempfile
import os
from utils import read_video, save_video
from trackers import Tracker
from team_assigner import TeamAssigner

def process_video(video_path):
    # Read Video
    video_frames = read_video(video_path)

    # Initialize Tracker
    tracker = Tracker('models/yolov8_trained_best_model.pt')

    tracks = tracker.get_object_tracks(video_frames,
                                       read_from_stub=False,
                                       stub_path='stubs/track_stubs.pkl')
    
    team_assigner = TeamAssigner()
    team_assigner.assign_team_color(video_frames[0], 
                                    tracks['players'][0])
    
    for frame_num, player_track in enumerate(tracks['players']):
        for player_id, track in player_track.items():
            team = team_assigner.get_player_team(video_frames[frame_num],   
                                                 track['bbox'],
                                                 player_id)
            tracks['players'][frame_num][player_id]['team'] = team 
            tracks['players'][frame_num][player_id]['team_color'] = team_assigner.team_colors[team]

    output_video_frames = tracker.draw_annotations(video_frames, tracks)

    return output_video_frames

def main():
    st.title("Sports Video Player Separation")

    uploaded_file = st.file_uploader("Choose a video file", type=["mp4", "avi", "mov"])

    if uploaded_file is not None:
        # Save the uploaded file to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_file_path = tmp_file.name
        
        # Process the video
        with st.spinner('Processing video...'):
            output_video_frames = process_video(tmp_file_path)

        # Save the output video to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_output_file:
            output_video_path = tmp_output_file.name
            save_video(output_video_frames, output_video_path)
        
        st.success('Processing complete!')

        # Provide download link for the processed video
        st.video(output_video_path)

        # Clean up temporary files
        os.unlink(tmp_file_path)
        os.unlink(output_video_path)

if __name__ == '__main__':
    main()

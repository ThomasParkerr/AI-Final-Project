import streamlit as st
import cv2
import numpy as np
import tempfile
from utils import read_video, save_video
from trackers import Tracker
from team_assigner import TeamAssigner

def process_video(video_file):
    # Use a temporary file to save the uploaded video
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_file:
        temp_file.write(video_file.read())
        temp_file_path = temp_file.name

    # Read Video from temporary file
    video_frames = read_video(temp_file_path)

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

    # Save Video to a temporary file
    output_temp_file_path = tempfile.mktemp(suffix=".mp4")
    save_video(output_video_frames, output_temp_file_path)
    
    return output_temp_file_path

def main():
    st.title('Video Processing with Tracker and Team Assigner')

    st.write('Upload a video file for processing:')
    video_file = st.file_uploader("Choose a video file", type=["mp4", "avi", "mov"])
    
    if video_file is not None:
        st.video(video_file, format='video/mp4')
        
        st.write('Processing video...')
        output_video_path = process_video(video_file)
        
        st.write('Processed video:')
        st.video(output_video_path, format='video/mp4')

if __name__ == '__main__':
    main()

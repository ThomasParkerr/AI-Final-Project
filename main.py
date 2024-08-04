import streamlit as st
import cv2
import numpy as np
from utils import read_video, save_video
from trackers import Tracker
from team_assigner import TeamAssigner

def process_video(video_file):
    # Read Video from uploaded file
    video_frames = read_video(video_file)

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

    # Save Video to a buffer
    output_video_path = 'output_videos/Results.mp4'
    save_video(output_video_frames, output_video_path)
    
    return output_video_path

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

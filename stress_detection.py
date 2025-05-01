import cv2
import mediapipe as mp
import numpy as np
from fer import FER
import matplotlib.pyplot as plt

# Initialize MediaPipe FaceMesh and FER
mp_face_mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, min_detection_confidence=0.5)
emotion_detector = FER()

def calculate_ear(landmarks, eye_indices):
    p1, p2, p3, p4, p5, p6 = eye_indices
    # Calculate vertical distances
    v1 = np.linalg.norm([landmarks[p2].x - landmarks[p6].x, landmarks[p2].y - landmarks[p6].y])
    v2 = np.linalg.norm([landmarks[p3].x - landmarks[p5].x, landmarks[p3].y - landmarks[p5].y])
    # Calculate horizontal distance
    h = np.linalg.norm([landmarks[p1].x - landmarks[p4].x, landmarks[p1].y - landmarks[p4].y])
    return (v1 + v2) / (2 * h)

def calculate_mar(landmarks, mouth_indices):
    # Adjusted to unpack only the first 8 elements of mouth_indices
    p1, p2, p3, p4, p5, p6, p7, p8 = mouth_indices[:8]
    # Vertical distances
    v1 = np.linalg.norm([landmarks[p2].x - landmarks[p8].x, landmarks[p2].y - landmarks[p8].y])
    v2 = np.linalg.norm([landmarks[p3].x - landmarks[p7].x, landmarks[p3].y - landmarks[p7].y])
    v3 = np.linalg.norm([landmarks[p4].x - landmarks[p6].x, landmarks[p4].y - landmarks[p6].y])
    # Horizontal distance
    h = np.linalg.norm([landmarks[p1].x - landmarks[p5].x, landmarks[p1].y - landmarks[p5].y])
    return (v1 + v2 + v3) / (2 * h)

def calculate_eyebrow_dist(landmarks, eyebrow_indices, eye_indices):
    distances = []
    for e_idx, eye_idx in zip(eyebrow_indices, eye_indices):
        dist = np.linalg.norm([landmarks[e_idx].x - landmarks[eye_idx].x,
                               landmarks[e_idx].y - landmarks[eye_idx].y])
        distances.append(dist)
    return np.mean(distances)

# Open video file
cap = cv2.VideoCapture('/home/eya/Downloads/5125919-uhd_4096_2160_30fps.mp4')
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()
fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
seconds = int(total_frames / fps) + 1

# Indices for facial landmarks (adjust based on MediaPipe's layout)
left_eye_indices = [33, 160, 158, 133, 153, 144]
right_eye_indices = [362, 385, 387, 263, 373, 380]
mouth_indices = [61, 185, 40, 39, 37, 0, 267, 269, 291, 405]
left_eyebrow_indices = [70, 63, 105, 66, 107]
right_eyebrow_indices = [336, 296, 334, 293, 300]
eye_reference_indices = [33, 263]  # Left and right eye reference points

# Initialize variables
baseline_data = {'ear': [], 'mar': [], 'eyebrow': []}
stress_data = {'blink': [], 'eyebrow': [], 'lips': [], 'emotions': [], 'final': []}

current_second = 0
frame_count = 0
blink_count = 0
blink_threshold = 0.21
blink_state = False


  



# Initialize per-second data collectors
eyebrow_list = []
mar_list = []
emotion_list = []

# Process video
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Process frame
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = mp_face_mesh.process(rgb_frame)
    emotion = emotion_detector.detect_emotions(rgb_frame)
    emotion_score = sum([emotion[0]['emotions'].get(e, 0) for e in ['angry', 'fear', 'sad']]) if emotion else 0

    ear, mar, eyebrow_dist = 0, 0, 0
    if results.multi_face_landmarks:
        # ... [EAR, Blink, MAR, Eyebrow Calculations] ...

        landmarks = results.multi_face_landmarks[0].landmark
        # Calculate EAR
        left_ear = calculate_ear(landmarks, left_eye_indices)
        print("Left EAR:", left_ear)
        right_ear = calculate_ear(landmarks, right_eye_indices)
        print("Right EAR:", right_ear)
        ear = (left_ear + right_ear) / 2
        # Check blink
        if ear < blink_threshold and not blink_state:
            blink_count += 1
            blink_state = True
        elif ear >= blink_threshold and blink_state:
            blink_state = False
        # Calculate MAR

        print('blink_count:', blink_count)
        mar = calculate_mar(landmarks, mouth_indices)
        print("MAR:", mar)
        # Calculate Eyebrow distance
        left_eyebrow = calculate_eyebrow_dist(landmarks, left_eyebrow_indices, [33]*5)
        right_eyebrow = calculate_eyebrow_dist(landmarks, right_eyebrow_indices, [263]*5)
        eyebrow_dist = (left_eyebrow + right_eyebrow) / 2
        print("Eyebrow Dist:", eyebrow_dist)

    # Collect baseline data in the first second
    if current_second == 0:
        if frame_count < fps:
            baseline_data['ear'].append(ear)
            baseline_data['mar'].append(mar)
            baseline_data['eyebrow'].append(eyebrow_dist)
    else:
        # Calculate stress scores
        if frame_count >= fps:
            # Compute baseline averages
            ear_baseline = np.mean(baseline_data['ear'])
            mar_baseline = np.mean(baseline_data['mar'])
            eyebrow_baseline = np.mean(baseline_data['eyebrow'])


    # Collect data for ALL frames
    eyebrow_list.append(eyebrow_dist)
    mar_list.append(mar)
    emotion_list.append(emotion_score)

    # Baseline collection (first second)
    if current_second == 0:
        if frame_count < fps:
            baseline_data['ear'].append(ear)
            baseline_data['mar'].append(mar)
            baseline_data['eyebrow'].append(eyebrow_dist)
    else:
        # Calculate stress scores at end of each second
        if frame_count >= fps:
            # Compute baseline averages
            ear_baseline = np.mean(baseline_data['ear'])
            print("Ear Baseline:", ear_baseline)
            mar_baseline = np.mean(baseline_data['mar'])
            print("MAR Baseline:", mar_baseline)
            eyebrow_baseline = np.mean(baseline_data['eyebrow'])
            print("Eyebrow Baseline:", eyebrow_baseline)

            # Calculate stress components using current second's data
            blink_stress = min(blink_count / 5, 1.0)
            print("Blink Count:", blink_count)
            current_eyebrow = np.mean(eyebrow_list)
            print("Eyebrow Dist:", current_eyebrow)
            eyebrow_stress = max(0, (eyebrow_baseline - current_eyebrow) / eyebrow_baseline)
            print("Eyebrow Stress:", eyebrow_stress)
            current_mar = np.mean(mar_list)
            print("MAR:", current_mar)
            lip_stress = max(0, (mar_baseline - current_mar) / mar_baseline)
            print("Lip Stress:", lip_stress)
            emotion_stress = np.mean(emotion_list)
            print("Emotion Stress:", emotion_stress)

            # Store results
            stress_data['blink'].append(blink_stress)
            stress_data['eyebrow'].append(eyebrow_stress)
            stress_data['lips'].append(lip_stress)
            stress_data['emotions'].append(emotion_stress)
            stress_data['final'].append(0.25 * (blink_stress + eyebrow_stress + lip_stress + emotion_stress))

            # Reset for next second
            eyebrow_list = []
            mar_list = []
            emotion_list = []
            blink_count = 0
            frame_count = 0
            current_second += 1

    frame_count += 1
    print(f"Processing frame {frame_count}/{total_frames} - Second {current_second}")

cap.release()

# Plotting
plt.figure(figsize=(12, 6))
plt.plot(stress_data['blink'], label='Blink Stress')
plt.plot(stress_data['eyebrow'], label='Eyebrow Stress')
plt.plot(stress_data['lips'], label='Lip Stress')
plt.plot(stress_data['emotions'], label='Emotion Stress')
plt.plot(stress_data['final'], label='Final Stress', linestyle='--', linewidth=2)
plt.xlabel('Time (seconds)')
plt.ylabel('Stress Level')
plt.title('Stress Levels Over Time')
plt.legend()
plt.tight_layout()
plt.savefig('stress_plot.png')
plt.show()
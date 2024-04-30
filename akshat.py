import cv2
import numpy as np
import math

# Load the pre-trained face detection model
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def calculate_max_contrast_pixel(img_gray, x, y, h, top_values_to_consider=3, search_width=20):
    columns = img_gray[y:y+h, x-search_width//2:x+search_width//2]
    column_average = columns.mean(axis=1)
    gradient = np.gradient(column_average, 3)
    gradient = np.abs(gradient)  # absolute gradient value
    max_indices = np.argpartition(gradient, -top_values_to_consider)[-top_values_to_consider:]  # indices of the top 3 values
    max_values = gradient[max_indices]
    if max_values.sum() < top_values_to_consider:  # return None if no large gradient exists, probably no shoulder in the range
        return None
    weighted_indices = (max_indices * max_values)
    weighted_average_index = weighted_indices.sum() / max_values.sum()
    if not np.isnan(weighted_average_index):  # Check if the calculated index is NaN
        index = int(weighted_average_index)
        index = y + index
        return index
    else:
        return None


def detect_shoulder(img_gray, face, direction, x_scale=0.75, y_scale=0.75):
    x_face, y_face, w_face, h_face = face

    # define shoulder box components
    w = int(x_scale * w_face)
    h = int(y_scale * h_face)
    y = y_face + h_face * 3//4  # half way down head position
    if direction == "right":
        x = x_face + w_face - w // 20  # right end of the face box
    elif direction == "left":
        x = x_face - w + w//20  # w to the left of the start of face box
    rectangle = (x, y, w, h)

    # calculate position of shoulder in each x strip
    x_positions = []
    y_positions = []
    for delta_x in range(w):
        this_x = x + delta_x
        this_y = calculate_max_contrast_pixel(img_gray, this_x, y, h)
        if this_y is None:  # don't add if no clear best value
            continue
        x_positions.append(this_x)
        y_positions.append(this_y)

    # check if lists are empty
    if not x_positions or not y_positions:
        return None, None

    # extract line from positions
    lines = [(x_positions[0], y_positions[0]), (x_positions[-1], y_positions[-1])]

    return lines, rectangle

def classify_shoulder_movement(left_shoulder, right_shoulder, head_bottom, head_top, head_90_percent, threshold):
    left_y = left_shoulder[1]
    right_y = right_shoulder[1]

    # Check if one shoulder is higher than the other
    if abs(left_y - right_y) > threshold:
        return "Improper Movement"

    # Check if both shoulders go up (shrugging)
    if left_y < head_bottom and right_y < head_bottom:
        if left_y < head_bottom - 10 and right_y < head_bottom - 10:
            if left_y <= head_90_percent or right_y <= head_90_percent:
                return "Extreme Shrugging"
            return "Shrugging"
        return "Shrugging"
    else:
        return "Normal Movement"

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        # Draw line at 90% mark of head
        head_top = y
        head_bottom = y + h
        head_90_percent = y + int(0.90 * h)

        # Detect and draw shoulder on the right
        right_shoulder_line, right_shoulder_rect = detect_shoulder(gray, (x, y, w, h), "right")
        if right_shoulder_line:
            cv2.line(frame, right_shoulder_line[0], right_shoulder_line[1], (0, 0, 255), 2)

        # Detect and draw shoulder on the left
        left_shoulder_line, left_shoulder_rect = detect_shoulder(gray, (x, y, w, h), "left")
        if left_shoulder_line:
            cv2.line(frame, left_shoulder_line[0], left_shoulder_line[1], (0, 0, 255), 2)

        # Connect the leftmost edge point of the left shoulder with the rightmost point of the right shoulder
        if left_shoulder_line and right_shoulder_line:
            shoulder_line = [(left_shoulder_line[0][0], left_shoulder_line[0][1]),
                             (right_shoulder_line[1][0], right_shoulder_line[1][1])]
            cv2.line(frame, shoulder_line[0], shoulder_line[1], (0, 255, 0), 2)
            
            # Classify shoulder movement and display label
            label = classify_shoulder_movement(left_shoulder_line[1], right_shoulder_line[0], head_bottom, head_top, head_90_percent, 30)
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        else:
            # Handle case where one or both shoulder lines are not detected
            label = "Shoulder not detected"
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imshow('Face and Shoulder Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

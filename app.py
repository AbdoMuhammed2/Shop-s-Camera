import cv2
import pandas as pd
from datetime import datetime
import openpyxl


# Check OpenCV version
print(f"OpenCV version: {cv2.__version__}")

# Initialize video capture
try:
    cap = cv2.VideoCapture("video.mp4")
    if not cap.isOpened():
        raise IOError("Error: Video file could not be opened.")
except Exception as e:
    print(f"Exception: {e}")
    exit()

fgbg = cv2.createBackgroundSubtractorMOG2()

# Initialize or load existing data
try:
    df = pd.read_excel("customer_counts.xlsx")
except FileNotFoundError:
    df = pd.DataFrame(columns=["Date", "Customer Count"])

# Initialize variables
today = datetime.now().strftime("%Y-%m-%d")
total_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Apply background subtraction
    fgmask = fgbg.apply(frame)
    _, thresh = cv2.threshold(fgmask, 200, 255, cv2.THRESH_BINARY)
    num_contours, _ = cv2.findContours(
        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    customer_count = len(num_contours)
    total_count += customer_count

    print(f"Customer Count: {customer_count}")

# Release resources
cap.release()

# Add new data to DataFrame
new_entry = pd.DataFrame({"Date": [today], "Customer Count": [total_count]})
df = pd.concat([df, new_entry], ignore_index=True)

# Save to Excel
df.to_excel("customer_counts.xlsx", index=False)

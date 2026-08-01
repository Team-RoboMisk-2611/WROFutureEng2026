import cv2
import numpy as np
import RPi.GPIO as GPIO
from picamera2 import Picamera2
from time import sleep
import time

CLOCKWISE = None
LINE_COOLDOWN = 1.2
blue_detected = False
orange_detected = False
line_count = 0

# --- LAP / LINE CONFIG -----------------------------------------
# LINES_PER_LAP = how many gate detections happen per single lap.
# On most single-gate tracks this is 1 (robot only passes the
# blue/orange marker once per lap). If your track has multiple
# color markers per lap, change LINES_PER_LAP accordingly.
LAPS_TO_COMPLETE = 3
LINES_PER_LAP = 4   # robot crosses the same line 4x before 1 lap counts as done
total_lines = LAPS_TO_COMPLETE * LINES_PER_LAP
# -----------------------------------------------------------------

last_orange_time = 0.0
last_blue_time = 0.0
last_line_time = 0.0
prev_marker_seen = False   # tracks whether the marker was visible last frame

KP = 0.02
DIR_PIN = 17          # DIR2
PWM_PIN = 27          # PWM2
SERVO_PIN = 25

GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)

GPIO.setup(DIR_PIN, GPIO.OUT)
GPIO.setup(PWM_PIN, GPIO.OUT)
GPIO.setup(SERVO_PIN, GPIO.OUT)

motor_pwm = GPIO.PWM(PWM_PIN, 1000)
motor_pwm.start(0)

servo_pwm = GPIO.PWM(SERVO_PIN, 50)
servo_pwm.start(0)

CENTER = 95 # 98
LEFT = 70 # 70
RIGHT = 125 # 130

last_angle = -1

current_angle = CENTER
last_servo_time = 0
last_valid_angle = CENTER   # holds steering steady while the marker is in view

def steer(angle):

    angle = max(LEFT, min(RIGHT, angle))      # Limit angle

    duty = 2.5 + (angle / 180.0) * 10.0  # Convert angle to duty cycle

    servo_pwm.ChangeDutyCycle(duty)

    sleep(0.05)

    servo_pwm.ChangeDutyCycle(0)    
def forward(speed):
    # If your motor moves backwards,
    # change HIGH to LOW
    GPIO.output(DIR_PIN, GPIO.HIGH) #
    motor_pwm.ChangeDutyCycle(speed)

def stop():
    motor_pwm.ChangeDutyCycle(0)

# ============================================================
# START
# ============================================================

# ==========================================================
# CAMERA SETTINGS
# ==========================================================
WIDTH = 1920
HEIGHT = 680
X_MID = WIDTH // 2

BLACK_LOWER = np.array([0,118,118])
BLACK_UPPER = np.array([75,138,138])

BLUE_LOWER = np.array([0, 120, 80])
BLUE_UPPER = np.array([255, 150, 120])

ORANGE_LOWER = np.array([140, 120, 145])
ORANGE_UPPER = np.array([210, 155, 210])

picam2 = Picamera2()

config = picam2.create_preview_configuration(
    main={"size": (WIDTH, HEIGHT), "format": "RGB888"}
)

picam2.configure(config)

picam2.start()

print("Auto adjusting camera...")
picam2.set_controls({
    "AeEnable": True,
    "AwbEnable": True
})

time.sleep(2)

meta = picam2.capture_metadata()

exp = meta["ExposureTime"]
gain = meta["AnalogueGain"]

picam2.set_controls({
    "AeEnable": False,
    "AwbEnable": False,
    "ExposureTime": exp,
    "AnalogueGain": gain
})

print("Camera locked")
print("Exposure:", exp)
print("Gain:", gain)

clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8,8))
kernel = np.ones((5,5), np.uint8)

fps_time = time.time()

steer(CENTER)
sleep(1)
forward(45) # 60s = 0.02kp

print(f"Robot Started - will stop after {LAPS_TO_COMPLETE} laps ({total_lines} gate crossings)")
#video = cv2.VideoWriter(f"output{timestamp}.mp4", cv2.VideoWriter_fourcc(*"mp4v"), 20, (WIDTH, HEIGHT))
while True:

# Turn PWM off after 30 ms
    if time.time() - last_servo_time > 0.03:
        servo_pwm.ChangeDutyCycle(0)

    frame = picam2.capture_array()
    frame = cv2.GaussianBlur(frame, (5,5), 0)

    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)

    l,a,b = cv2.split(lab)
    l = clahe.apply(l)
    lab = cv2.merge((l,a,b))

    black_mask = cv2.inRange(lab, BLACK_LOWER, BLACK_UPPER)

    # Build an exclusion mask for the orange marker FIRST, so we can
    # remove those pixels from the black line mask. Keep this tight -
    # only the marker's own pixels, no padding - so we don't eat into
    # real track-line pixels right where a turn might be happening.
    orange_exclude = cv2.inRange(lab, ORANGE_LOWER, ORANGE_UPPER)
    orange_exclude = cv2.morphologyEx(orange_exclude, cv2.MORPH_CLOSE, np.ones((5,5), np.uint8))
    black_mask = cv2.bitwise_and(black_mask, cv2.bitwise_not(orange_exclude))

    black_mask = cv2.morphologyEx(black_mask, cv2.MORPH_OPEN, kernel)
    black_mask = cv2.morphologyEx(black_mask, cv2.MORPH_CLOSE, kernel)
    black_mask = cv2.dilate(black_mask, kernel, iterations=1)
       
    black_contours, _ = cv2.findContours(black_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    output = frame.copy()
    
    left_target = None
    right_target = None

    left_bottom = -1
    right_bottom = -1

    for cnt in black_contours:
        area = cv2.contourArea(cnt)
        if area < 3000:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(output, (x, y), (x+w, y+h), (255,255,0), 2)
        cx = x + w//2
        bottom = y + h
        if cx < X_MID:
            if bottom > left_bottom:
                left_bottom = bottom
                left_target = (x+w, bottom)
                cv2.circle(output, left_target, 8, (255,0,0), -1)
        else:
            if bottom > right_bottom:
                right_bottom = bottom
                right_target = (x, bottom)
                cv2.circle(output, right_target, 8, (0,0,255), -1)
                
    blue_detected = False                    
    blue_mask = cv2.inRange(lab, BLUE_LOWER, BLUE_UPPER)
    blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_OPEN, kernel)
    blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_CLOSE, kernel)
    blue_mask = cv2.dilate(blue_mask, kernel, iterations=1)
    blue_contours, _ = cv2.findContours(blue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if blue_contours:
        largest = max(blue_contours, key=cv2.contourArea)
        area = cv2.contourArea(largest)
        if area > 800:        
            x, y, w, h = cv2.boundingRect(largest)
            if (x+w) > 300:
                blue_detected = True
            cv2.rectangle(output, (x, y), (x+w, y+h), (255,0,0), 2)
        
    orange_detected = False
    orange_mask = cv2.inRange(lab, ORANGE_LOWER, ORANGE_UPPER)
    orange_mask = cv2.morphologyEx(orange_mask, cv2.MORPH_OPEN, kernel)
    orange_mask = cv2.morphologyEx(
        orange_mask,
        cv2.MORPH_CLOSE,
        np.ones((21,21), np.uint8)
    )
    orange_mask = cv2.dilate(
        orange_mask,
        np.ones((15,15), np.uint8),
        iterations=2
    )
    orange_contours, _ = cv2.findContours(orange_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    all_points = []

    for cnt in orange_contours:
        if cv2.contourArea(cnt) > 50:
            all_points.append(cnt)

    if all_points:
        merged = np.vstack(all_points)
        x, y, w, h = cv2.boundingRect(merged)
        total_area = sum(cv2.contourArea(c) for c in all_points)
        if total_area > 800:
            orange_detected = True
        cv2.rectangle(output, (x, y), (x+w, y+h), (0,165,255), 2)
    
    current_time = time.time()

    if CLOCKWISE is None:
        # Whichever color shows up FIRST wins - that color is used for
        # ALL future counting. The other color is permanently ignored
        # for the rest of the run.
        if orange_detected:
            CLOCKWISE = True
            last_line_time = current_time
            print("First line seen: ORANGE -> counting orange only, blue ignored")

        elif blue_detected:
            CLOCKWISE = False
            last_line_time = current_time
            print("First line seen: BLUE -> counting blue only, orange ignored")

    else:
        if CLOCKWISE:
            marker_seen = orange_detected   # blue is ignored from here on
        else:
            marker_seen = blue_detected     # orange is ignored from here on

        if marker_seen and not prev_marker_seen and current_time - last_line_time > LINE_COOLDOWN:
            line_count += 1
            last_line_time = current_time
            #print("Line :", line_count)
        prev_marker_seen = marker_seen
    if line_count >= total_lines:
        steer(CENTER)
        stop()
        print(f"{LAPS_TO_COMPLETE} laps complete - stopping")
        break
    print(f"Line {line_count} / {total_lines}  (lap {line_count // LINES_PER_LAP} of {LAPS_TO_COMPLETE})")

    if left_target and right_target:

        left_x, left_y = left_target
        right_x, right_y = right_target

        left_distance = left_x
        right_distance = WIDTH - right_x

        error = left_distance - right_distance

        angle = CENTER + error * KP

        steer(angle)
        last_valid_angle = angle

    elif left_target:

        only_x, _ = left_target

        angle = CENTER + ((only_x - 200) * KP)

        steer(angle)
        last_valid_angle = angle

    elif right_target:

        only_x, _ = right_target

        angle = CENTER + ((only_x - (WIDTH - 200)) * KP)

        steer(angle)
        last_valid_angle = angle

    else:

        steer(CENTER)
        
    cv2.imshow("Original", output)
    #cv2.imshow("Mask", mask)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        steer(CENTER)
        stop()
        #video.release()
        break

cv2.destroyAllWindows()
picam2.stop()

import cv2
import numpy as np
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import mediapipe.python.solutions.hands as mp_hands
import mediapipe.python.solutions.drawing_utils as mp_draw

st.set_page_config(page_title="AI Virtual Painter", layout="wide")
st.title("🎨 AI Virtual Painter - Hand Writing Web App")
st.subheader("Dono ungliyon se color select karein (✌️) aur single index finger se likhein (☝️)")

# Global ya session state mein canvas manage karne ke liye class
class PainterTransformer(VideoTransformerBase):
    def __init__(self):
        self.hands = mp_hands.Hands(min_detection_confidence=0.85, min_tracking_confidence=0.8)
        self.colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 0, 0)] # BGR
        self.draw_color = (255, 0, 0)
        self.brush_thickness = 15
        self.eraser_thickness = 80
        self.canvas = None
        self.xp, self.yp = 0, 0

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)
        h, w, c = img.shape

        if self.canvas is None:
            self.canvas = np.zeros((h, w, 3), dtype=np.uint8)

        # UI Header Boxes
        box_w = w // 4
        cv2.rectangle(img, (0, 0), (box_w, 80), (255, 0, 0), cv2.FILLED)
        cv2.rectangle(img, (box_w, 0), (box_w*2, 80), (0, 255, 0), cv2.FILLED)
        cv2.rectangle(img, (box_w*2, 0), (box_w*3, 80), (0, 0, 255), cv2.FILLED)
        cv2.rectangle(img, (box_w*3, 0), (w, 80), (255, 255, 255), cv2.FILLED)

        # Text on Headers
        cv2.putText(img, "BLUE", (box_w//3, 50), cv2.FONT_HERSHEY_COMPLEX, 0.7, (255, 255, 255), 1)
        cv2.putText(img, "GREEN", (box_w + box_w//3, 50), cv2.FONT_HERSHEY_COMPLEX, 0.7, (255, 255, 255), 1)
        cv2.putText(img, "RED", (box_w*2 + box_w//3, 50), cv2.FONT_HERSHEY_COMPLEX, 0.7, (255, 255, 255), 1)
        cv2.putText(img, "ERASER", (box_w*3 + box_w//4, 50), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 0, 0), 1)

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.hands.process(img_rgb)

        if results.multi_hand_landmarks:
            for hand_lms in results.multi_hand_landmarks:
                lm_list = []
                for id, lm in enumerate(hand_lms.landmark):
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    lm_list.append([id, cx, cy])

                if len(lm_list) != 0:
                    x1, y1 = lm_list[8][1], lm_list[8][2]
                    index_up = lm_list[8][2] < lm_list[6][2]
                    middle_up = lm_list[12][2] < lm_list[10][2]

                    # Selection Mode (Dono fingers up)
                    if index_up and middle_up:
                        self.xp, self.yp = 0, 0
                        if y1 < 80:
                            if 0 <= x1 < box_w: self.draw_color = self.colors[0]
                            elif box_w <= x1 < box_w*2: self.draw_color = self.colors[1]
                            elif box_w*2 <= x1 < box_w*3: self.draw_color = self.colors[2]
                            elif box_w*3 <= x1 <= w: self.draw_color = self.colors[3]

                    # Drawing Mode (Sirf index finger up)
                    elif index_up and not middle_up:
                        cv2.circle(img, (x1, y1), 12, self.draw_color, cv2.FILLED)
                        if self.xp == 0 and self.yp == 0:
                            self.xp, self.yp = x1, y1

                        if self.draw_color == (0, 0, 0):
                            cv2.line(self.canvas, (self.xp, self.yp), (x1, y1), self.draw_color, self.eraser_thickness)
                        else:
                            cv2.line(self.canvas, (self.xp, self.yp), (x1, y1), self.draw_color, self.brush_thickness)
                        self.xp, self.yp = x1, y1

                mp_draw.draw_landmarks(img, hand_lms, mp_hands.HAND_CONNECTIONS)

        img_gray = cv2.cvtColor(self.canvas, cv2.COLOR_BGR2GRAY)
        _, img_inv = cv2.threshold(img_gray, 50, 255, cv2.THRESH_BINARY_INV)
        img_inv = cv2.cvtColor(img_inv, cv2.COLOR_GRAY2BGR)
        img = cv2.bitwise_and(img, img_inv)
        img = cv2.bitwise_or(img, self.canvas)

        return img

# Streamlit Component to render webcam
webrtc_streamer(key="painter", video_transformer_factory=PainterTransformer)

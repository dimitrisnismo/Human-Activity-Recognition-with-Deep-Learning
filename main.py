import cv2
from matplotlib.pyplot import connect
import mediapipe as mp
from numpy.lib.function_base import select
import pandas as pd
import numpy as np
import pymongo
import os
import tempfile
import os
import time
import streamlit as st
import altair as alt
from Core.functions import (
    output_folder,
    symmetry_func,
    empty_dataframe,
    mediapipe_packs,
    video_metrics,
    Selection_points_default,
    calculate_results,
    data_preparation,
    calculating_loops_of_excercise_right,
    checking_height_of_wrist,
    checking_negative_knee,
    checking_wrist_x_whips,
    stable_body,
    elbow_vs_shoulder,
    wrist_front_of_elbow,
    wrist_vs_elbow,
)
import altair as alt

### TO INCULDE IN THE FINAL GRAPHS LEGEND WHEN THE EXC IS WRONG

# Headers for the Streamlit App
st.title("Deep learning Mediapipe Pose Estimation")
st.subheader("MSc Big Data & Analytics")
st.text("Dimitris Chortarias")

# Points of the body
points = pd.read_pickle("points.pkl")
select_point = [
    "NONE",
    "LEFT_SHOULDER",
    "RIGHT_SHOULDER",
    "LEFT_ELBOW",
    "RIGHT_ELBOW",
    "LEFT_WRIST",
    "RIGHT_WRIST",
    "LEFT_HIP",
    "RIGHT_HIP",
    "LEFT_KNEE",
    "RIGHT_KNEE",
]

# Uploading the file
uploaded_file = st.file_uploader("Import video file")
if uploaded_file is None:
    st.stop()
else:
    tfile = tempfile.NamedTemporaryFile(delete=True)
    tfile.write(uploaded_file.read())
    vidcap = cv2.VideoCapture(tfile.name)

# uploaded_file = r"C:\Users\Dimit\OneDrive\Videos\  outshoulderturnswithdumbbells.mp4"
# vidcap = cv2.VideoCapture(uploaded_file)
# Options
write_data_to_mongo = False

df = empty_dataframe()

## File
file_name = uploaded_file.name
# file_name = "outshoulderturnswithdumbbells.mp4"
path = output_folder(file_name)


## Loading packages of mediapipe
mp_drawing, mp_pose, mp_holistic = mediapipe_packs()

# Video Metrics & Creating the new video


(
    success,
    image_height,
    image_width,
    total_frames,
    new_width,
    new_height,
    fps,
    video,
) = video_metrics(vidcap, file_name, path)


# Selections Points
excercises = [
    "Please Choose",
    "Οut shoulder turns with dumbbells",
    "Ηand stretches with dumbbells",
    "lifting dumbbells in an upright position",
]
select_point_new = st.selectbox("Select gym exercise ", excercises, key=2910)
if select_point_new in excercises[1:]:
    select_x=select_point_new
else:
    st.warning("Please select excercise.")
    st.stop()
# select_x = "Οut shoulder turns with dumbbells"

select_point_one = st.selectbox("Select Point 1", select_point, key=1)
select_point_two = st.selectbox("Select Point 2", select_point, key=2)
select_point_three = st.selectbox("Select Point 3", select_point, key=3)
select_point_four = st.selectbox("Select Point 4", select_point, key=4)
select_point_five = st.selectbox("Select Point 5", select_point, key=5)
(
    select_one,
    select_two,
    select_three,
    select_four,
    select_five,
) = Selection_points_default(
    points,
    select_point_one,
    select_point_two,
    select_point_three,
    select_point_four,
    select_point_five,
)


# Check to be completed the Selection Points
if select_point_five == "NONE":
    st.warning("Please select all options.")
    st.stop()

# Creating the progress bar
st.text("Deep Learning Model is Calculating the Points of the body... Keep Calm! :) ")
progress_bar = st.progress(0)

# Applying Deep Learning model for each dataframe
frames = 1


with mp_pose.Pose(
    static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5
) as pose:
    # creating a temporary dataframe
    tempdf = pd.DataFrame(index=np.arange(1))
    # Reading each frame
    while success:
        success, image = vidcap.read()
        # Checking if there is another frame in order to break at the last one
        try:
            len(image)
        except:
            break
        # Returning the results of pose
        results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        # creating an image
        annotated_image = image.copy()
        # drawing in the image the points
        mp_drawing.draw_landmarks(
            annotated_image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS
        )
        # shape of the image
        height, width, layers = annotated_image.shape
        # reading the results
        for i in mp_holistic.PoseLandmark:
            df = calculate_results(
                df, file_name, image_height, image_width, frames, tempdf, results, i
            )
        frames = frames + 1
        # Updating Progress Bar
        progress_bar.progress(frames / total_frames)
        # Resize image
        resized = cv2.resize(
            annotated_image, (new_width, new_height), interpolation=cv2.INTER_AREA
        )
        # Write resized image into video
        video.write(resized)

# Releasing the video results
video.release()

# Print that deep learning video has been completed
st.success("Deep Learning Model finished the job :) ")

# Reading the created video with the points
created_video = cv2.VideoCapture(
    path + "/" + file_name.replace(".mp4", "") + "_mediapipe.mp4"
)

##Calculating data transforms
df, s_df = data_preparation(points, df, file_name, path)


##Formulas for the metrics
dfloops = calculating_loops_of_excercise_right(df)
kneeposition = checking_negative_knee(df)
stable_body = stable_body(df)

if select_x == "Οut shoulder turns with dumbbells":
    right_elbow_vs_shoulder = elbow_vs_shoulder(df, "RIGHT")
    left_elbow_vs_shoulder = elbow_vs_shoulder(df, "LEFT")
    right_wrist_vs_elbow = wrist_front_of_elbow(df, "RIGHT")
    left_wrist_vs_elbow = wrist_front_of_elbow(df, "LEFT")
    right_wrist_x_elbow = wrist_vs_elbow(df, "RIGHT")
    left_wrist_x_elbow = wrist_vs_elbow(df, "LEFT")
    results_1 = df[["frame"]].drop_duplicates()
    results_1 = pd.merge(
        results_1,
        right_elbow_vs_shoulder.rename(
            columns={"final_flag": "right_elbow_vs_shoulder"}
        ),
        on="frame",
        how="left",
    )
    results_1 = pd.merge(
        results_1,
        left_elbow_vs_shoulder.rename(columns={"final_flag": "left_elbow_vs_shoulder"}),
        on="frame",
        how="left",
    )
    results_1 = pd.merge(
        results_1,
        right_wrist_vs_elbow.rename(columns={"final_flag": "right_wrist_vs_elbow"}),
        on="frame",
        how="left",
    )
    results_1 = pd.merge(
        results_1,
        left_wrist_vs_elbow.rename(columns={"final_flag": "left_wrist_vs_elbow"}),
        on="frame",
        how="left",
    )
    results_1 = pd.merge(
        results_1,
        right_wrist_x_elbow.rename(columns={"final_flag": "right_wrist_x_elbow"}),
        on="frame",
        how="left",
    )
    results_1 = pd.merge(
        results_1,
        left_wrist_x_elbow.rename(columns={"final_flag": "left_wrist_x_elbow"}),
        on="frame",
        how="left",
    )
    results_1 = pd.merge(
        results_1,
        stable_body.rename(columns={"final_flag": "stable_body"}),
        on="frame",
        how="left",
    )
    results_1 = pd.merge(
        results_1, s_df.rename(columns={"s_flag": "symmetry"}), on="frame", how="left"
    )
    results_1["Check"] = np.where(
        (results_1["right_elbow_vs_shoulder"] == "Elbow Position is ok")
        & (results_1["left_elbow_vs_shoulder"] == "Elbow Position is ok")
        & (results_1["right_wrist_vs_elbow"] == "Wrist Position is ok")
        & (results_1["left_wrist_vs_elbow"] == "Wrist Position is ok")
        & (results_1["right_wrist_x_elbow"] == "Wrist Position is ok")
        & (results_1["left_wrist_x_elbow"] == "Wrist Position is ok")
        & (results_1["stable_body"] == "Your body is stable")
        & (results_1["symmetry"] == "Ok"),
        "OK",
        "NOT",
    )
elif select_x == "lifting dumbbells in an upright position":
    wrist_y_right = checking_height_of_wrist(df, "RIGHT")
    wrist_y_left = checking_height_of_wrist(df, "LEFT")
    wrist_x_right = checking_wrist_x_whips(df, "RIGHT")
    wrist_x_left = checking_wrist_x_whips(df, "LEFT")
    results_1 = df[["frame"]].drop_duplicates()
    results_1 = pd.merge(
        results_1,
        wrist_y_right.rename(
            columns={"final_flag": "wrist_y_right"}
            # Wrist Position is ok
        ),
        on="frame",
        how="left",
    )
    results_1 = pd.merge(
        results_1,
        wrist_y_left.rename(
            columns={"final_flag": "wrist_y_left"}
            # Wrist Position is ok
        ),
        on="frame",
        how="left",
    )
    results_1 = pd.merge(
        results_1,
        wrist_x_right.rename(
            columns={"final_flag": "wrist_x_right"}
            # Wrist Position is ok
        ),
        on="frame",
        how="left",
    )
    results_1 = pd.merge(
        results_1,
        wrist_x_left.rename(
            columns={"final_flag": "wrist_x_left"}
            # Wrist Position is ok
        ),
        on="frame",
        how="left",
    )
    results_1 = pd.merge(
        results_1,
        stable_body.rename(columns={"final_flag": "stable_body"}),
        on="frame",
        how="left",
    )
    results_1 = pd.merge(
        results_1, s_df.rename(columns={"s_flag": "symmetry"}), on="frame", how="left"
    )
    results_1["Check"] = np.where(
        (results_1["wrist_y_right"] == "Wrist Position is ok")
        & (results_1["wrist_y_left"] == "Wrist Position is ok")
        & (results_1["wrist_x_right"] == "Wrist Position is ok")
        & (results_1["wrist_x_left"] == "Wrist Position is ok")
        & (results_1["stable_body"] == "Your body is stable")
        & (results_1["symmetry"] == "Ok"),
        "OK",
        "NOT",
    )
else:
    results_1 = df[["frame"]].drop_duplicates()
    results_1 = pd.merge(
        results_1,
        stable_body.rename(columns={"final_flag": "stable_body"}),
        on="frame",
        how="left",
    )
    results_1 = pd.merge(
        results_1, s_df.rename(columns={"s_flag": "symmetry"}), on="frame", how="left"
    )
    results_1["Check"] = np.where(
        (results_1["stable_body"] == "Your body is stable")
        & (results_1["symmetry"] == "Ok"),
        "OK",
        "NOT",
    )
results_1 = results_1[["frame", "Check"]]
# Begining the code to appear results
streamimage = st.empty()
text_1 = st.empty()
text_2 = st.empty()
text_3 = st.empty()
text_4 = st.empty()
text_5 = st.empty()
text_6 = st.empty()
text_7 = st.empty()
text_8 = st.empty()
text_9 = st.empty()
text_10 = st.empty()
text_11 = st.empty()
text_12 = st.empty()
text_13 = st.empty()
text_14 = st.empty()
text_15 = st.empty()
text_16 = st.empty()


# Print video and selected points
frames = 1
while True:
    # Read image of the video
    success, image = created_video.read()
    # Checking that image exists
    if not success:
        break
    # printing image
    streamimage.image(image, channels="BGR")
    # Printing Flag Symmetry
    try:
        text_1.text("Symmetry is " + s_df[s_df["frame"] == frames]["s_flag"].values[0])
    except:
        pass
    try:
        text_4.text(
            "Total Loops:" + str(dfloops[dfloops["frame"] == frames]["Loops"].values[0])
        )
    except:
        pass
    try:
        text_5.text(
            "height of right wrist:"
            + wrist_y_right[wrist_y_right["frame"] == frames]["final_flag"].values[0]
        )
    except:
        pass
    try:
        text_6.text(
            "height of left wrist:"
            + wrist_y_left[wrist_y_left["frame"] == frames]["final_flag"].values[0]
        )
    except:
        pass
    try:
        text_7.text(
            "knee position:"
            + kneeposition[kneeposition["frame"] == frames]["final_flag"].values[0]
        )
    except:
        pass
    try:
        text_8.text(
            "Stable Body: "
            + str(stable_body[stable_body["frame"] == frames]["final_flag"].values[0])
        )
    except:
        pass
    try:
        text_9.text(
            "right wrist based on hips:"
            + str(
                wrist_x_right[wrist_x_right["frame"] == frames]["final_flag"].values[0]
            )
        )
    except:
        pass
    try:
        text_10.text(
            "left wrist based on hips:"
            + str(wrist_x_left[wrist_x_left["frame"] == frames]["final_flag"].values[0])
        )
    except:
        pass
    try:
        text_11.text(
            "right wrist based on elbow:"
            + str(
                right_wrist_vs_elbow[right_wrist_vs_elbow["frame"] == frames][
                    "final_flag"
                ].values[0]
            )
        )
    except:
        pass
    try:
        text_12.text(
            "left wrist based on elbow:"
            + str(
                left_wrist_vs_elbow[left_wrist_vs_elbow["frame"] == frames][
                    "final_flag"
                ].values[0]
            )
        )
    except:
        pass
    try:
        text_13.text(
            "right shoulder based on elbow:"
            + str(
                right_elbow_vs_shoulder[right_elbow_vs_shoulder["frame"] == frames][
                    "final_flag"
                ].values[0]
            )
        )
    except:
        pass
    try:
        text_14.text(
            "left shoulder based on elbow:"
            + str(
                left_elbow_vs_shoulder[left_elbow_vs_shoulder["frame"] == frames][
                    "final_flag"
                ].values[0]
            )
        )
    except:
        pass
    try:
        text_15.text(
            "right wrist based on elbow:"
            + str(
                right_wrist_x_elbow[right_wrist_x_elbow["frame"] == frames][
                    "final_flag"
                ].values[0]
            )
        )
    except:
        pass
    try:
        text_16.text(
            "left wrist based on elbow:"
            + str(
                left_wrist_x_elbow[left_wrist_x_elbow["frame"] == frames][
                    "final_flag"
                ].values[0]
            )
        )
    except:
        pass
    # printing the selected points
    text_2.text("Selection Points")
    text_3.text(
        df[
            (
                df["Point"].isin(
                    [select_one, select_two, select_three, select_four, select_five]
                )
            )
            & (df["frame"] == frames)
        ][["DESCRIPTION", "x", "y", "z", "x_center", "y_center"]].to_string(index=False)
    )
    # applying a delay in order to match the fps of the video
    time.sleep(1 / fps)
    frames = frames + 1

st.header("Totals")
st.text(
    "The "
    + str(
        round(
            results_1[results_1["Check"] == "OK"]["Check"].count()
            / results_1["Check"].count(),
            3,
        )
        * 100
    )
 + "% of the workout was accurrate.")
##Totals
##Total Symmetry in the body
s_df = s_df.groupby(["s_flag"]).count().reset_index()
st.text(
    "There is  "
    + str(
        round(s_df[s_df["s_flag"] == "Ok"]["frame"].sum() / s_df.frame.sum(), 3) * 100
    )
    + "%  Total upper body symmetry between left and right. "
)
c = (
    alt.Chart(s_df)
    .mark_bar()
    .encode(x=alt.X("sum(frame)", stack="normalize"), color="s_flag")
)
# st.altair_chart(c, use_container_width=True)

##Total loops
# dfloops = dfloops.groupby(["s_flag"]).count().reset_index()
st.text("There have been  " + str(dfloops["Loops"].max()) + "  Total Loops ")
try:
    ##Total right wrist height
    wrist_y_right = wrist_y_right.groupby(["final_flag"]).count().reset_index()
    st.text(
        "There is  "
        + str(
            round(
                wrist_y_right[wrist_y_right["final_flag"] == "Wrist Position is ok"][
                    "frame"
                ].sum()
                / wrist_y_right.frame.sum(),
                3,
            )
            * 100
        )
        + "% right wrist height was right. "
    )
except:
    pass
try:

    ##Total left wrist height
    wrist_y_left = wrist_y_left.groupby(["final_flag"]).count().reset_index()
    st.text(
        "There is  "
        + str(
            round(
                wrist_y_left[wrist_y_left["final_flag"] == "Wrist Position is ok"][
                    "frame"
                ].sum()
                / wrist_y_left.frame.sum(),
                3,
            )
            * 100
        )
        + "% left wrist height was right. "
    )
except:
    pass
try:
    ##Total knees position
    kneeposition = kneeposition.groupby(["final_flag"]).count().reset_index()
    st.text(
        "There is  "
        + str(
            round(
                kneeposition[kneeposition["final_flag"] == "Knees Position are ok"][
                    "frame"
                ].sum()
                / kneeposition.frame.sum(),
                3,
            )
            * 100
        )
        + "% knees position was right. "
    )
except:
    pass
try:

    ##Total stable_body
    stable_body = stable_body.groupby(["final_flag"]).count().reset_index()
    st.text(
        "There is  "
        + str(
            round(
                stable_body[stable_body["final_flag"] == "Your body is stable"][
                    "frame"
                ].sum()
                / stable_body.frame.sum(),
                3,
            )
            * 100
        )
        + "% stable body. "
    )
except:
    pass
try:
    ##Total wrist_x_right
    wrist_x_right = wrist_x_right.groupby(["final_flag"]).count().reset_index()
    st.text(
        "There is  "
        + str(
            round(
                wrist_x_right[wrist_x_right["final_flag"] == "Wrist Position is ok"][
                    "frame"
                ].sum()
                / wrist_x_right.frame.sum(),
                3,
            )
            * 100
        )
        + "% right of right wrist based on hips. "
    )
except:
    pass
try:
    ##Total wrist_x_left
    wrist_x_left = wrist_x_left.groupby(["final_flag"]).count().reset_index()
    st.text(
        "There is  "
        + str(
            round(
                wrist_x_left[wrist_x_left["final_flag"] == "Wrist Position is ok"][
                    "frame"
                ].sum()
                / wrist_x_left.frame.sum(),
                3,
            )
            * 100
        )
        + "% right of right wrist based on hips. "
    )
except:
    pass
try:
    ##Total right_wrist_vs_elbow
    right_wrist_vs_elbow = (
        right_wrist_vs_elbow.groupby(["final_flag"]).count().reset_index()
    )
    st.text(
        "There is  "
        + str(
            round(
                right_wrist_vs_elbow[
                    right_wrist_vs_elbow["final_flag"] == "Wrist Position is ok"
                ]["frame"].sum()
                / right_wrist_vs_elbow.frame.sum(),
                3,
            )
            * 100
        )
        + "% of right wrist front of the elbow. "
    )
except:
    pass
try:
    ##Total left_wrist_vs_elbow
    left_wrist_vs_elbow = (
        left_wrist_vs_elbow.groupby(["final_flag"]).count().reset_index()
    )
    st.text(
        "There is  "
        + str(
            round(
                left_wrist_vs_elbow[
                    left_wrist_vs_elbow["final_flag"] == "Wrist Position is ok"
                ]["frame"].sum()
                / left_wrist_vs_elbow.frame.sum(),
                3,
            )
            * 100
        )
        + "% of left wrist front of the elbow. "
    )
except:
    pass
try:
    ##Total right_elbow_vs_shoulder
    right_elbow_vs_shoulder = (
        right_elbow_vs_shoulder.groupby(["final_flag"]).count().reset_index()
    )
    st.text(
        "There is  "
        + str(
            round(
                right_elbow_vs_shoulder[
                    right_elbow_vs_shoulder["final_flag"] == "Elbow Position is ok"
                ]["frame"].sum()
                / right_elbow_vs_shoulder.frame.sum(),
                3,
            )
            * 100
        )
        + "%  of left elbow in front of your shoulder. "
    )
except:
    pass
try:
    ##Total left_elbow_vs_shoulder
    left_wrist_vs_elbow = (
        left_wrist_vs_elbow.groupby(["final_flag"]).count().reset_index()
    )
    st.text(
        "There is  "
        + str(
            round(
                left_wrist_vs_elbow[
                    left_wrist_vs_elbow["final_flag"] == "Wrist Position is ok"
                ]["frame"].sum()
                / left_wrist_vs_elbow.frame.sum(),
                3,
            )
            * 100
        )
        + "%  of right elbow in front of your shoulder. "
    )
except:
    pass

try:
    ##Total right_wrist_vs_elbow
    right_wrist_x_elbow = (
        right_wrist_x_elbow.groupby(["final_flag"]).count().reset_index()
    )
    st.text(
        "There is  "
        + str(
            round(
                right_wrist_x_elbow[
                    right_wrist_vs_elbow["final_flag"] == "Wrist Position is ok"
                ]["frame"].sum()
                / right_wrist_x_elbow.frame.sum(),
                3,
            )
            * 100
        )
        + "%  of right  wrist in the same straight as your elbow. "
    )
except:
    pass
try:
    ##Total left_elbow_vs_shoulder
    left_wrist_x_elbow = (
        left_wrist_x_elbow.groupby(["final_flag"]).count().reset_index()
    )
    st.text(
        "There is  "
        + str(
            round(
                left_wrist_x_elbow[
                    left_wrist_x_elbow["final_flag"] == "Elbow Position is ok"
                ]["frame"].sum()
                / left_wrist_x_elbow.frame.sum(),
                3,
            )
            * 100
        )
        + "%  of left wrist in the same straight as your elbow."
    )
except:
    pass
# Data formats for the graphs

lop = ["SHOULDER", "ELBOW", "WRIST", "HIP"]
for x in lop:
    dftemp = df[df["SPoint"] == x]
    st.header(x)
    chart1 = (
        alt.Chart(dftemp[["frame", "x_center", "RL"]])
        .mark_line()
        .encode(x="frame", y="x_center", color="RL")
    )
    chart2 = (
        alt.Chart(dftemp[["frame", "y_center", "RL"]])
        .mark_line()
        .encode(x="frame", y="y_center", color="RL")
    )
    st.altair_chart(chart1 | chart2)
results_1['val']=1

chart3 = (
        alt.Chart(results_1[["frame", "val", "Check"]])
        .mark_line()
        .encode(x="frame", y="val", color="Check")
    )
# st.altair_chart(chart3 )
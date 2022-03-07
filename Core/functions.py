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


def symmetry_func(df, x1, y1, name):
    ## Calculating symmetry
    x2 = x1 + 1
    y2 = y1 + 1
    sym_1 = df[df["POINT"].isin([x1, x2])]
    sym_1["POINT"] = sym_1["POINT"] + (y1 - x1)
    sym_1 = pd.merge(
        sym_1[["x", "y", "z", "frame", "POINT"]],
        df[df["POINT"].isin([y1, y2])],
        on=["frame", "POINT"],
        suffixes=("_1", "_2"),
        how="inner",
    )
    sym_1["Distance"] = np.sqrt(
        np.power(sym_1["x_2"] - sym_1["x_1"], 2)
        + np.power(sym_1["y_2"] - sym_1["y_1"], 2)
        + np.power(sym_1["z_2"] - sym_1["z_1"], 2) * 1.0
    )
    sym_1 = sym_1.pivot_table(
        values="Distance", index="frame", columns="RL"
    ).reset_index()
    sym_1["dif"] = np.abs(sym_1["LEFT"] / sym_1["RIGHT"]) - 1
    sym_1["flag"] = sym_1["dif"].rolling(10).mean()

    sym_1["Category"] = name
    sym_1 = sym_1[["frame", "Category", "flag"]]
    return sym_1


def empty_dataframe():
    # Creating an empty dataframe in order to append all results
    df = pd.DataFrame()
    df["file_name"] = ""
    df["frame"] = -1
    df["Point"] = -1
    df["x"] = 0.0
    df["y"] = 0.0
    df["z"] = 0.0
    df["visibility"] = 0.0
    return df


def output_folder(file_name):
    # Creating the output folder
    path = "C:\\Users\\Dimit\\OneDrive\\Research/" + file_name + "/"
    access_rights = 0o666
    try:
        os.mkdir(path, access_rights)
    except OSError:
        print("Creation of the directory %s failed" % path)
    else:
        print("Successfully created the directory %s" % path)
    return path


def mediapipe_packs():
    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose
    mp_holistic = mp.solutions.holistic
    return mp_drawing, mp_pose, mp_holistic


def video_metrics(vidcap, file_name, path):
    # Video Metrics & Creating the new video
    success, image = vidcap.read()
    image_height, image_width, _ = image.shape
    total_frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    new_width = 640
    new_height = int(new_width / image_width * image_height)
    fps = vidcap.get(cv2.CAP_PROP_FPS)
    video = cv2.VideoWriter(
        path + "/" + file_name.replace(".mp4", "") + "_mediapipe.mp4",
        cv2.VideoWriter_fourcc(*"MP4V"),
        fps,
        (new_width, new_height),
    )
    return (
        success,
        image_height,
        image_width,
        total_frames,
        new_width,
        new_height,
        fps,
        video,
    )


def Selection_points_default(
    points,
    select_point_one,
    select_point_two,
    select_point_three,
    select_point_four,
    select_point_five,
):
    select_one = points[points["DESCRIPTION"] == select_point_one]["POINT"].sum()
    select_two = points[points["DESCRIPTION"] == select_point_two]["POINT"].sum()
    select_three = points[points["DESCRIPTION"] == select_point_three]["POINT"].sum()
    select_four = points[points["DESCRIPTION"] == select_point_four]["POINT"].sum()
    select_five = points[points["DESCRIPTION"] == select_point_five]["POINT"].sum()
    return select_one, select_two, select_three, select_four, select_five


def calculate_results(
    df, file_name, image_height, image_width, frames, tempdf, results, i
):
    try:
        # to_import = {
        #     "video": file_name,
        #     "frame": frames,
        #     "Point": i,
        #     "x": results.pose_landmarks.landmark[i].x * image_width,
        #     "y": results.pose_landmarks.landmark[i].y * image_height,
        #     "z": results.pose_landmarks.landmark[i].z,
        #     "visibility": results.pose_landmarks.landmark[i].visibility,
        # }
        # Importing the results in the dataframetempdf["file_name"] = file_name
        tempdf["frame"] = frames
        tempdf["Point"] = i
        tempdf["x"] = results.pose_landmarks.landmark[i].x
        tempdf["y"] = results.pose_landmarks.landmark[i].y
        tempdf["z"] = results.pose_landmarks.landmark[i].z
        tempdf["visibility"] = results.pose_landmarks.landmark[i].visibility
        df = df.append(tempdf)
    except:
        pass
    return df


def data_preparation(points, df, file_name, path):
    df = pd.merge(df, points, left_on="Point", right_on="POINT", how="left")
    df_cob = df[df["DESCRIPTION"].isin(["LEFT_HIP", "RIGHT_HIP"])]
    df_cob = df_cob.groupby(["frame"]).agg({"x": np.mean, "y": np.mean}).reset_index()
    df_cob = df_cob.rename(columns={"x": "x_cob", "y": "y_cob"})
    df = pd.merge(df, df_cob, on="frame", how="left")
    df["x_center"] = df["x"] - df["x_cob"]
    df["y_center"] = df["y"] - df["y_cob"]
    df["RL"] = df["DESCRIPTION"].str.split("_").str[0]
    df["SPoint"] = df["DESCRIPTION"].str.split("_").str[1]
    s_shoulder_wrist = symmetry_func(df, 11, 15, "wrist_shoulder")
    s_wrist_hip = symmetry_func(df, 15, 23, "s_wrist_hip")
    s_elbow_hip = symmetry_func(df, 13, 23, "s_elbow_hip")
    s_df = pd.merge(s_shoulder_wrist, s_wrist_hip, on="frame", how="inner")
    s_df = pd.merge(s_df, s_elbow_hip, on="frame", how="inner")
    s_df["s_flag"] = (
        np.abs(s_df["flag_x"]) + np.abs(s_df["flag_y"]) + np.abs(s_df["flag"])
    ) / 3
    s_df["s_flag"] = np.where(s_df["s_flag"] > 0.15, "No Symmetry", "Ok")
    s_df = s_df.dropna()
    s_df.to_excel(path + "/" + file_name.replace(".mp4", "") + "_mediapipe.xlsx")
    s_df = s_df[["frame", "s_flag"]]
    return df, s_df


def checking_negative_knee(df):
    df_knees = df[df["SPoint"].isin(["KNEE", "HIP"])]
    df_knees = df_knees[["frame", "z", "RL", "SPoint"]]
    df_knees["z"] = df_knees["z"].rolling(10).mean()
    df_knees = df_knees.pivot_table(
        values="z",
        index=["frame", "RL"],
        columns=["SPoint"],
        aggfunc="mean",
    ).reset_index()
    df_knees["final_flag"] = np.where(
        df_knees["KNEE"] - df_knees["HIP"] <= 0.05,
        "Knees Position are ok",
        "Bent Knees",
    )
    return df_knees


def checking_height_of_wrist(df, rlcat):
    df_wrist = df[df["SPoint"].isin(["WRIST", "SHOULDER"])]
    df_wrist = df_wrist[["frame", "y_center", "RL", "SPoint"]]
    df_wrist = df_wrist[df_wrist["RL"] == rlcat]
    df_wrist = df_wrist.pivot_table(
        values="y_center",
        index=["frame", "RL"],
        columns=["SPoint"],
        aggfunc="mean",
    ).reset_index()
    df_wrist["SHOULDER"] = df_wrist["SHOULDER"].rolling(10).mean()
    df_wrist["WRIST"] = df_wrist["WRIST"].rolling(10).mean()
    df_wrist["final_flag"] = np.where(
        df_wrist["SHOULDER"] - df_wrist["WRIST"] <= 0.01,
        "Wrist Position is ok",
        "Dont move so high your wrist",
    )
    return df_wrist


def checking_wrist_x_whips(df, rlcat):
    df_wrist_hip = df[df["SPoint"].isin(["WRIST", "HIP"])]
    df_wrist_hip = df_wrist_hip[["frame", "x", "RL", "SPoint"]]
    df_wrist_hip = df_wrist_hip[df_wrist_hip["RL"] == rlcat]
    df_wrist_hip = df_wrist_hip.pivot_table(
        values="x",
        index=["frame", "RL"],
        columns=["SPoint"],
        aggfunc="mean",
    ).reset_index()
    df_wrist_hip["HIP"] = df_wrist_hip["HIP"].rolling(10).mean()
    df_wrist_hip["WRIST"] = df_wrist_hip["WRIST"].rolling(10).mean()
    df_wrist_hip["final_flag"] = np.where(
        (abs(df_wrist_hip["HIP"] - df_wrist_hip["WRIST"]) <= 0.02),
        "Wrist Position is ok",
        "Keep your wrist in the same straight as your hip.",
    )
    return df_wrist_hip


def calculating_loops_of_excercise_right(df):
    # for the right Hand only
    df_loops = df[df["SPoint"].isin(["WRIST"])]
    df_loops = df_loops[df_loops["RL"].isin(["RIGHT"])]
    df_loops = df_loops[["frame", "y"]]
    df_loops["y_rolling"] = df_loops["y"].rolling(25).mean()
    df_loops["y_delta"] = df_loops["y_rolling"] - df_loops["y_rolling"].shift(1)
    df_loops["y_flag_change"] = np.where(
        (df_loops["y_delta"] <= -0.0001),
        "down",
        np.where(
            df_loops["y_delta"] >= 0.0001,
            "up",
            "other",
        ),
    )
    dfloops = (
        df_loops[["frame", "y_flag_change"]]
        .groupby(["frame", "y_flag_change"])
        .count()
        .reset_index()
    )
    dfloops["check"] = np.where(
        dfloops["y_flag_change"] == dfloops["y_flag_change"].shift(1), 1, 0
    )
    dfloops["sum"] = dfloops.groupby("y_flag_change")["check"].apply(
        lambda x: x * (x.groupby((x != x.shift()).cumsum()).cumcount() + 1)
    )
    dfloops["Loops"] = np.where(
        (dfloops["y_flag_change"] == "up")
        & (dfloops["sum"] >= 15)
        & (dfloops["sum"].shift(-1) == 0),
        1,
        0,
    )
    dfloops["Loops"] = dfloops["Loops"].cumsum()

    return dfloops


def stable_body(df):
    df["SPoint"] = df["SPoint"].fillna("NOSE")
    df_stable = df[df["SPoint"].isin(["NOSE", "HIP", "KNEE", "ANKLE"])]
    df_stable = df_stable[["frame", "x", "y", "z", "RL", "SPoint"]]
    df_stable_means = (
        df_stable.groupby(["RL", "SPoint"])
        .mean()
        .reset_index()
        .drop(columns=["frame"])
        .rename(columns={"x": "x_stable", "y": "y_stable", "z": "z_stable"})
    )
    df_stable = pd.merge(
        df_stable,
        df_stable_means,
        on=[
            "RL",
            "SPoint",
        ],
    )
    df_stable["x_delta"] = abs(df_stable["x"] - df_stable["x_stable"])
    df_stable["y_delta"] = abs(df_stable["y"] - df_stable["y_stable"])
    df_stable["z_delta"] = abs(df_stable["z"] - df_stable["z_stable"])
    df_stable["x_flag"] = np.where((df_stable["x_delta"] >= 0.2), 1, 0)
    df_stable["y_flag"] = np.where((df_stable["y_delta"] >= 0.2), 1, 0)
    df_stable["z_flag"] = np.where((df_stable["z_delta"] >= 0.2), 1, 0)
    df_stable["s_flag"] = (
        df_stable["x_flag"] + df_stable["z_flag"] + df_stable["y_flag"]
    )
    df_stable["final_flag"] = np.where(
        df_stable["s_flag"] > 1, "Keep your rest body stable", "Your body is stable"
    )
    return df_stable


def wrist_front_of_elbow(df, rlcat):
    df_wrist_hip = df[df["SPoint"].isin(["WRIST", "ELBOW"])]
    df_wrist_hip = df_wrist_hip[["frame", "z", "RL", "SPoint"]]
    df_wrist_hip = df_wrist_hip[df_wrist_hip["RL"] == rlcat]
    df_wrist_hip = df_wrist_hip.pivot_table(
        values="z",
        index=["frame", "RL"],
        columns=["SPoint"],
        aggfunc="mean",
    ).reset_index()
    df_wrist_hip["ELBOW"] = df_wrist_hip["ELBOW"].rolling(10).mean()
    df_wrist_hip["WRIST"] = df_wrist_hip["WRIST"].rolling(10).mean()
    df_wrist_hip["final_flag"] = np.where(
        (df_wrist_hip["ELBOW"]+0.05 >= df_wrist_hip["WRIST"]) ,
        "Wrist Position is ok",
        "Keep your wrist in front of your elbow",
    )
    return df_wrist_hip


def elbow_vs_shoulder(df, rlcat):
    df_wrist_hip = df[df["SPoint"].isin(["ELBOW", "SHOULDER"])]
    df_wrist_hip = df_wrist_hip[["frame", "y", "RL", "SPoint"]]
    df_wrist_hip = df_wrist_hip[df_wrist_hip["RL"] == rlcat]
    df_wrist_hip = df_wrist_hip.pivot_table(
        values="y",
        index=["frame", "RL"],
        columns=["SPoint"],
        aggfunc="mean",
    ).reset_index()
    df_wrist_hip["SHOULDER"] = df_wrist_hip["SHOULDER"].rolling(10).mean()
    df_wrist_hip["ELBOW"] = df_wrist_hip["ELBOW"].rolling(10).mean()
    df_wrist_hip["final_flag"] = np.where(
        abs(df_wrist_hip["ELBOW"] - df_wrist_hip["SHOULDER"]) >= 0.02,
        "Elbow Position is ok",
        "Keep your elbow in front of your Shoulder",
    )
    return df_wrist_hip


def wrist_vs_elbow(df, rlcat):
    df_wrist_hip = df[df["SPoint"].isin(["ELBOW", "WRIST"])]
    df_wrist_hip = df_wrist_hip[["frame", "x", "RL", "SPoint"]]
    df_wrist_hip = df_wrist_hip[df_wrist_hip["RL"] == rlcat]
    df_wrist_hip = df_wrist_hip.pivot_table(
        values="x",
        index=["frame", "RL"],
        columns=["SPoint"],
        aggfunc="mean",
    ).reset_index()
    df_wrist_hip["WRIST"] = df_wrist_hip["WRIST"].rolling(10).mean()
    df_wrist_hip["ELBOW"] = df_wrist_hip["ELBOW"].rolling(10).mean()
    df_wrist_hip["final_flag"] = np.where(
        abs(df_wrist_hip["ELBOW"] - df_wrist_hip["WRIST"]) <= 0.09,
        "Wrist Position is ok",
        "Keep your wrist in the same straight as your elbow",
    )
    return df_wrist_hip

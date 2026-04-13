# CVA Posture Analysis Dashboard

A data analysis dashboard built on [BoogiBoogi](https://github.com/kong33/forward_head_posture_detect_application).<br/>
Raw CVA measurements from the app contained noise from camera misdetection and sudden head movements, <br/>
so I built a 3 stage filtering pipeline to clean the data and visualize the results.

## Pipeline

**Stage 1 | Presence filter**: drops frames where the camera failed to detect a person (`hasPose: false`)

**Stage 2 | IQR outlier removal**: removes angle values that fall outside 1.5× the interquartile range — catches sudden spikes from recognition errors

**Stage 3 | Continuity filter**: removes samples where the angle changes too sharply between consecutive frames, which usually indicates a transient misread rather than a real posture shift

## Dashboard

Live demo → [https://boogiboogi-app-data.streamlit.app](https://boogiboogi-app-data.streamlit.app/)

- Before/after comparison of angle distributions
- Per-stage noise removal breakdown
- Daily CVA trend with turtle-neck threshold overlay
- Session-level summary table (avg angle, turtle alert count, turtle ratio)

## Stack

Python, pandas, matplotlib, Streamlit

## Data

Sensor data exported from IndexedDB (browser-side storage) of the posture monitoring app.<br/>
Timestamps stored and displayed as Unix milliseconds, consistent with the original data format.

## Furthermore 
We are planning to add this filtering logic into our application code.

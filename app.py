import streamlit as st
import pandas as pd
import time
from datetime import datetime
import os

ts = time.time()
date = datetime.fromtimestamp(ts).strftime("%d-%m-%Y")
timestamp = datetime.fromtimestamp(ts).strftime("%H:%M-%S")

from streamlit_autorefresh import st_autorefresh

count = st_autorefresh(interval=2000, limit=100, key="fizzbuzzcounter")

if count == 0:
    st.write("Count is zero")
elif count % 3 == 0 and count % 5 == 0:
    st.write("FizzBuzz")
elif count % 3 == 0:
    st.write("Fizz")
elif count % 5 == 0:
    st.write("Buzz")
else:
    st.write(f"Count: {count}")

attendance_file = os.path.join("Attendance", f"Attendance_{date}.csv")
if not os.path.exists(attendance_file):
    st.warning(f"No attendance file for today: {attendance_file}")
    st.write("Run the recognition script and take attendance (press 'o') to generate the CSV.")
else:
    try:
        df = pd.read_csv(attendance_file)
        st.dataframe(df.style.highlight_max(axis=0))
    except Exception as e:
        st.error(f"Failed to read attendance CSV: {e}")
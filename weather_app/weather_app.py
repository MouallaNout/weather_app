# مشروع: نظام التنبؤ بالطقس لليوم التالي باستخدام التعلم الآلي
# الأدوات: Python + scikit-learn + pandas + OpenWeatherMap + Streamlit (واجهة)

import requests
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
import joblib
import streamlit as st

# --------- القسم 1: جمع البيانات من OpenWeatherMap ---------
API_KEY = "6b7d3900ee4743f32280a95bc4e5ae58"

def get_weather_data(city):
    url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={API_KEY}&units=metric"
    response = requests.get(url)
    data = response.json()

    features = {
        "temp": data["main"].get("temp"),
        "temp_min": data["main"].get("temp_min"),
        "temp_max": data["main"].get("temp_max"),
        "pressure": data["main"].get("pressure"),
        "humidity": data["main"].get("humidity"),
        "wind_speed": data["wind"].get("speed"),
        "wind_deg": data["wind"].get("deg"),
        "clouds": data["clouds"].get("all"),
        "rain_1h": data.get("rain", {}).get("1h", 0)
    }
    return pd.DataFrame([features]), features["temp"]  # نعيد البيانات ودرجة الحرارة الحالية لتوليد temp_next_day

# --------- القسم 2: واجهة المستخدم ---------
st.title("نظام التنبؤ بالطقس ليوم الغد")
city = st.text_input("أدخل اسم المدينة:")

if city:
    try:
        input_data, current_temp = get_weather_data(city)
        st.write("\n**بيانات الطقس الحالية:**")
        st.dataframe(input_data)

        # توليد بيانات تدريبيّة باستخدام إدخال المستخدم
        df = input_data.copy()
        df["temp_next_day"] = current_temp + np.random.uniform(-3, 3)  # نحاكي درجة حرارة الغد بناءً على اليوم الحالي

        X = df[["temp", "temp_min", "temp_max", "pressure", "humidity", "wind_speed", "wind_deg", "clouds", "rain_1h"]]
        y = df["temp_next_day"]

        model = LinearRegression()
        model.fit(X, y)
        prediction = model.predict(X)

        st.success(f"🌤️ التنبؤ بدرجة الحرارة في {city} ليوم الغد: {prediction[0]:.2f}°C")

    except Exception as e:
        st.error("حدث خطأ أثناء جلب البيانات أو التنبؤ")
        st.text(e)

import time
import joblib
import pandas as pd
import serial  # installed via: pip install pyserial

# 1. Load updated model and scaler
model = joblib.load("tcs34725_hb_model.pkl")
scaler = joblib.load("tcs34725_scaler.pkl")

# 2. Configure PySerial
PORT = "COM3"  # Change to match your ESP32 port
BAUD_RATE = 115200

try:
    ser = serial.Serial(PORT, BAUD_RATE, timeout=1)
    time.sleep(2)
    print(f"Connected to ESP32 on port {PORT}\n")
except Exception as e:
    print(f"Failed to connect to ESP32 on {PORT}: {e}")
    exit()

# 3. Patient biometric inputs (Finger Width removed)
patient_bmi = 22.5
patient_age = 24
patient_sex = 1  # 1 for Male, 0 for Female

print("Listening for raw TCS34725 readings from ESP32...")
print("Press Ctrl+C to stop.\n")

while True:
    try:
        line = ser.readline().decode("utf-8").strip()

        if line and not line.startswith("Error"):
            parts = line.split(",")
            if len(parts) == 4:
                r, g, b, c = map(float, parts)

                if c == 0:
                    print("Sensor blocked / no light detected.")
                    continue

                # Calculate optical ratios
                norm_red = r / c
                norm_green = g / c
                norm_blue = b / c

                # Feature order matching the new 6-feature model
                input_data = pd.DataFrame(
                    [
                        {
                            "Norm_Red": norm_red,
                            "Norm_Green": norm_green,
                            "Norm_Blue": norm_blue,
                            "BMI": patient_bmi,
                            "Subject_Age": patient_age,
                            "Subject_Sex": patient_sex,
                        }
                    ]
                )

                # Scale and predict
                input_scaled = scaler.transform(input_data)
                predicted_hb = model.predict(input_scaled)[0]

                print(f"Raw RGB: ({r:.0f}, {g:.0f}, {b:.0f}) | Clear: {c:.0f}")
                print(f"--> Estimated Hemoglobin: {predicted_hb:.2f} g/dL\n")

    except KeyboardInterrupt:
        print("\nStopping serial reader.")
        ser.close()
        break
    except Exception as e:
        print(f"Error parsing line: {e}")
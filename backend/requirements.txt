# backend/utils/alerts.py
import smtplib
from email.mime.text import MIMEText
import os

def send_email_alert(subject: str, body: str, to_email: str):
    smtp_server = os.getenv("SMTP_SERVER", "smtp.example.com")
    smtp_port = int(os.getenv("SMTP_PORT", "587"))
    from_email = os.getenv("ALERT_FROM_EMAIL", "alert@example.com")
    password = os.getenv("ALERT_EMAIL_PASSWORD", "password")
    
    msg = MIMEText(body)
    msg["Subject"] = subject
    msg["From"] = from_email
    msg["To"] = to_email

    with smtplib.SMTP(smtp_server, smtp_port) as server:
        server.starttls()
        server.login(from_email, password)
        server.send_message(msg)
    print("Alert email sent.")

def check_for_alerts(game_data: dict):
    threshold = 20
    for game in game_data.get("response", []):
        home_score = game.get("scores", {}).get("home", {}).get("total", 0)
        away_score = game.get("scores", {}).get("away", {}).get("total", 0)
        diff = abs(home_score - away_score)
        if diff >= threshold:
            subject = "High Score Difference Alert"
            body = f"Game ID {game.get('id')} has a score difference of {diff}."
            send_email_alert(subject, body, os.getenv("ALERT_TO_EMAIL", "recipient@example.com"))

if __name__ == "__main__":
    dummy_data = {
        "response": [
            {
                "id": 1912,
                "scores": {"home": {"total": 130}, "away": {"total": 100}}
            }
        ]
    }
    check_for_alerts(dummy_data)

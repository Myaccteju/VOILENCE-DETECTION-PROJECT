from email.mime.application import MIMEApplication
import os
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.image import MIMEImage
from email.mime.base import MIMEBase
from email import encoders
from email.message import EmailMessage
import ssl
import smtplib

def send_email(receiver_email, subject, body, attachment_folder):
    sender_email = "tejashreebmestry@gmail.com"  
    sender_password = "bgwm tgtn vlxk xssy"
    message = MIMEMultipart()
    message["From"] = sender_email
    message["To"] = receiver_email
    message["Subject"] = subject
    message.attach(MIMEText(body, "plain"))

    for filename in os.listdir(attachment_folder):
        file_path = os.path.join(attachment_folder, filename)
        if os.path.isfile(file_path):
            with open(file_path, "rb") as file:
                part = MIMEApplication(file.read(), Name=os.path.basename(file_path))
                part['Content-Disposition'] = f'attachment; filename="{os.path.basename(file_path)}"'
                message.attach(part)

    with smtplib.SMTP("smtp.gmail.com", 587) as server:
        server.starttls()
        server.login(sender_email, sender_password)
        server.sendmail(sender_email, receiver_email, message.as_string())

# Example usage
receiver_email = "pooja.mestry67@gmail.com"
subject = "**VIOLENCE ALERT**"
body = "Please find attached screenshots of the incident."
attachment_folder = "punch_screenshots"  

send_email(receiver_email, subject, body, attachment_folder)
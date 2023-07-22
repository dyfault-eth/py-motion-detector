import cv2
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
import ssl
import os
from dotenv import load_dotenv

load_dotenv()


def send_email(subject, body, image_filename=None):
    user = os.getenv('user')
    password = os.getenv('password')

    recipients = os.getenv('dest')

    sender = user

    msg = MIMEMultipart()
    msg['From'] = sender
    msg['To'] = recipients
    msg['Subject'] = subject
    msg.attach(MIMEText(body, 'plain'))

    # attach img in email
    if image_filename:
        with open(image_filename, 'rb') as image_file:
            image_part = MIMEImage(image_file.read(), name=image_filename)
            msg.attach(image_part)

    context = ssl.create_default_context()

    # send e-mail
    with smtplib.SMTP_SSL('smtp.gmail.com', 465, context=context) as smtp:
        smtp.login(user, password)
        smtp.sendmail(user, recipients, msg.as_string())


def main():
    # open webcam (use another number if you're using extern webcam)
    cap = cv2.VideoCapture(0)

    # check if cam is open
    if not cap.isOpened():
        print("The camera cannot be opened.")
        return

    # Capture the first frame to use it as a reference.
    ret, frame1 = cap.read()
    if not ret:
        print("Error while capturing the first frame.")
        cap.release()
        return

    # Conversion of the first frame to grayscale.
    gray_frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)

    bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=1000, varThreshold=30, detectShadows=False)

    # Retrieve the dimensions of the video from the first frame.
    height, width = frame1.shape[:2]

    # Set the codec and create a VideoWriter object to save the video.
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('mouvement_detecte.avi', fourcc, 20.0, (width, height))

    while True:
        # Capture a new frame
        ret, frame2 = cap.read()
        if not ret:
            print("Erreur lors de la capture d'un nouveau cadre.")
            break

        # Conversion of the new frame to grayscale.
        gray_frame2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

        fg_mask = bg_subtractor.apply(frame2)

        # Threshold to obtain a binary mask of the region with the detected movement.
        _, threshold = cv2.threshold(fg_mask, 20, 255, cv2.THRESH_BINARY)

        diff_frame = cv2.absdiff(gray_frame1, gray_frame2)

        # Apply thresholding to obtain a binary mask of the moving areas from the image difference.
        _, diff_threshold = cv2.threshold(diff_frame, 20, 255, cv2.THRESH_BINARY)

        combined_mask = cv2.bitwise_or(threshold, diff_threshold)

        # Noise reduction by applying a blurring operation.
        combined_mask = cv2.GaussianBlur(combined_mask, (5, 5), 0)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        combined_mask = cv2.erode(combined_mask, kernel, iterations=1)
        combined_mask = cv2.dilate(combined_mask, kernel, iterations=1)

        # Detection of contours in the combined binary mask.
        contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Draw the detected contours on the original frame.
        for contour in contours:
            if cv2.contourArea(contour) > 1000:  # Ignore small contours that could be noise.
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(frame2, (x, y), (x + w, y + h), (0, 255, 0), 2)

                # Save the image with the detected movement.
                cv2.imwrite("mouvement_detecte.png", frame2)

                # Send an email with the image attached when movement is detected.
                send_email("Mouvement détecté !", "Un mouvement a été détecté. Veuillez vérifier l'activité.", "mouvement_detecte.png")
                out.write(frame2)

        # Displaying the frame with the detected contours.
        cv2.imshow("Mouvement détecté", frame2)

        # Exit the loop if the 'q' key is pressed.
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # Update the first frame for the next iteration.
        gray_frame1 = gray_frame2

    # Release the camera and close the windows.
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

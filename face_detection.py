import cv2


def main():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    out = cv2.VideoWriter('face_detection.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 30, (int(cap.get(3)), int(cap.get(4))))
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.05, 5)
            for (x, y, w, h) in faces:
                frame = cv2.rectangle(frame, (x, y), (x + w, y + h), (150, 255, 0), 2)
            cv2.putText(frame, '[Press Q or Esc to Exit]', (int(cap.get(3) * 0.65), int(cap.get(4) * 0.95)), cv2.FONT_HERSHEY_DUPLEX, 0.5, (150, 0, 255), 1, cv2.LINE_AA)
            cv2.imshow('frame', frame)
            out.write(frame)
        else:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        ch_ = cv2.waitKey(1)
        if ch_ == 27 or ch_ == ord('q') or ch_ == ord('Q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()

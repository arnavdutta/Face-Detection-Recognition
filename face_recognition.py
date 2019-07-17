import numpy as np
import cv2
import os

subjects = ["", "Robert Downey Jr", "Chris Evans", "John Cena", "Unknown", "Roman Reigns"]

# function to detect face using OpenCV
def detect_face(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier('./haarcascade_frontalface_default.xml')
    face = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)

    # if no faces are detected then return original img
    if len(face) == 0:
        return None, None

    # under the assumption that there will be only one face, extract the face area
    x, y, w, h = face[0]

    # return only the face part of the image
    return gray[y:y + w, x:x + h], face[0]


# this function will read all persons' training images, detect face from each image
# and will return two lists of exactly same size, one list of faces and another list of labels for each face

def prepare_training_data(data_folder_path):
    # get the directories (one directory for each subject) in data folder
    dirs = os.listdir(data_folder_path)

    # list to hold all subject faces
    faces = []
    # list to hold labels for all subjects
    labels = []

    # let's go through each directory and read images within it
    for dir_name in dirs:

        # our subject directories start with letter 's' so
        # ignore any non-relevant directories if any
        if not dir_name.startswith("s"):
            continue

        # extract label number of subject from dir_name
        # format of dir name = slabel
        # removing letter 's' from dir_name will give us label
        label = int(dir_name.replace("s", ""))

        # build path of directory containing images for current subject subject
        # sample subject_dir_path = "training-data/s1"
        subject_dir_path = './dataset/training' + "/" + dir_name

        # get the images names that are inside the given subject directory
        subject_images_names = os.listdir(subject_dir_path)

        # go through each image name, read image,
        # detect face and add face to list of faces
        for image_name in subject_images_names:

            # ignore system files like .DS_Store
            if image_name.startswith("."):
                continue
            # build image path
            image_path = subject_dir_path + "/" + image_name

            # read image
            image = cv2.imread(image_path)

            # detect face
            face, rect = detect_face(image)

            if face is not None:
                faces.append(face)
                # print(face)
            # add label for this face
            labels.append(label)
            # print(label)

            # display an image window to show the image
            # cv2.imshow("Training on image...", image)
            # cv2.waitKey(100)
            # cv2.destroyAllWindows()

    return faces, labels


print("Preparing data...")
faces, labels = prepare_training_data("./dataset/training")
print("Total training faces: ", len(faces))
print("Total training labels: ", len(labels))
print("Preparing data...DONE")

# create our LBPH face recognizer
face_recognizer = cv2.face.LBPHFaceRecognizer_create()

# train our face recognizer of our training faces
face_recognizer.train(faces, np.array(labels))


# function to draw circle on image according to given (x, y) coordinates and given width and height
def draw_circle(img, rect):
    (x, y, w, h) = rect
    # cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.circle(img, (int((x + x + w) / 2), int((y + y + h) / 2)), int(h / 1.8), (180, 180, 100), 2)


# function to draw text on give image starting from passed (x, y) coordinates.
def draw_text(img, text, x, y):
    cv2.putText(img, text, (x, y - int(0.5 * y)), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)


# this function recognizes the person in image passed and draws a circle around detected face with name of the subject
def predict(test_img):
    # make a copy of the image as we don't want to change original image
    img = test_img.copy()
    # detect face from the image
    face, rect = detect_face(img)

    # predict the image using our face recognizer
    label = face_recognizer.predict(face)
    label = list(label)
    # print(label)

    # get name of respective label returned by face recognizer
    label_text = subjects[label[0]]

    # draw a circle around face detected
    draw_circle(img, rect)
    # draw name of predicted person
    draw_text(img, label_text, rect[0], rect[1] - 5)

    return img, label


test_list = [img for img in os.listdir("./dataset/test/") if img.endswith('.jpg')]
for img in test_list:

	print("Predicting images...")

	# load test images
	test_img = cv2.imread('./dataset/test/' + img)

	# perform a prediction
	predicted_img, label = predict(test_img)
	print(subjects[label[0]])
	cv2.imshow(subjects[label[0]], predicted_img)
	cv2.waitKey(0)
	cv2.destroyAllWindows()
print("All Predictions complete")
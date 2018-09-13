class Detector:
    def __init__(self, face_crop_size=160, face_crop_margin=32, minsize=110, threshold=None, factor=0.7):
        self.pnet, self.rnet, self.onet = self._setup_mtcnn()
        self.face_crop_size = face_crop_size
        self.face_crop_margin = face_crop_margin
        self.minsize = 110  # minimum size of face
        self.threshold = [0.6, 0.7, 0.7] if threshold is None else threshold # three steps's threshold
        self.factor = 0.709  # scale factor


    def _setup_mtcnn(self):
        with tf.Graph().as_default():
            gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_memory_fraction)
            sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
            with sess.as_default():
                return detect_face.create_mtcnn(sess, None)

    def find_faces(self, image):
        faces = []

        bounding_boxes = detect_face.detect_face(
            image, 
            self.minsize,
            self.pnet, self.rnet, self.onet,
            self.threshold, self.factor
        )
        for bb in bounding_boxes:
            bounding_box = np.zeros(4, dtype=np.int32)
            img_size = np.asarray(image.shape)[0:2]
            bounding_box[0] = np.maximum(bb[0] - self.face_crop_margin / 2, 0)
            bounding_box[1] = np.maximum(bb[1] - self.face_crop_margin / 2, 0)
            bounding_box[2] = np.minimum(bb[2] + self.face_crop_margin / 2, img_size[1])
            bounding_box[3] = np.minimum(bb[3] + self.face_crop_margin / 2, img_size[0])
            cropped = image[bounding_box[1]:bounding_box[3], bounding_box[0]:bounding_box[2], :]
            face = skimage.transform.resize(
                cropped, 
                (self.face_crop_size, self.face_crop_size),
                preserve_range=True,
                mode='reflect'
            )
            face = face.astype(dtype=np.uint8)
            faces.append(face)

        return faces


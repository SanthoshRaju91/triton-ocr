import math
import numpy as np
import cv2
from tritonclient.grpc import InferenceServerClient, InferInput, InferRequestedOutput


class TritonDBNet:
    def __init__(
        self,
        model_name,
        server_url="localhost:8001",
        text_threshold=0.2,
        bbox_min_score=0.2,
        bbox_min_size=3,
        max_candidates=0,
        BGR_MEAN=(123.68, 116.78, 103.94),
        device="cpu",
    ):
        self.model_name = model_name
        self.client = InferenceServerClient(server_url)
        self.text_threshold = text_threshold
        self.bbox_min_score = bbox_min_score
        self.bbox_min_size = bbox_min_size
        self.max_candidates = max_candidates
        self.BGR_MEAN = np.array(BGR_MEAN, dtype=np.float32)
        self.device = device
        self.min_detection_size = 640
        self.max_detection_size = 2560
        

    def resize_image(self, img, detection_size = None):
        height, width, _ = img.shape
        if detection_size is None:
            detection_size =  max(self.min_detection_size, min(height, width, self.max_detection_size))

        if height < width:
            new_height = int(math.ceil(detection_size / 32) * 32)
            new_width = int(math.ceil(new_height / height * width / 32) * 32)
        else:
            new_width = int(math.ceil(detection_size / 32) * 32)
            new_height = int(math.ceil(new_width / width * height / 32) * 32)
        resized_img = cv2.resize(img, (new_width, new_height))
        return resized_img, (height, width)

    def normalize_image(self, img):
        return (img.astype(np.float32) - self.BGR_MEAN) / 255.0

    def load_image(self, image_path):
        img = cv2.imread(image_path)
        if img is None:
            raise FileNotFoundError(f"Image not found at {image_path}")

        # Resize and normalize the image
        resized_img, original_shape = self.resize_image(img)
        normalized_img = self.normalize_image(resized_img)

        # Convert to CHW format and add batch dimension
        chw_image = np.transpose(normalized_img, (2, 0, 1))
        batch_image = np.expand_dims(chw_image, axis=0).astype(np.float32)
        return batch_image, original_shape

    def image2hmap(self, image_tensor):
        # Prepare Triton inputs
        print(image_tensor.shape)
        inputs = [InferInput("input", image_tensor.shape, "FP32")]
        print(inputs[0].set_data_from_numpy(image_tensor))
        inputs[0].set_data_from_numpy(image_tensor)

        # Prepare Triton outputs
        outputs = [InferRequestedOutput("output")]

        # Run inference
        response = self.client.infer(
            model_name=self.model_name, inputs=inputs, outputs=outputs
        )

        # Extract heatmap from the output
        heatmap = response.as_numpy("output")
        return heatmap

    def hmap2bbox(self, image_tensor, original_shape, hmap, as_polygon=False):
        segmentation = hmap[0, ..., 0] > self.text_threshold
        segmentation = (segmentation * 255).astype(np.uint8)

        contours, _ = cv2.findContours(
            segmentation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        original_height, original_width = original_shape
        resized_height, resized_width = image_tensor.shape[2:]

        scale_x = original_width / resized_width
        scale_y = original_height / resized_height

        bounding_boxes = []
        for contour in contours:
            if as_polygon:
                polygon = cv2.approxPolyDP(
                    contour, epsilon=0.01 * cv2.arcLength(contour, True), closed=True
                )
                polygon = np.array(polygon).reshape(-1, 2)
                scaled_polygon = polygon * [scale_x, scale_y]
                bounding_boxes.append(scaled_polygon.tolist())
            else:
                x, y, w, h = cv2.boundingRect(contour)
                x, y, w, h = (
                    int(x * scale_x),
                    int(y * scale_y),
                    int(w * scale_x),
                    int(h * scale_y),
                )
                if w > self.bbox_min_size and h > self.bbox_min_size:
                    bounding_boxes.append((x, y, x + w, y + h))

        return bounding_boxes

    def inference(self, image_path, as_polygon=False):
        # Preprocess the image
        image_tensor, original_shape = self.load_image(image_path)

        # Run inference to get heatmap
        heatmap = self.image2hmap(image_tensor)

        # Convert heatmap to bounding boxes or polygons
        results = self.hmap2bbox(image_tensor, original_shape, heatmap, as_polygon)

        return results


def plot_detections(image_path, bounding_boxes=None, polygons=None, output_path="output_image.jpg"):
    # Load the original image
    image = cv2.imread(image_path)

    if image is None:
        raise FileNotFoundError(f"Image not found at {image_path}")

    # Plot bounding boxes
    if bounding_boxes:
        for (x1, y1, x2, y2) in bounding_boxes:
            cv2.rectangle(image, (x1, y1), (x2, y2), color=(0, 255, 0), thickness=2)  # Green rectangle

    # Plot polygons
    if polygons:
        for polygon in polygons:
            pts = np.array(polygon, np.int32)  # Convert polygon points to NumPy array
            pts = pts.reshape((-1, 1, 2))  # Reshape for OpenCV polylines
            cv2.polylines(image, [pts], isClosed=True, color=(255, 0, 0), thickness=2)  # Blue polygon

    # Save the resulting image
    cv2.imwrite(output_path, image)
    print(f"Output image saved to {output_path}")


def scale_bounding_boxes(bboxes, original_shape, resized_shape):
    original_height, original_width = original_shape
    resized_height, resized_width = resized_shape

    scale_x = original_width / resized_width
    scale_y = original_height / resized_height

    scaled_bboxes = []
    for x1, y1, x2, y2 in bboxes:
        x1 = int(x1 * scale_x)
        y1 = int(y1 * scale_y)
        x2 = int(x2 * scale_x)
        y2 = int(y2 * scale_y)
        scaled_bboxes.append((x1, y1, x2, y2))
    return scaled_bboxes

if __name__ == "__main__":
    dbnet = TritonDBNet(
        model_name="detection_model",
        server_url="localhost:8001",
        text_threshold=0.2,
        bbox_min_score=0.2,
        bbox_min_size=3
    )
    image_path = "simple-text.png"
    bounding_boxes = dbnet.inference(image_path=image_path, as_polygon=False)
    scaled_bounding_boxes = scale_bounding_boxes(bounding_boxes, (1664, 928), (830, 465))
    plot_detections(image_path=image_path, bounding_boxes=scaled_bounding_boxes, polygons=None, output_path="output_image_with_bbox.jpg")

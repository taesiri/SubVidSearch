import cv2
import numpy as np
from scipy.spatial.distance import cosine
from fastdtw import fastdtw
import matplotlib.pyplot as plt
from PIL import Image
import torch
from torchvision import models, transforms
from tqdm import tqdm


class VideoFeatureExtractor:
    def __init__(self, feature_type="resnet"):
        """Initialize the feature extractor.

        Args:
            feature_type (str): Type of features to extract ('resnet', 'vgg', etc.)
        """
        self.feature_type = feature_type

        # Set up the model for feature extraction
        if feature_type == "resnet":
            model = models.resnet50(pretrained=True)
            # Remove the final fully connected layer
            self.model = torch.nn.Sequential(*list(model.children())[:-1])
        elif feature_type == "vgg":
            model = models.vgg16(pretrained=True)
            # Use features from the last convolutional layer
            self.model = model.features
        else:
            raise ValueError(f"Unsupported feature type: {feature_type}")

        self.model.eval()
        if torch.cuda.is_available():
            self.model = self.model.cuda()

        # Set up the transformation pipeline
        self.transform = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    def extract_features(self, video_path, sample_rate=1):
        """Extract features from video frames.

        Args:
            video_path (str): Path to the video file
            sample_rate (int): Sample every Nth frame

        Returns:
            np.ndarray: Array of features, one per sampled frame
        """
        cap = cv2.VideoCapture(video_path)
        features = []
        frame_indices = []

        with torch.no_grad():
            frame_idx = 0
            pbar = tqdm(desc=f"Extracting features from {video_path}")

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_idx % sample_rate == 0:
                    # Convert from BGR to RGB and from numpy to PIL
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    pil_image = Image.fromarray(frame_rgb)

                    # Apply transforms and add batch dimension
                    input_tensor = self.transform(pil_image).unsqueeze(0)
                    if torch.cuda.is_available():
                        input_tensor = input_tensor.cuda()

                    # Extract features
                    feature = self.model(input_tensor)
                    feature = feature.squeeze().cpu().numpy()

                    features.append(feature)
                    frame_indices.append(frame_idx)

                frame_idx += 1
                pbar.update(1)

            pbar.close()

        cap.release()
        return np.array(features), frame_indices


def compute_distance_matrix(features1, features2):
    """Compute pairwise distance matrix between two sets of features.

    Args:
        features1 (np.ndarray): Features from first video
        features2 (np.ndarray): Features from second video

    Returns:
        np.ndarray: Distance matrix
    """
    n = len(features1)
    m = len(features2)
    distance_matrix = np.zeros((n, m))

    for i in range(n):
        for j in range(m):
            distance_matrix[i, j] = cosine(features1[i], features2[j])

    return distance_matrix


def apply_dtw(features1, features2):
    """Apply Dynamic Time Warping to find the optimal alignment.

    Args:
        features1 (np.ndarray): Features from first video
        features2 (np.ndarray): Features from second video

    Returns:
        tuple: (distance, path)
    """
    distance, path = fastdtw(features1, features2, dist=cosine)
    return distance, path


def visualize_matches(video1_path, video2_path, matches, output_path):
    """Visualize matching frames side by side.

    Args:
        video1_path (str): Path to first video
        video2_path (str): Path to second video
        matches (list): List of (frame_idx1, frame_idx2) pairs
        output_path (str): Path to save visualization
    """
    cap1 = cv2.VideoCapture(video1_path)
    cap2 = cv2.VideoCapture(video2_path)

    fig, axes = plt.subplots(len(matches), 2, figsize=(10, 3 * len(matches)))

    for i, (idx1, idx2) in enumerate(matches):
        # Get frame from first video
        cap1.set(cv2.CAP_PROP_POS_FRAMES, idx1)
        ret, frame1 = cap1.read()
        frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)

        # Get frame from second video
        cap2.set(cv2.CAP_PROP_POS_FRAMES, idx2)
        ret, frame2 = cap2.read()
        frame2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)

        # Display frames
        axes[i, 0].imshow(frame1)
        axes[i, 0].set_title(f"Video 1, Frame {idx1}")
        axes[i, 0].axis("off")

        axes[i, 1].imshow(frame2)
        axes[i, 1].set_title(f"Video 2, Frame {idx2}")
        axes[i, 1].axis("off")

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

    cap1.release()
    cap2.release()


def find_matching_frames(video1_path, video2_path, sample_rate=1, num_matches=5):
    """Find matching frames between two videos using DTW.

    Args:
        video1_path (str): Path to first video
        video2_path (str): Path to second video
        sample_rate (int): Sample every Nth frame
        num_matches (int): Number of matching pairs to return

    Returns:
        list: List of (frame_idx1, frame_idx2) pairs
    """
    # Extract features
    extractor = VideoFeatureExtractor()
    print("Extracting features from first video...")
    features1, indices1 = extractor.extract_features(video1_path, sample_rate)
    print("Extracting features from second video...")
    features2, indices2 = extractor.extract_features(video2_path, sample_rate)

    # Apply DTW
    print("Applying DTW...")
    _, path = apply_dtw(features1, features2)

    # Convert path to frame indices
    matches = [(indices1[i], indices2[j]) for i, j in path]

    # Select evenly spaced matches
    if len(matches) > num_matches:
        step = len(matches) // num_matches
        selected_matches = [matches[i * step] for i in range(num_matches)]
    else:
        selected_matches = matches

    return selected_matches


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Match frames between two videos using DTW"
    )
    parser.add_argument("--video1", required=True, help="Path to first video")
    parser.add_argument("--video2", required=True, help="Path to second video")
    parser.add_argument(
        "--sample_rate", type=int, default=5, help="Sample every Nth frame"
    )
    parser.add_argument(
        "--num_matches", type=int, default=5, help="Number of matching pairs to show"
    )
    parser.add_argument(
        "--output", default="matches.png", help="Output visualization path"
    )

    args = parser.parse_args()

    matches = find_matching_frames(
        args.video1,
        args.video2,
        sample_rate=args.sample_rate,
        num_matches=args.num_matches,
    )

    print(f"Found {len(matches)} matching frames")
    visualize_matches(args.video1, args.video2, matches, args.output)
    print(f"Visualization saved to {args.output}")


# # Example paths - update these!
# VIDEO_PATH_1 = (
#     "match_results/match_Dw9cBXaT0ao_vs_KJkPMbhlMnU_at_030944670/KJkPMbhlMnU_full.mp4"
# )
# VIDEO_PATH_2 = "match_results/match_Dw9cBXaT0ao_vs_KJkPMbhlMnU_at_030944670/Dw9cBXaT0ao_segment_030944670.mp4"
# MATCHED_FRAMES_OUTPUT_DIR = (
#     "matched_frames_output"  # Directory to save matched frame pairs
# )

# python video_frame_matching.py --video1 path/to/video1.mp4 --video2 path/to/video2.mp4 --sample_rate 5 --num_matches 10 --output matches.png

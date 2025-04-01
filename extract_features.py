import cv2
import numpy as np
from PIL import Image
import torch
from torchvision import models, transforms
from tqdm import tqdm
import argparse
import os


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
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)
        print(f"Using device: {self.device}")

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
            tuple: (np.ndarray features, list frame_indices)
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError(f"Could not open video file: {video_path}")

        features = []
        frame_indices = []
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"Total frames in {os.path.basename(video_path)}: {total_frames}")

        with torch.no_grad():
            frame_idx = 0
            pbar = tqdm(
                total=total_frames,
                desc=f"Extracting features from {os.path.basename(video_path)}",
                unit="frame",
            )

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_idx % sample_rate == 0:
                    # Convert from BGR to RGB and from numpy to PIL
                    try:
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        pil_image = Image.fromarray(frame_rgb)
                    except Exception as e:
                        print(f"Error processing frame {frame_idx}: {e}")
                        frame_idx += 1
                        pbar.update(1)
                        continue

                    # Apply transforms and add batch dimension
                    input_tensor = self.transform(pil_image).unsqueeze(0)
                    input_tensor = input_tensor.to(self.device)

                    # Extract features
                    feature = self.model(input_tensor)
                    # Flatten the feature vector for ResNet/VGG
                    feature = torch.flatten(feature, start_dim=1)
                    feature = feature.squeeze().cpu().numpy()

                    features.append(feature)
                    frame_indices.append(frame_idx)

                frame_idx += 1
                pbar.update(1)

            pbar.close()

        cap.release()
        if not features:
            print(
                f"Warning: No features extracted from {video_path}. Check sample rate and video content."
            )
            return np.array([]), []
        return np.array(features), frame_indices


def main():
    parser = argparse.ArgumentParser(
        description="Extract features from two video files."
    )
    parser.add_argument("--video1", required=True, help="Path to the first video file.")
    parser.add_argument(
        "--video2", required=True, help="Path to the second video file."
    )
    parser.add_argument(
        "--output_dir",
        default="features",
        help="Directory to save the output feature files.",
    )
    parser.add_argument(
        "--sample_rate", type=int, default=5, help="Sample every Nth frame."
    )
    parser.add_argument(
        "--feature_type",
        default="resnet",
        choices=["resnet", "vgg"],
        help="Type of CNN features to extract.",
    )

    args = parser.parse_args()

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Initialize extractor
    extractor = VideoFeatureExtractor(feature_type=args.feature_type)

    # Process first video
    print(f"\nProcessing video 1: {args.video1}")
    output_path1 = os.path.join(
        args.output_dir,
        f"{os.path.splitext(os.path.basename(args.video1))[0]}_features.npz",
    )
    if os.path.exists(output_path1):
        print(
            f"Features for video 1 already exist at {output_path1}. Skipping extraction."
        )
    else:
        features1, indices1 = extractor.extract_features(args.video1, args.sample_rate)
        if len(features1) > 0:
            np.savez(output_path1, features=features1, indices=indices1)
            print(
                f"Saved features for video 1 ({len(features1)} frames) to {output_path1}"
            )
        else:
            print(f"No features extracted for video 1.")

    # Process second video
    print(f"\nProcessing video 2: {args.video2}")
    output_path2 = os.path.join(
        args.output_dir,
        f"{os.path.splitext(os.path.basename(args.video2))[0]}_features.npz",
    )
    if os.path.exists(output_path2):
        print(
            f"Features for video 2 already exist at {output_path2}. Skipping extraction."
        )
    else:
        features2, indices2 = extractor.extract_features(args.video2, args.sample_rate)
        if len(features2) > 0:
            np.savez(output_path2, features=features2, indices=indices2)
            print(
                f"Saved features for video 2 ({len(features2)} frames) to {output_path2}"
            )
        else:
            print(f"No features extracted for video 2.")

    print("\nFeature extraction complete.")


if __name__ == "__main__":
    main()

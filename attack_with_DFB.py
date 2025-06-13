import torch
import torchvision.transforms as T
import matplotlib.pyplot as plt
from timm import create_model
import transferattack
from transferattack.attack import Attack
from transferattack.utils import EnsembleModel
import sys
import os
import cv2
from tqdm import tqdm


#set path to import /mnt/data/DeepfakeBench/DeepfakeBench directory
sys.path.append("/mnt/data/DeepfakeBench/DeepfakeBench")

from export_model import load_model, face_cropper, face_paster

# Create a wrapper class to make our model compatible with the attack
class DFBModel(torch.nn.Module):
    def __init__(self, model_name):
        super().__init__()
        model, process, configs = load_model(model_name, {"test_batch_size": 1})
        self.model = model.eval().cuda()
        self.configs = configs
        self.process = process
        print(f"[DEBUG] Wrapped model type: {model.__class__.__name__}")
    
    def forward(self, x):

        x=x.cuda()
        # Process the input image
        x = self.process(x)

        output = self.model({"image": x, "label": torch.tensor([1]).cuda()}, inference=True)
        return output['cls']


def attack(attack_name, images, model_name, batch_size=16):
    
    # Load the attack class from transferattack
    attack_class = transferattack.load_attack_class(attack_name)
    print(f"[DEBUG] Loaded attack class: {attack_class.__name__}")

    def custom_load_model(self, model_name):
        if isinstance(model_name, list):
            return EnsembleModel([DFBModel(name) for name in model_name])
        else:
            return DFBModel(model_name)


    attack_class.load_model = custom_load_model

    # Directly edit the function in the attacker instance

    # Wrap the obtained attack_class instance with AttackWrapper
    attacker = attack_class(model_name)


    attacker.epsilon = 10 / 255.0
    attacker.epoch = 10


    transform = T.ToPILImage()

    output_perturbed_images = []

    # Process images in batches
    for i in tqdm(range(0, images.size(0), batch_size), desc="Processing batches"):
        batch = images[i:i + batch_size].cuda()  # Slice the batch and move to GPU

        labels = torch.tensor([1] * len(batch)).cuda()  # Example label tensor, adjust as needed

        # Perform the attack
        if attack_name in ['ttp', 'm3d']: 
            raise NotImplementedError(f"{attack_name} is not supported yet")


        perturbations = attacker(batch, labels)

        # Add perturbation in normalized space
        perturbed_images = batch + perturbations

        # Convert perturbed images to PIL format
        for j in range(perturbed_images.shape[0]):
            # convert tensor to PIL image for visualization
            image = transform(perturbed_images[j].cpu())
            output_perturbed_images.append(image)

            #Display the perturbed image
            # plt.imshow(image)
            # plt.title("Perturbed Image")
            # plt.axis("off")
            # plt.show()

    return output_perturbed_images

if __name__ == "__main__":

    attack_name = 'svre'  # or 'mifgsm'
    model_name = ['xception', 'ucf', 'f3net']  # or 'xception' for single model

    # video usage example
    video_root_path = "/mnt/data/Test Samples"

    # scan all mp4 files in the video_root_path
    video_paths = [os.path.join(video_root_path, f) for f in os.listdir(video_root_path) if (f.endswith('.mp4') and not f.endswith('_real.mp4'))]

    for i in range(len(video_paths)):
        video_path = video_paths[i]

        # break into frames
        print(f"\n[Stage] Processing video {i}/{len(video_paths)}")
        print(f"[Stage] Extracting frames from video: {video_path}")
        video = cv2.VideoCapture(video_path)
        frames = []
        while True:
            ret, frame = video.read()
            if not ret:
                break
            frames.append(frame)

        video.release()

        print(f"[Stage] Extracting faces from {len(frames)} video frames.")
        faces, bbox = face_cropper(frames, size=256)

        # record indexes of None faces and remove them
        # will need to add back the None faces later
        none_face_indexes = [i for i, face in enumerate(faces) if face is None]
        faces = [T.ToTensor()(face) for face in faces if face is not None]

        faces = torch.stack(faces).cuda()  # Stack faces into a single tensor and move to GPU


        print(f"[Stage] Training perturbation.")

        perturbed_faces = attack(attack_name, faces, model_name)

        print(f"[Stage] Adding back faces to {len(frames)} video frames.")

        # Add back the None faces to perturbed frames
        for index in none_face_indexes:
            perturbed_faces.insert(index, None)

        # Paste perturbed faces back to the original frames
        perturbed_frames = face_paster(perturbed_faces, bbox, frames)

        # Save perturbed frames to a new video file
        output_video_root_path = "./output_videos"
        if not os.path.exists(output_video_root_path):
            os.makedirs(output_video_root_path)

        model_name_str = "_".join(model_name) if isinstance(model_name, list) else model_name

        output_video_path = os.path.join(output_video_root_path, f"{attack_name}_{model_name_str}_" + os.path.basename(video_path))


        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        height, width = perturbed_frames[0].shape[:2]
        out = cv2.VideoWriter(output_video_path, fourcc, 30.0, (width, height))
        for frame in perturbed_frames:
            if frame is not None:
                out.write(frame)
            else:
                # If the frame is None, write a black frame
                black_frame = np.zeros((height, width, 3), dtype=np.uint8)
                out.write(black_frame)
        out.release()
        print(f"[Stage] Perturbed video saved to {output_video_path}")

    


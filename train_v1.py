import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
from translate import Translator
from PIL import Image
import pandas as pd
import os
from tqdm import tqdm
import numpy as np
from torch.optim import AdamW
from torch.nn import CrossEntropyLoss


class Flickr30kDataset(Dataset):
    def __init__(self, image_dir, captions_file, image_processor, tokenizer, max_length=128):
        self.image_dir = image_dir
        self.image_processor = image_processor
        self.tokenizer = tokenizer
        self.max_length = max_length

        # Load captions
        self.data = pd.read_csv(captions_file)
        self.image_paths = self.data['image'].tolist()
        self.captions = self.data['caption'].tolist()

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load and preprocess image
        image_path = os.path.join(self.image_dir, self.image_paths[idx])
        image = Image.open(image_path).convert('RGB')
        pixel_values = self.image_processor(
            image, return_tensors="pt").pixel_values

        # Process caption
        caption = self.captions[idx]
        # For translate_first approach, caption is already in Nepali
        # For translate_output approach, caption remains in English

        # Tokenize caption
        encoding = self.tokenizer(
            caption,
            padding="max_length",
            max_length=self.max_length,
            truncation=True,
            return_tensors="pt"
        )

        return {
            'pixel_values': pixel_values.squeeze(),
            'labels': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze()
        }


class NepaliImageCaptioning:
    def __init__(self, approach="translate_first", device="cuda"):
        self.approach = approach
        self.device = device

        # Initialize vision encoder-decoder model
        self.model = VisionEncoderDecoderModel.from_pretrained(
            "nlpconnect/vit-gpt2-image-captioning")
        self.image_processor = ViTImageProcessor.from_pretrained(
            "nlpconnect/vit-gpt2-image-captioning")

        if approach == "translate_first":
            self.tokenizer = AutoTokenizer.from_pretrained(
                "nepal-nlp/nepali-bert")
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(
                "nlpconnect/vit-gpt2-image-captioning")

        self.translator = Translator(to_lang="ne")
        self.model.to(device)

    def prepare_dataset(self, image_dir, captions_file, batch_size=32):
        """Prepare dataset for training"""
        if self.approach == "translate_first":
            # Translate captions to Nepali
            df = pd.read_csv(captions_file)
            print("Translating captions to Nepali...")
            df['caption'] = df['caption'].progress_apply(
                lambda x: self.translator.translate(x)
            )
            # Save translated captions
            translated_file = captions_file.replace('.csv', '_nepali.csv')
            df.to_csv(translated_file, index=False)
            captions_file = translated_file

        # Create dataset and dataloader
        dataset = Flickr30kDataset(
            image_dir=image_dir,
            captions_file=captions_file,
            image_processor=self.image_processor,
            tokenizer=self.tokenizer
        )

        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4
        )

        return dataloader

    def train(self, train_dataloader, num_epochs=10, learning_rate=1e-4):
        """Training loop implementation"""
        optimizer = AdamW(self.model.parameters(), lr=learning_rate)
        criterion = CrossEntropyLoss()
        scaler = torch.cuda.amp.GradScaler()

        best_loss = float('inf')

        for epoch in range(num_epochs):
            self.model.train()
            total_loss = 0
            progress_bar = tqdm(train_dataloader, desc=f'Epoch {
                                epoch + 1}/{num_epochs}')

            for batch in progress_bar:
                # Move batch to device
                pixel_values = batch['pixel_values'].to(self.device)
                labels = batch['labels'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)

                # Clear gradients
                optimizer.zero_grad()

                # Forward pass with automatic mixed precision
                with torch.cuda.amp.autocast():
                    outputs = self.model(
                        pixel_values=pixel_values,
                        labels=labels,
                        attention_mask=attention_mask
                    )

                    loss = outputs.loss

                # Backward pass with gradient scaling
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                total_loss += loss.item()
                progress_bar.set_postfix({'loss': loss.item()})

            avg_loss = total_loss / len(train_dataloader)
            print(f'Epoch {epoch + 1} - Average Loss: {avg_loss:.4f}')

            # Save best model
            if avg_loss < best_loss:
                best_loss = avg_loss
                self.save_model(f'best_model_epoch_{epoch + 1}')

    def save_model(self, path):
        """Save model and tokenizer"""
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        self.image_processor.save_pretrained(path)

    def generate_caption(self, image):
        """Generate caption for an image"""
        self.model.eval()
        with torch.no_grad():
            pixel_values = self.image_processor(
                image, return_tensors="pt").pixel_values.to(self.device)

            output_ids = self.model.generate(
                pixel_values,
                max_length=50,
                num_beams=4,
                early_stopping=True,
                no_repeat_ngram_size=2
            )

            if self.approach == "translate_first":
                caption = self.tokenizer.decode(
                    output_ids[0], skip_special_tokens=True)
            else:
                caption = self.tokenizer.decode(
                    output_ids[0], skip_special_tokens=True)
                try:
                    caption = self.translator.translate(caption)
                except Exception as e:
                    print(f"Translation error: {e}")

            return caption


def main():
    # Configuration
    IMAGE_DIR = "path/to/flickr30k/images"
    CAPTIONS_FILE = "path/to/flickr30k/captions.csv"
    BATCH_SIZE = 32
    NUM_EPOCHS = 10
    LEARNING_RATE = 1e-4
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    APPROACH = "translate_output"  # or "translate_first"

    # Initialize model
    model = NepaliImageCaptioning(approach=APPROACH, device=DEVICE)

    # Prepare dataset
    train_dataloader = model.prepare_dataset(
        image_dir=IMAGE_DIR,
        captions_file=CAPTIONS_FILE,
        batch_size=BATCH_SIZE
    )

    # Train model
    model.train(
        train_dataloader=train_dataloader,
        num_epochs=NUM_EPOCHS,
        learning_rate=LEARNING_RATE
    )

    # Test generation
    test_image = Image.open("path/to/test/image.jpg").convert('RGB')
    caption = model.generate_caption(test_image)
    print(f"Generated caption: {caption}")


if __name__ == "__main__":
    main()

from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace
from pathlib import Path
from torch.utils.data import DataLoader
from model import build_transformer

from dataset import BilingualDataset

from datasets import load_dataset
from torch.utils.data import random_split

raw_dataset = load_dataset("Helsinki-NLP/opus_books", "en-it", split="train")

train_size = int(0.9 * len(raw_dataset))
val_size = len(raw_dataset) - train_size
raw_train_ds, raw_val_ds = random_split(raw_dataset, [train_size, val_size])

print(f"Loaded successfully! Total rows: {len(raw_dataset)}")


def get_all_sentences(ds, lang):
    """A helper generator to yield sentences for tokenizer training"""
    for item in ds:
        yield item['translation'][lang]

def get_or_build_tokenizer(ds, lang, vocab_path):
    vocab_path = Path(vocab_path)
    if not vocab_path.exists():
        tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = Whitespace()
        
        trainer = WordLevelTrainer(
            special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"], 
            min_frequency=2
        )
        
        tokenizer.train_from_iterator(get_all_sentences(ds, lang), trainer=trainer)
        tokenizer.save(str(vocab_path))
    else:
        tokenizer = Tokenizer.from_file(str(vocab_path))
        
    return tokenizer

tokenizer_src = get_or_build_tokenizer(raw_dataset, "en", "tokenizer_en.json")
tokenizer_tgt = get_or_build_tokenizer(raw_dataset, "it", "tokenizer_it.json")

print("Tokenizers built and loaded successfully!")


seq_len = 350
src_lang = "en"
tgt_lang = "it"

train_ds = BilingualDataset(
    raw_train_ds, 
    tokenizer_src, 
    tokenizer_tgt, 
    src_lang, 
    tgt_lang, 
    seq_len
)

val_ds = BilingualDataset(
    raw_val_ds, 
    tokenizer_src, 
    tokenizer_tgt, 
    src_lang, 
    tgt_lang, 
    seq_len
)



train_dataloader = DataLoader(train_ds, batch_size = 8, shuffle = True)
test_dataloader = DataLoader(val_ds, batch_size = 8, shuffle = True)




# Training loop....
import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

src_vocab_size = tokenizer_src.get_vocab_size()
tgt_vocab_size = tokenizer_tgt.get_vocab_size()

# FIX 2: Use target tokenizer for target padding index
pad_token_id = tokenizer_tgt.token_to_id("[PAD]")

model = build_transformer(
    src_vocab=src_vocab_size,
    tgt_vocab=tgt_vocab_size,
    src_seq_len=seq_len,
    tgt_seq_len=seq_len,
    d_model=512,
    dropout=0.1,
    h=8,
    N=6,
    d_ff=2048
).to(device)

# FIX 3: Lowered learning rate from 1e-3 to 1e-4 for stability
optimizer = torch.optim.AdamW(params=model.parameters(), lr=1e-4, betas=(0.9, 0.98), eps=1e-9)
loss_fn = nn.CrossEntropyLoss(ignore_index=pad_token_id, label_smoothing=0.1)

epochs = 3
for epoch in range(epochs):
    model.train()
    total_loss = 0

    for batch_idx, batch in enumerate(train_dataloader):
        optimizer.zero_grad()
        
        encoder_input = batch['encoder_input'].to(device)
        decoder_input = batch['decoder_input'].to(device)
        label = batch['label'].to(device)
        encoder_mask = batch['encoder_mask'].to(device)
        decoder_mask = batch['decoder_mask'].to(device)

        encoder_output = model.encode(encoder_input, encoder_mask)
        decoder_output = model.decode(decoder_input, encoder_output, encoder_mask, decoder_mask)
        projection_output = model.project(decoder_output)

        loss = loss_fn(projection_output.view(-1, tgt_vocab_size), label.view(-1))

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        if batch_idx % 100 == 0:
            # FIX 1: Use loss.item() instead of loss
            print(f"Epoch: {epoch+1}/{epochs} | Batch: {batch_idx}/{len(train_dataloader)} | Loss: {loss.item():.4f}")
    
    avg_loss = total_loss / len(train_dataloader)
    print(f"\nEpoch {epoch + 1} Finished. Average Loss: {avg_loss:.4f}\n")

print("Training complete..")
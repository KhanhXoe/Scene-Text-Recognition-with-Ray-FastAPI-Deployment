import torch

def evaluate(model, data_loader, criterion, device):
    model.eval()
    val_losses = []
    with torch.no_grad():
        for img, label, text_len in data_loader:
            img = img.to(device)
            label = label.to(device)
            text_len = text_len.to(device)

            outputs = model(img)
            logits_lens = torch.full(
                size=(outputs.size(1),),
                fill_value=outputs.size(0),
                dtype=torch.long
            ).to(device)

            loss = criterion(outputs, label, logits_lens, text_len)
            val_losses.append(loss.item())

    loss = sum(val_losses) / len(val_losses)
    return loss

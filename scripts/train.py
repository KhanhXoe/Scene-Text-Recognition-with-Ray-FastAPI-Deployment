import torch
import time 
from scripts import evaluate

def fit(
    model, 
    train_loader, val_loader, 
    criterion, optimizer, scheduler, 
    device, epochs, patience=5
):
    train_losses = []
    val_losses = []

    best_val_loss = float("inf")
    patience_counter = 0
    best_model_state = None

    for epoch in range(epochs):
        start = time.time()
        batch_train_losses = []

        model.train()
        for idx, (inputs, labels, labels_len) in enumerate(train_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            labels_len = labels_len.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)

            logits_lens = torch.full(
                size=(outputs.size(1),),
                fill_value=outputs.size(0),
                dtype=torch.long
            ).to(device)

            loss = criterion(outputs, labels, logits_lens, labels_len)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()

            batch_train_losses.append(loss.item())

        train_loss = sum(batch_train_losses) / len(batch_train_losses)
        train_losses.append(train_loss)

        val_loss = evaluate(model, val_loader, criterion, device)
        val_losses.append(val_loss)

        print(
            f"EPOCH {epoch + 1}: "
            f"\tTrain loss: {train_loss:.4f}"
            f"\tVal loss: {val_loss:.4f}"
            f"\tTime: {time.time() - start:.2f} seconds"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = model.state_dict()  # save best model
        else:
            patience_counter += 1
            print(f"  Patience counter: {patience_counter}/{patience}")

        if patience_counter >= patience:
            print("⚠️ Early stopping triggered!")
            break

        scheduler.step()
        
    if best_model_state:
        model.load_state_dict(best_model_state)

    return train_losses, val_losses, model
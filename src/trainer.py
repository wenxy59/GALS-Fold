import os
import numpy as np
from tqdm import tqdm
import wandb
from sklearn.metrics import confusion_matrix

import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from src.evaluator import evaluate
from src.constants import NUM_TO_LETTER


def train(
        config, 
        model, 
        train_loader, 
        val_loader, 
        test_loader, 
        device
    ):
    """
    Train RNA inverse folding model using the specified config and data loaders.

    Args:
        config (dict): wandb configuration dictionary 
        model (nn.Module): RNA inverse folding model to be trained
        train_loader (DataLoader): training data loader
        val_loader (DataLoader): validation data loader
        test_loader (DataLoader): test data loader
        device (torch.device): device to train the model on
    """

    # Initialise loss function
    train_loss_fn = torch.nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)
    eval_loss_fn = torch.nn.CrossEntropyLoss(label_smoothing=0.0)
    
    # Optimizer and scheduler
    lr = config.lr
    weight_decay = getattr(config, 'weight_decay', 0.0)
    optimizer = Adam(model.parameters(), lr, weight_decay=weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.9, patience=1, min_lr=0.00001)

    if device.type == 'xpu':
        import intel_extension_for_pytorch as ipex
        model, optimizer = ipex.optimize(model, optimizer=optimizer)

    # Save directory and checkpoint prefix
    save_dir = getattr(config, 'save_dir', './trainedmodels')
    os.makedirs(save_dir, exist_ok=True)
    ckpt_prefix = getattr(config, 'checkpoint_prefix', '')
    ckpt_prefix = f"{ckpt_prefix}_" if ckpt_prefix else ""

    # Lookup table for nucleotides
    lookup = train_loader.dataset.featurizer.num_to_letter

    # Auxiliary loss weight
    aux_loss_weight = getattr(config, 'aux_loss_weight', 0.0)
    if aux_loss_weight == 0.0 and hasattr(model, 'aux_loss_weight'):
        aux_loss_weight = model.aux_loss_weight

    # Early stopping
    best_epoch, best_val_loss, best_val_acc = -1, np.inf, 0
    patience = getattr(config, 'patience', 5)
    early_stopping_counter = 0

    ##################################
    # Training loop over mini-batches
    ##################################

    for epoch in range(config.epochs):

        # Training iteration
        model.train()
        train_loss, train_acc, train_confusion, train_aux_loss = loop(
            model, train_loader, train_loss_fn, optimizer, device, aux_loss_weight)
        print_and_log(epoch, train_loss, train_acc, train_confusion, lr=lr, mode="train",
                      lookup=lookup, aux_loss=train_aux_loss if aux_loss_weight > 0 else None)

        if epoch % config.val_every == 0 or epoch == config.epochs - 1:

            model.eval()
            with torch.no_grad():

                # Evaluate on validation set
                val_loss, val_acc, val_confusion, _ = loop(model, val_loader, eval_loss_fn, None, device)
                print_and_log(epoch, val_loss, val_acc, val_confusion, mode="val", lookup=lookup)

                # LR scheduler step
                scheduler.step(val_acc)
                lr = optimizer.param_groups[0]['lr']

                if val_acc > best_val_acc:
                    print(f"Validation accuracy improved ({best_val_acc:.4f} --> {val_acc:.4f}). Saving model...")
                    early_stopping_counter = 0
                    best_epoch, best_val_loss, best_val_acc = epoch, val_loss, val_acc

                    test_loss, test_acc, test_confusion, _ = loop(model, test_loader, eval_loss_fn, None, device)
                    print_and_log(epoch, test_loss, test_acc, test_confusion, mode="test", lookup=lookup)

                    wandb.run.summary["best_epoch"] = best_epoch
                    wandb.run.summary["best_val_perp"] = np.exp(best_val_loss)
                    wandb.run.summary["best_val_acc"] = best_val_acc
                    wandb.run.summary["best_test_perp"] = np.exp(test_loss)
                    wandb.run.summary["best_test_acc"] = test_acc

                    if config.save:
                        checkpoint_path = os.path.join(save_dir, f"{ckpt_prefix}best_checkpoint.h5")
                        torch.save(model.state_dict(), checkpoint_path)
                        wandb.run.summary["best_checkpoint"] = checkpoint_path

                else:
                    early_stopping_counter += 1
                    print(f"Validation accuracy did not improve. EarlyStopping counter: {early_stopping_counter}/{patience}")
                    if early_stopping_counter >= patience:
                        print(f"Early stopping triggered after {patience} epochs of no improvement.")
                        break

        if config.save:
            torch.save(model.state_dict(), os.path.join(save_dir, f"{ckpt_prefix}current_checkpoint.h5"))

    print("--- End of Training ---")


def loop(model, dataloader, loss_fn, optimizer=None, device='cpu', aux_loss_weight=0.0):
    """
    Single epoch training/evaluation loop.

    Args:
        model: RNA inverse folding model.
        dataloader: Data loader for current epoch.
        loss_fn: Loss function.
        optimizer: Optimizer (None for evaluation).
        device: Device to run on.
        aux_loss_weight: Weight for auxiliary loss.

    Returns:
        Tuple of (avg_loss, avg_accuracy, confusion_matrix, avg_aux_loss).
    """
    confusion = np.zeros((model.out_dim, model.out_dim))
    total_loss, total_correct, total_count = 0, 0, 0
    total_aux_loss = 0.0
    has_aux_loss = hasattr(model, 'aux_loss_weight')

    t = tqdm(dataloader)
    for batch in t:
        if optimizer: optimizer.zero_grad()
        batch = batch.to(device)

        try:
            if optimizer and has_aux_loss and aux_loss_weight > 0:
                logits, aux_loss = model(batch, return_aux_loss=True)
            else:
                logits = model(batch, return_aux_loss=False) if has_aux_loss else model(batch)
                aux_loss = 0.0
        except RuntimeError as e:
            if "CUDA out of memory" not in str(e): raise(e)
            print('Skipped batch due to OOM', flush=True)
            for p in model.parameters():
                if p.grad is not None:
                    del p.grad
            torch.cuda.empty_cache()
            continue

        task_loss = loss_fn(logits, batch.seq)
        if optimizer and has_aux_loss and aux_loss_weight > 0:
            loss_value = task_loss + aux_loss_weight * aux_loss
        else:
            loss_value = task_loss

        if optimizer:
            loss_value.backward()
            optimizer.step()

        num_nodes = int(batch.seq.size(0))
        total_loss += float(task_loss.item()) * num_nodes
        total_count += num_nodes
        if isinstance(aux_loss, torch.Tensor):
            total_aux_loss += float(aux_loss.item()) * num_nodes
        pred = torch.argmax(logits, dim=-1).detach().cpu().numpy()
        true = batch.seq.detach().cpu().numpy()
        total_correct += (pred == true).sum()
        confusion += confusion_matrix(true, pred, labels=range(model.out_dim))

        t.set_description("%.5f" % float(total_loss/total_count))

    avg_aux_loss = total_aux_loss / total_count if total_count > 0 else 0.0
    return total_loss / total_count, total_correct / total_count, confusion, avg_aux_loss


def print_and_log(
        epoch,
        loss,
        acc,
        confusion,
        recovery=None,
        lr=None,
        mode="train",
        lookup=NUM_TO_LETTER,
        aux_loss=None,
    ):
    """Print and log training metrics to wandb."""
    log_str = f"\nEPOCH {epoch} {mode.upper()} loss: {loss:.4f} perp: {np.exp(loss):.4f} acc: {acc:.4f}"
    wandb_metrics = {
        f"{mode}/loss": loss,
        f"{mode}/perp": np.exp(loss),
        f"{mode}/acc": acc,
        "epoch": epoch
    }

    if lr is not None:
        log_str += f" lr: {lr:.6f}"
        wandb_metrics["lr"] = lr

    if recovery is not None:
        log_str += f" rec: {np.mean(recovery):.4f}"
        wandb_metrics[f"{mode}/recovery"] = np.mean(recovery)

    if aux_loss is not None:
        log_str += f" aux_loss: {aux_loss:.6f}"
        wandb_metrics[f"{mode}/aux_loss"] = aux_loss

    print(log_str)
    print_confusion(confusion, lookup=lookup)
    wandb.log(wandb_metrics)


def print_confusion(mat, lookup):
    """Print confusion matrix for nucleotide predictions."""
    counts = mat.astype(np.int32)
    mat = (counts.T / counts.sum(axis=-1, keepdims=True).T).T
    mat = np.round(mat * 1000).astype(np.int32)
    res = '\n'
    for i in range(len(lookup.keys())):
        res += '\t{}'.format(lookup[i])
    res += '\tCount\n'
    for i in range(len(lookup.keys())):
        res += '{}\t'.format(lookup[i])
        res += '\t'.join('{}'.format(n) for n in mat[i])
        res += '\t{}\n'.format(sum(counts[i]))
    print(res)

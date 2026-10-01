"""The training loop, shared by all ten runs.

One function, called once per model per input size. It early-stops on
validation loss and writes a checkpoint carrying the full history and the
configuration, so notebooks 03 to 05 can review a run without repeating it.

The checkpoint schema is fixed, because notebook 03 reads it:

    model_name        str
    epoch             int, the epoch whose weights are stored
    model_state_dict  weights at the best epoch
    val_loss          best validation loss
    val_dice          validation Dice at the best epoch
    history           train_loss, val_loss, train_dice, val_dice, lr, epoch_time
    config            img_size, batch_size, lr, input_channels, params,
                      loss_function, early_stopped_at, patience, pp_kernel,
                      erosion_kernel

Post-processing is NOT applied here. It belongs to evaluation, so training and
post-processing stay separable and the effect of each can be reported on its own.
"""

import copy
import time

import torch


def train_model(model, train_loader, val_loader, criterion, optimizer,
                scheduler, max_epochs, patience, save_path, model_name,
                device, pp_kernel, erosion_kernel, batch_size, learning_rate,
                input_channels, img_size):
    """Train one model, early-stop on validation loss, save the best epoch."""

    history = {'train_loss': [], 'val_loss': [], 'train_dice': [],
               'val_dice': [], 'lr': [], 'epoch_time': []}

    parameter_count = 0
    for p in model.parameters():
        parameter_count += p.numel()

    best_val_loss = float('inf')
    best_val_dice = 0.0
    best_epoch = 0
    best_weights = copy.deepcopy(model.state_dict())
    no_improve = 0
    stopped_at = max_epochs

    print('%s | %s | %d parameters | %d channels'
          % (model_name, type(criterion).__name__, parameter_count,
             input_channels))
    print('-' * 78)

    for epoch in range(1, max_epochs + 1):
        started = time.time()

        model.train()
        train_loss_total = 0.0
        train_dice_total = 0.0
        train_batches = 0
        for images, masks in train_loader:
            images = images.to(device)
            masks = masks.to(device)
            optimizer.zero_grad()
            predictions = model(images)
            loss, dice = criterion(predictions, masks)
            loss.backward()
            optimizer.step()
            train_loss_total += loss.item()
            train_dice_total += dice
            train_batches += 1

        model.eval()
        val_loss_total = 0.0
        val_dice_total = 0.0
        val_batches = 0
        with torch.no_grad():
            for images, masks in val_loader:
                images = images.to(device)
                masks = masks.to(device)
                predictions = model(images)
                loss, dice = criterion(predictions, masks)
                val_loss_total += loss.item()
                val_dice_total += dice
                val_batches += 1

        train_loss = train_loss_total / train_batches
        train_dice = train_dice_total / train_batches
        val_loss = val_loss_total / val_batches
        val_dice = val_dice_total / val_batches
        current_lr = optimizer.param_groups[0]['lr']
        elapsed = time.time() - started

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_dice'].append(train_dice)
        history['val_dice'].append(val_dice)
        history['lr'].append(current_lr)
        history['epoch_time'].append(elapsed)

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_dice = val_dice
            best_epoch = epoch
            best_weights = copy.deepcopy(model.state_dict())
            no_improve = 0
            marker = '  <- best'
        else:
            no_improve += 1
            marker = ''

        if epoch % 5 == 0 or marker:
            print('epoch %3d | train %.4f dice %.4f | val %.4f dice %.4f | '
                  'lr %.2e | %4.1fs%s'
                  % (epoch, train_loss, train_dice, val_loss, val_dice,
                     current_lr, elapsed, marker))

        if no_improve >= patience:
            stopped_at = epoch
            print('early stopping at epoch %d, no improvement for %d epochs'
                  % (epoch, patience))
            break

    model.load_state_dict(best_weights)

    torch.save({
        'model_name': model_name,
        'epoch': best_epoch,
        'model_state_dict': best_weights,
        'val_loss': best_val_loss,
        'val_dice': best_val_dice,
        'history': history,
        'config': {
            'img_size': img_size,
            'batch_size': batch_size,
            'lr': learning_rate,
            'input_channels': input_channels,
            'params': parameter_count,
            'loss_function': type(criterion).__name__,
            'early_stopped_at': stopped_at,
            'patience': patience,
            'pp_kernel': pp_kernel,
            'erosion_kernel': erosion_kernel,
        },
    }, save_path)

    total_minutes = sum(history['epoch_time']) / 60.0
    print('-' * 78)
    print('best epoch %d | val loss %.4f | val dice %.4f | %d epochs in %.1f min'
          % (best_epoch, best_val_loss, best_val_dice, len(history['train_loss']),
             total_minutes))
    print('saved %s' % save_path)
    print()

    return history
